//! `nonzero` plan — Category L (non-differentiable).
//!
//! Returns the coordinates where the input is non-zero. Output is a
//! flat `[max_nz, rank]` i32 coordinate table plus a single-element
//! i32 counter that the launcher zeros via `cudaMemsetAsync` before
//! the kernel runs.
//!
//! Trailblazer dtype coverage: `f32, f64, i32, bool` input. Output is
//! always i32 coordinates.
//!
//! **Output ordering caveat**: the kernel uses a single global atomic
//! counter for slot assignment, so the output is NOT row-major (rather
//! the natural CUDA-block race order). PyTorch's `torch.nonzero`
//! returns row-major-sorted coords; callers that need that ordering
//! should sort the `[count, rank]` rows afterward. A future milestone
//! may replace the atomic counter with a two-pass prefix-sum kernel
//! that preserves row-major order.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, IndexingKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::gather::map_status;

/// Descriptor for a `nonzero` op.
#[derive(Copy, Clone, Debug)]
pub struct NonzeroDescriptor<const N: usize> {
    /// Input tensor shape.
    pub shape: [i32; N],
    /// Maximum number of nonzero coords the output table can hold.
    /// Caller-supplied upper bound — coords past this count are
    /// discarded by the kernel. Set to `numel` for a worst-case-safe
    /// bound.
    pub max_nz: i32,
    /// Input element type.
    pub element: ElementKind,
}

/// Args bundle for a `nonzero` launch.
pub struct NonzeroArgs<'a, T: Element, const N: usize> {
    /// Input tensor.
    pub x: TensorRef<'a, T, N>,
    /// Output coordinate table — 1-D flat i32 view of a logical
    /// `[max_nz, rank]` table. Length must be at least
    /// `max_nz * rank` cells.
    pub out_coords: TensorMut<'a, i32, 1>,
    /// Single-element i32 counter (launcher zeros it on the stream
    /// before the kernel runs; after the run, the counter holds the
    /// actual number of nonzero coords, which may be ≥ max_nz if some
    /// were discarded).
    pub counter: TensorMut<'a, i32, 1>,
}

/// `nonzero` plan.
pub struct NonzeroPlan<T: Element, const N: usize> {
    desc: NonzeroDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> NonzeroPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &NonzeroDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::NonzeroPlan: descriptor element != type parameter T",
            ));
        }
        if N == 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::NonzeroPlan: rank-0 tensors not supported",
            ));
        }
        if desc.max_nz < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::NonzeroPlan: max_nz must be non-negative",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::NonzeroPlan: shape dims must be non-negative",
                ));
            }
        }

        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::I32 | ElementKind::Bool
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::NonzeroPlan: today only input dtypes \
                 `f32`, `f64`, `i32`, `bool` wired",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Atomic-counter ordering is non-deterministic across
            // launches — but the *content* of the output (set of
            // nonzero coordinates) is fully determined by the input.
            bit_stable_on_same_hardware: false,
            deterministic: false,
        };
        let sku = KernelSku {
            category: OpCategory::Indexing,
            op: IndexingKind::Nonzero as u16,
            element: T::KIND,
            aux_element: Some(ElementKind::I32),
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::Bespoke,
            precision_guarantee,
        };
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &NonzeroArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::NonzeroPlan: x shape mismatch with descriptor",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::NonzeroPlan: tensor rank > 8 not supported",
            ));
        }
        let max_nz = self.desc.max_nz as i64;
        let rank = N as i64;
        let needed_coords = max_nz.saturating_mul(rank);
        let coords_len = args.out_coords.data.len() as i64;
        let counter_len = args.counter.data.len() as i64;
        if coords_len < needed_coords {
            return Err(Error::BufferTooSmall {
                needed: needed_coords as usize,
                got: coords_len as usize,
            });
        }
        if counter_len < 1 {
            return Err(Error::BufferTooSmall {
                needed: 1,
                got: counter_len as usize,
            });
        }
        let x_numel = args.x.numel();
        let x_len = args.x.data.len() as i64;
        if x_len < x_numel {
            return Err(Error::BufferTooSmall {
                needed: x_numel as usize,
                got: x_len as usize,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel this plan picked.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees for this plan's kernel.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: NonzeroArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.x.numel();
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let coords_ptr = args.out_coords.data.as_raw().0 as *mut c_void;
        let counter_ptr = args.counter.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let shape = self.desc.shape;
        let stride_x = args.x.stride;
        let rank = N as i32;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_nonzero_f32_run(
                    numel,
                    rank,
                    self.desc.max_nz,
                    shape.as_ptr(),
                    stride_x.as_ptr(),
                    x_ptr,
                    coords_ptr,
                    counter_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_nonzero_f64_run(
                    numel,
                    rank,
                    self.desc.max_nz,
                    shape.as_ptr(),
                    stride_x.as_ptr(),
                    x_ptr,
                    coords_ptr,
                    counter_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_nonzero_i32_run(
                    numel,
                    rank,
                    self.desc.max_nz,
                    shape.as_ptr(),
                    stride_x.as_ptr(),
                    x_ptr,
                    coords_ptr,
                    counter_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bool => unsafe {
                baracuda_kernels_sys::baracuda_kernels_nonzero_bool_run(
                    numel,
                    rank,
                    self.desc.max_nz,
                    shape.as_ptr(),
                    stride_x.as_ptr(),
                    x_ptr,
                    coords_ptr,
                    counter_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::NonzeroPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}
