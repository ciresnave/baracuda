//! `nonzero` plan — Category L (non-differentiable).
//!
//! Returns the coordinates where the input is non-zero. Output is a
//! flat `[max_nz, rank]` coordinate table plus a single-element counter
//! that the launcher zeros via `cudaMemsetAsync` before the kernel runs.
//!
//! Phase 15.2: the output coord dtype is generic over [`IndexElement`]
//! (`i32` or `i64`); the counter shares the same dtype. Default
//! `I = i32` for source-compat with pre-Phase-15.2 callers.
//!
//! Trailblazer dtype coverage: `f32, f64, i32, bool` input.
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
    ArchSku, BackendKind, Element, ElementKind, IndexElement, IndexElementKind, IndexingKind,
    KernelSku, MathPrecision, OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef,
    Workspace,
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
///
/// Phase 15.2: the output coord dtype is generic over `I: IndexElement`
/// (`i32` or `i64`). The counter shares the same dtype. Defaults to
/// `I = i32` for source-compat with pre-Phase-15.2 callers (the i64
/// variant matches PyTorch's `torch.nonzero` convention).
pub struct NonzeroArgs<'a, T: Element, const N: usize, I: IndexElement = i32> {
    /// Input tensor.
    pub x: TensorRef<'a, T, N>,
    /// Output coordinate table — 1-D flat view of a logical
    /// `[max_nz, rank]` table. Length must be at least `max_nz * rank`
    /// cells. Type parameter `I` selects `i32` (legacy) or `i64`
    /// (PyTorch convention).
    pub out_coords: TensorMut<'a, I, 1>,
    /// Single-element counter (launcher zeros it on the stream before
    /// the kernel runs; after the run, the counter holds the actual
    /// number of nonzero coords, which may be ≥ max_nz if some were
    /// discarded). Shares the index dtype `I` with `out_coords`.
    pub counter: TensorMut<'a, I, 1>,
}

/// `nonzero` plan.
///
/// Emits the multi-index coordinates of every nonzero element of `x`
/// into a flat `[max_nz, rank]` index-dtype table. Non-differentiable.
///
/// **When to use**: extract sparse indices from a dense tensor (mask
/// → coords, sparse-from-dense conversion).
///
/// **Dtypes**: input `{f32, f64, i32, bool}`; output coords `{i32, i64}`
/// (generic over [`IndexElement`], default `i32` for source-compat).
///
/// **Shape limits**: input rank `N ∈ [1, 8]`. Output table length
/// must be ≥ `max_nz * N`. Caller picks `max_nz` (set to `numel`
/// for the worst-case-safe bound).
///
/// **Workspace**: none, but caller must (a) provide an `out_coords`
/// table of at least `max_nz * rank` cells and (b) provide a 1-cell
/// `counter` tensor which the launcher zeros via `cudaMemsetAsync`
/// before the kernel runs. After the launch, the counter holds the
/// true count of nonzeros (may exceed `max_nz` — overflow coords
/// are discarded).
///
/// **Precision guarantee**: **non-deterministic ordering** — the
/// kernel uses a single global atomic counter for slot assignment,
/// so coordinates appear in CUDA-block race order (not row-major).
/// The *set* of coords is fully determined by the input; only the
/// row order varies. Callers needing PyTorch-style row-major coords
/// should sort the `[count, rank]` rows afterward.
///
/// **Known limitations**: a future milestone may replace the atomic
/// counter with a two-pass prefix-sum kernel that preserves row-major
/// order.
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
    ///
    /// Phase 15.2: generic over `I: IndexElement` so both i32 (legacy)
    /// and i64 (PyTorch default) output-coord buffers are accepted.
    pub fn can_implement<I: IndexElement>(&self, args: &NonzeroArgs<'_, T, N, I>) -> Result<()> {
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
    ///
    /// Phase 15.2: generic over `I: IndexElement`. Dispatches to the
    /// matching `_i64idx_` FFI symbol when
    /// `I::KIND == IndexElementKind::I64` (the i64-output-coords
    /// variant); default `I = i32` keeps the legacy path.
    pub fn run<I: IndexElement>(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: NonzeroArgs<'_, T, N, I>,
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

        let status = match (T::KIND, I::KIND) {
            (ElementKind::F32, IndexElementKind::I32) => unsafe {
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
            (ElementKind::F64, IndexElementKind::I32) => unsafe {
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
            (ElementKind::I32, IndexElementKind::I32) => unsafe {
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
            (ElementKind::Bool, IndexElementKind::I32) => unsafe {
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
            (ElementKind::F32, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_nonzero_i64idx_f32_run(
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
            (ElementKind::F64, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_nonzero_i64idx_f64_run(
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
            (ElementKind::I32, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_nonzero_i64idx_i32_run(
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
            (ElementKind::Bool, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_nonzero_i64idx_bool_run(
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
