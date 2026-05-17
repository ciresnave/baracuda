//! `index_select` plan — Category L.
//!
//! `out[..., j, ...] = src[..., idx[j], ...]` along the `select_dim`
//! axis. `idx` is a 1-D `i32` tensor; output shape == source shape
//! with `select_dim` replaced by `idx.numel()`. PyTorch
//! `torch.index_select`.
//!
//! Faster / simpler than [`crate::indexing::GatherPlan`] when the index
//! tensor is 1-D — `gather` accepts an N-D index broadcast to the
//! output shape, while `index_select` collapses to a single 1-D lookup.
//!
//! Trailblazer dtype coverage: `f32, f64, i32`. The kernel does no
//! arithmetic — pure load + store — so output is bit-exact at every
//! dtype.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, IndexingKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for an `index_select` op.
#[derive(Copy, Clone, Debug)]
pub struct IndexSelectDescriptor<const N: usize> {
    /// Output tensor shape.
    pub out_shape: [i32; N],
    /// Axis along which to select. Must be in `[0, N)`.
    pub select_dim: i32,
    /// Extent of `src` along `select_dim` (bounds check on indices).
    pub src_dim_size: i32,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for an `index_select` launch.
pub struct IndexSelectArgs<'a, T: Element, const N: usize> {
    /// Source tensor.
    pub src: TensorRef<'a, T, N>,
    /// Index tensor (1-D, i32). `idx.numel()` must equal
    /// `out_shape[select_dim]`.
    pub idx: TensorRef<'a, i32, 1>,
    /// Output. Shape == descriptor `out_shape`.
    pub out: TensorMut<'a, T, N>,
}

/// `index_select` plan.
pub struct IndexSelectPlan<T: Element, const N: usize> {
    desc: IndexSelectDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> IndexSelectPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &IndexSelectDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::IndexSelectPlan: descriptor element != type parameter T",
            ));
        }
        if N == 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IndexSelectPlan: rank-0 tensors not supported",
            ));
        }
        if desc.select_dim < 0 || desc.select_dim >= N as i32 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IndexSelectPlan: select_dim out of range [0, N)",
            ));
        }
        if desc.src_dim_size < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IndexSelectPlan: src_dim_size must be non-negative",
            ));
        }

        let supported =
            matches!(T::KIND, ElementKind::F32 | ElementKind::F64 | ElementKind::I32);
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::IndexSelectPlan: today only `f32`, `f64`, `i32` wired",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Indexing,
            op: IndexingKind::IndexSelect as u16,
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
    pub fn can_implement(&self, args: &IndexSelectArgs<'_, T, N>) -> Result<()> {
        if args.out.shape != self.desc.out_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IndexSelectPlan: out shape mismatch with descriptor",
            ));
        }
        let expected_idx = self.desc.out_shape[self.desc.select_dim as usize];
        if args.idx.shape[0] != expected_idx {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IndexSelectPlan: idx.shape[0] must equal \
                 out_shape[select_dim]",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::IndexSelectPlan: tensor rank > 8 not supported",
            ));
        }
        let out_numel = args.out.numel();
        let idx_numel = args.idx.numel();
        let out_len = args.out.data.len() as i64;
        let idx_len = args.idx.data.len() as i64;
        if out_len < out_numel {
            return Err(Error::BufferTooSmall {
                needed: out_numel as usize,
                got: out_len as usize,
            });
        }
        if idx_len < idx_numel {
            return Err(Error::BufferTooSmall {
                needed: idx_numel as usize,
                got: idx_len as usize,
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
        args: IndexSelectArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let out_numel = args.out.numel();
        if out_numel == 0 {
            return Ok(());
        }
        let src_ptr = args.src.data.as_raw().0 as *const c_void;
        let idx_ptr = args.idx.data.as_raw().0 as *const c_void;
        let out_ptr = args.out.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let out_shape = self.desc.out_shape;
        let stride_src = args.src.stride;
        let stride_out = args.out.stride;
        let rank = N as i32;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_index_select_f32_run(
                    out_numel,
                    rank,
                    self.desc.select_dim,
                    self.desc.src_dim_size,
                    out_shape.as_ptr(),
                    stride_src.as_ptr(),
                    stride_out.as_ptr(),
                    src_ptr,
                    idx_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_index_select_f64_run(
                    out_numel,
                    rank,
                    self.desc.select_dim,
                    self.desc.src_dim_size,
                    out_shape.as_ptr(),
                    stride_src.as_ptr(),
                    stride_out.as_ptr(),
                    src_ptr,
                    idx_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_index_select_i32_run(
                    out_numel,
                    rank,
                    self.desc.select_dim,
                    self.desc.src_dim_size,
                    out_shape.as_ptr(),
                    stride_src.as_ptr(),
                    stride_out.as_ptr(),
                    src_ptr,
                    idx_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::IndexSelectPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        super::gather::map_status(status)
    }
}
