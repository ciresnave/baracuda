//! `index_select_backward` plan — Category L BW.
//!
//! Adjoint of [`crate::indexing::IndexSelectPlan`]:
//! `dsrc[..., idx[j], ...] += dout[..., j, ...]` along the `select_dim`
//! axis (atomicAdd). Trailblazer dtype coverage: `f32, f64`.
//!
//! Caller must zero `dsrc` before launch (or pre-populate to accumulate).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, IndexElement, IndexElementKind, IndexingKind,
    KernelSku, MathPrecision, OpCategory, PlanPreference, PrecisionGuarantee, TensorMut,
    TensorRef, Workspace,
};

use super::gather::map_status;

/// Descriptor for an `index_select_backward` op.
#[derive(Copy, Clone, Debug)]
pub struct IndexSelectBackwardDescriptor<const N: usize> {
    /// Shape of `dout` (== FW output shape).
    pub out_shape: [i32; N],
    /// Select axis (must match the FW pass).
    pub select_dim: i32,
    /// Extent of `dsrc` along `select_dim`.
    pub src_dim_size: i32,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for an `index_select_backward` launch.
///
/// Phase 11.5: `I: IndexElement` generic (`i32` or `i64`).
pub struct IndexSelectBackwardArgs<'a, T: Element, const N: usize, I: IndexElement = i32> {
    /// Upstream gradient.
    pub dout: TensorRef<'a, T, N>,
    /// Index tensor (1-D) from the FW pass.
    pub idx: TensorRef<'a, I, 1>,
    /// Gradient w.r.t. `src`. Caller MUST zero this before launch.
    pub dsrc: TensorMut<'a, T, N>,
}

/// `index_select_backward` plan.
///
/// Adjoint of [`crate::IndexSelectPlan`]: scatter-adds `dout` into
/// `dsrc` at the rows pointed to by `idx`.
///
/// **When to use**: backward for [`IndexSelectPlan`](crate::IndexSelectPlan).
///
/// **Dtypes**: `{f32, f64}` only (uses `atomicAdd`).
///
/// **Shape limits**: rank in `[1, 8]`; `select_dim ∈ [0, N)`;
/// `idx` is 1-D `i32` with `idx.numel() == out_shape[select_dim]`.
///
/// **Workspace**: none. Caller MUST zero `dsrc` before launch.
///
/// **Precision guarantee**: **non-deterministic** — atomicAdd
/// ordering varies across launches.
pub struct IndexSelectBackwardPlan<T: Element, const N: usize> {
    desc: IndexSelectBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> IndexSelectBackwardPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &IndexSelectBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::IndexSelectBackwardPlan: descriptor element != type parameter T",
            ));
        }
        if N == 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IndexSelectBackwardPlan: rank-0 tensors not supported",
            ));
        }
        if desc.select_dim < 0 || desc.select_dim >= N as i32 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IndexSelectBackwardPlan: select_dim out of range [0, N)",
            ));
        }
        if desc.src_dim_size < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IndexSelectBackwardPlan: src_dim_size must be non-negative",
            ));
        }

        let supported = matches!(T::KIND, ElementKind::F32 | ElementKind::F64);
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::IndexSelectBackwardPlan: today only `f32`, `f64` wired \
                 (BW uses atomicAdd)",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: if T::KIND == ElementKind::F64 {
                MathPrecision::F64
            } else {
                MathPrecision::F32
            },
            accumulator: T::KIND,
            bit_stable_on_same_hardware: false,
            deterministic: false,
        };
        let sku = KernelSku {
            category: OpCategory::Indexing,
            op: IndexingKind::IndexSelectBackward as u16,
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
    pub fn can_implement<I: IndexElement>(&self, args: &IndexSelectBackwardArgs<'_, T, N, I>) -> Result<()> {
        if args.dout.shape != self.desc.out_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IndexSelectBackwardPlan: dout shape mismatch with descriptor",
            ));
        }
        let expected_idx = self.desc.out_shape[self.desc.select_dim as usize];
        if args.idx.shape[0] != expected_idx {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IndexSelectBackwardPlan: idx.shape[0] must equal \
                 out_shape[select_dim]",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::IndexSelectBackwardPlan: tensor rank > 8 not supported",
            ));
        }
        let out_numel = args.dout.numel();
        let dout_len = args.dout.data.len() as i64;
        let idx_len = args.idx.data.len() as i64;
        let idx_numel = args.idx.numel();
        if dout_len < out_numel {
            return Err(Error::BufferTooSmall {
                needed: out_numel as usize,
                got: dout_len as usize,
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
    ///
    /// Phase 11.5: generic over `I: IndexElement`.
    pub fn run<I: IndexElement>(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: IndexSelectBackwardArgs<'_, T, N, I>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let out_numel = args.dout.numel();
        if out_numel == 0 {
            return Ok(());
        }
        let dout_ptr = args.dout.data.as_raw().0 as *const c_void;
        let idx_ptr = args.idx.data.as_raw().0 as *const c_void;
        let dsrc_ptr = args.dsrc.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let out_shape = self.desc.out_shape;
        let stride_dout = args.dout.stride;
        let stride_dsrc = args.dsrc.stride;
        let rank = N as i32;

        let status = match (T::KIND, I::KIND) {
            (ElementKind::F32, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_index_select_backward_f32_run(
                    out_numel, rank, self.desc.select_dim, self.desc.src_dim_size,
                    out_shape.as_ptr(), stride_dout.as_ptr(), stride_dsrc.as_ptr(),
                    dout_ptr, idx_ptr, dsrc_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_index_select_backward_f64_run(
                    out_numel, rank, self.desc.select_dim, self.desc.src_dim_size,
                    out_shape.as_ptr(), stride_dout.as_ptr(), stride_dsrc.as_ptr(),
                    dout_ptr, idx_ptr, dsrc_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_index_select_backward_i64idx_f32_run(
                    out_numel, rank, self.desc.select_dim, self.desc.src_dim_size,
                    out_shape.as_ptr(), stride_dout.as_ptr(), stride_dsrc.as_ptr(),
                    dout_ptr, idx_ptr, dsrc_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_index_select_backward_i64idx_f64_run(
                    out_numel, rank, self.desc.select_dim, self.desc.src_dim_size,
                    out_shape.as_ptr(), stride_dout.as_ptr(), stride_dsrc.as_ptr(),
                    dout_ptr, idx_ptr, dsrc_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::IndexSelectBackwardPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}
