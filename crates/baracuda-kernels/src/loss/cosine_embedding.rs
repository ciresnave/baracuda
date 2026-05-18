//! CosineEmbedding loss plan — Tier-2 (Milestone 5.3).
//!
//! **Formula** (per row):
//!   `cs = (x1·x2) / (||x1|| · ||x2||)`
//!   `term = (1 - cs)` if `t == 1` else `max(0, cs - margin)`
//!
//! **When to use**: similarity learning, e.g. siamese networks. Pair
//! with [`CosineEmbeddingLossBackwardPlan`] for autograd.
//!
//! **Dtypes / shape**: `{f32, f16, bf16, f64}`. 2-D inputs `[N, D]`,
//! target `[N]` encoded as `[N, 1]` (PyTorch uses ±1.0). Output `[N]`
//! for `None` mode, `[1]` for `Mean` / `Sum`.
//!
//! **Workspace**: `n_rows · sizeof(T)` for `Mean` / `Sum`; zero for
//! `None` (per-cell kernel writes directly to output).
//!
//! **Precision**: deterministic, bit-stable. f16 / bf16 accumulate in
//! f32 (FP detour); f64 keeps everything in double.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LossKind, LossReduction, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::common::{check_supported_dtype, map_status, unpack_workspace};

/// Descriptor for a CosineEmbedding loss op.
#[derive(Copy, Clone, Debug)]
pub struct CosineEmbeddingLossDescriptor {
    /// Number of rows N (batch).
    pub n_rows: i32,
    /// D extent (feature dim).
    pub d_extent: i32,
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Margin (PyTorch default 0.0).
    pub margin: f32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a CosineEmbedding FW launch.
pub struct CosineEmbeddingLossArgs<'a, T: Element> {
    /// First input tensor, shape [N, D].
    pub x1: TensorRef<'a, T, 2>,
    /// Second input tensor, shape [N, D].
    pub x2: TensorRef<'a, T, 2>,
    /// Target tensor, shape [N] (encoded as [N, 1]).
    pub t: TensorRef<'a, T, 2>,
    /// Output: [N] (None) or [1] (Mean/Sum), encoded as [N, 1] / [1, 1].
    pub out: TensorMut<'a, T, 2>,
}

/// CosineEmbedding forward plan.
pub struct CosineEmbeddingLossPlan<T: Element> {
    desc: CosineEmbeddingLossDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> CosineEmbeddingLossPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &CosineEmbeddingLossDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::CosineEmbeddingLossPlan: descriptor element != T",
            ));
        }
        if desc.n_rows < 0 || desc.d_extent < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CosineEmbeddingLossPlan: n_rows / d_extent must be ≥ 0",
            ));
        }
        check_supported_dtype::<T>()?;
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: if T::KIND == ElementKind::F64 {
                ElementKind::F64
            } else {
                ElementKind::F32
            },
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Loss,
            op: LossKind::CosineEmbedding as u16,
            element: T::KIND,
            aux_element: None,
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
    /// Workspace size in bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        match self.desc.reduction {
            LossReduction::None => 0,
            LossReduction::Mean | LossReduction::Sum => {
                (self.desc.n_rows as usize).saturating_mul(core::mem::size_of::<T>())
            }
        }
    }
    /// Kernel SKU identity.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }
    /// Numerical guarantees.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: CosineEmbeddingLossArgs<'_, T>,
    ) -> Result<()> {
        if self.desc.n_rows == 0 {
            return Ok(());
        }
        let (ws_ptr, ws_bytes) = unpack_workspace(workspace, self.workspace_size())?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let x1_ptr = args.x1.data.as_raw().0 as *const c_void;
        let x2_ptr = args.x2.data.as_raw().0 as *const c_void;
        let t_ptr = args.t.data.as_raw().0 as *const c_void;
        let out_ptr = args.out.data.as_raw().0 as *mut c_void;
        let mode = self.desc.reduction as i32;
        let margin = self.desc.margin;
        let n_rows = self.desc.n_rows as i64;
        let d_extent = self.desc.d_extent;
        let row_stride_x: i64 = d_extent as i64;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_cosine_embedding_f32_run(
                    n_rows, d_extent, row_stride_x, mode, margin, x1_ptr, x2_ptr, t_ptr, out_ptr,
                    ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_cosine_embedding_f16_run(
                    n_rows, d_extent, row_stride_x, mode, margin, x1_ptr, x2_ptr, t_ptr, out_ptr,
                    ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_cosine_embedding_bf16_run(
                    n_rows, d_extent, row_stride_x, mode, margin, x1_ptr, x2_ptr, t_ptr, out_ptr,
                    ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_cosine_embedding_f64_run(
                    n_rows, d_extent, row_stride_x, mode, margin, x1_ptr, x2_ptr, t_ptr, out_ptr,
                    ws_ptr, ws_bytes, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::CosineEmbeddingLossPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}

// =============================================================================
// BACKWARD
// =============================================================================

/// Descriptor for a CosineEmbedding backward op.
#[derive(Copy, Clone, Debug)]
pub struct CosineEmbeddingLossBackwardDescriptor {
    /// Number of rows N.
    pub n_rows: i32,
    /// D extent.
    pub d_extent: i32,
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Margin.
    pub margin: f32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a CosineEmbedding BW launch.
pub struct CosineEmbeddingLossBackwardArgs<'a, T: Element> {
    /// x1 saved from FW.
    pub x1: TensorRef<'a, T, 2>,
    /// x2 saved from FW.
    pub x2: TensorRef<'a, T, 2>,
    /// Target saved from FW.
    pub t: TensorRef<'a, T, 2>,
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, 2>,
    /// Output gradient w.r.t. x1.
    pub dx1: TensorMut<'a, T, 2>,
    /// Output gradient w.r.t. x2.
    pub dx2: TensorMut<'a, T, 2>,
}

/// CosineEmbedding backward plan.
pub struct CosineEmbeddingLossBackwardPlan<T: Element> {
    desc: CosineEmbeddingLossBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> CosineEmbeddingLossBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &CosineEmbeddingLossBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::CosineEmbeddingLossBackwardPlan: descriptor element != T",
            ));
        }
        if desc.n_rows < 0 || desc.d_extent < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CosineEmbeddingLossBackwardPlan: n_rows / d_extent must be ≥ 0",
            ));
        }
        check_supported_dtype::<T>()?;
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: if T::KIND == ElementKind::F64 {
                ElementKind::F64
            } else {
                ElementKind::F32
            },
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Loss,
            op: LossKind::CosineEmbedding as u16,
            element: T::KIND,
            aux_element: None,
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
    /// Workspace size in bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }
    /// Kernel SKU identity.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }
    /// Numerical guarantees.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: CosineEmbeddingLossBackwardArgs<'_, T>,
    ) -> Result<()> {
        if self.desc.n_rows == 0 {
            return Ok(());
        }
        let mode = self.desc.reduction as i32;
        let n_rows = self.desc.n_rows as i64;
        let d_extent = self.desc.d_extent;
        let inv_n_or_one: f32 = match self.desc.reduction {
            LossReduction::None => 0.0,
            LossReduction::Mean => 1.0 / (n_rows as f32),
            LossReduction::Sum => 1.0,
        };
        let margin = self.desc.margin;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let x1_ptr = args.x1.data.as_raw().0 as *const c_void;
        let x2_ptr = args.x2.data.as_raw().0 as *const c_void;
        let t_ptr = args.t.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx1_ptr = args.dx1.data.as_raw().0 as *mut c_void;
        let dx2_ptr = args.dx2.data.as_raw().0 as *mut c_void;
        let row_stride_x: i64 = d_extent as i64;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_cosine_embedding_backward_f32_run(
                    n_rows, d_extent, row_stride_x, mode, inv_n_or_one, margin, x1_ptr, x2_ptr,
                    t_ptr, dy_ptr, dx1_ptr, dx2_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_cosine_embedding_backward_f16_run(
                    n_rows, d_extent, row_stride_x, mode, inv_n_or_one, margin, x1_ptr, x2_ptr,
                    t_ptr, dy_ptr, dx1_ptr, dx2_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_cosine_embedding_backward_bf16_run(
                    n_rows, d_extent, row_stride_x, mode, inv_n_or_one, margin, x1_ptr, x2_ptr,
                    t_ptr, dy_ptr, dx1_ptr, dx2_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_cosine_embedding_backward_f64_run(
                    n_rows, d_extent, row_stride_x, mode, inv_n_or_one, margin, x1_ptr, x2_ptr,
                    t_ptr, dy_ptr, dx1_ptr, dx2_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::CosineEmbeddingLossBackwardPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}
