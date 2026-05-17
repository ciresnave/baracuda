//! MarginRanking loss plan — Tier-2 (Milestone 5.3).
//!
//! FW: `y = mean(max(0, -t · (x1 - x2) + margin))` (or sum / per-cell).
//! BW: `dx1[i] = -t[i]/N · dy if loss_i > 0 else 0`, `dx2 = -dx1`.
//!
//! Target `t` is of type `T` (PyTorch uses ±1.0 as floats here).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LossKind, LossReduction, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::common::{check_supported_dtype, map_status, unpack_workspace, validate_shape};

/// Descriptor for a MarginRanking loss op.
#[derive(Copy, Clone, Debug)]
pub struct MarginRankingLossDescriptor<const N: usize> {
    /// Input / target tensor shape (all three operands share this).
    pub input_shape: [i32; N],
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Margin scalar (PyTorch default 0.0).
    pub margin: f32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a MarginRanking FW launch.
pub struct MarginRankingLossArgs<'a, T: Element, const N: usize> {
    /// First operand tensor.
    pub x1: TensorRef<'a, T, N>,
    /// Second operand tensor.
    pub x2: TensorRef<'a, T, N>,
    /// Target tensor (±1.0).
    pub t: TensorRef<'a, T, N>,
    /// Output: per-cell for None mode, scalar for Mean / Sum.
    pub out: TensorMut<'a, T, N>,
}

/// MarginRanking forward plan.
pub struct MarginRankingLossPlan<T: Element, const N: usize> {
    desc: MarginRankingLossDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> MarginRankingLossPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &MarginRankingLossDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::MarginRankingLossPlan: descriptor element != T",
            ));
        }
        check_supported_dtype::<T>()?;
        validate_shape(&desc.input_shape, N)?;
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
            op: LossKind::MarginRanking as u16,
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
    /// Workspace size in bytes — `numel · sizeof(T)` for Mean/Sum; 0 for None.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        match self.desc.reduction {
            LossReduction::None => 0,
            LossReduction::Mean | LossReduction::Sum => {
                let mut numel: i64 = 1;
                for &d in self.desc.input_shape.iter() {
                    numel = numel.saturating_mul(d as i64);
                }
                (numel as usize).saturating_mul(core::mem::size_of::<T>())
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
        args: MarginRankingLossArgs<'_, T, N>,
    ) -> Result<()> {
        if args.x1.shape != self.desc.input_shape
            || args.x2.shape != self.desc.input_shape
            || args.t.shape != self.desc.input_shape
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::MarginRankingLossPlan: shape mismatch",
            ));
        }
        let numel = args.x1.numel();
        if numel == 0 {
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
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_margin_ranking_f32_run(
                    numel, mode, margin, x1_ptr, x2_ptr, t_ptr, out_ptr, ws_ptr, ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_margin_ranking_f16_run(
                    numel, mode, margin, x1_ptr, x2_ptr, t_ptr, out_ptr, ws_ptr, ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_margin_ranking_bf16_run(
                    numel, mode, margin, x1_ptr, x2_ptr, t_ptr, out_ptr, ws_ptr, ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_margin_ranking_f64_run(
                    numel, mode, margin, x1_ptr, x2_ptr, t_ptr, out_ptr, ws_ptr, ws_bytes,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::MarginRankingLossPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}

// =============================================================================
// BACKWARD
// =============================================================================

/// Descriptor for a MarginRanking backward op.
#[derive(Copy, Clone, Debug)]
pub struct MarginRankingLossBackwardDescriptor<const N: usize> {
    /// Input shape (all three operands).
    pub input_shape: [i32; N],
    /// Reduction mode used in FW.
    pub reduction: LossReduction,
    /// Margin scalar.
    pub margin: f32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a MarginRanking BW launch.
pub struct MarginRankingLossBackwardArgs<'a, T: Element, const N: usize> {
    /// x1 saved from FW.
    pub x1: TensorRef<'a, T, N>,
    /// x2 saved from FW.
    pub x2: TensorRef<'a, T, N>,
    /// target saved from FW.
    pub t: TensorRef<'a, T, N>,
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// Output: gradient w.r.t. x1.
    pub dx1: TensorMut<'a, T, N>,
    /// Output: gradient w.r.t. x2.
    pub dx2: TensorMut<'a, T, N>,
}

/// MarginRanking backward plan.
pub struct MarginRankingLossBackwardPlan<T: Element, const N: usize> {
    desc: MarginRankingLossBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> MarginRankingLossBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &MarginRankingLossBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::MarginRankingLossBackwardPlan: descriptor element != T",
            ));
        }
        check_supported_dtype::<T>()?;
        validate_shape(&desc.input_shape, N)?;
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
            op: LossKind::MarginRanking as u16,
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
        args: MarginRankingLossBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        if args.x1.shape != self.desc.input_shape
            || args.x2.shape != self.desc.input_shape
            || args.t.shape != self.desc.input_shape
            || args.dx1.shape != self.desc.input_shape
            || args.dx2.shape != self.desc.input_shape
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::MarginRankingLossBackwardPlan: shape mismatch",
            ));
        }
        let numel = args.x1.numel();
        if numel == 0 {
            return Ok(());
        }
        let mode = self.desc.reduction as i32;
        let inv_n_or_one: f32 = match self.desc.reduction {
            LossReduction::None => 0.0,
            LossReduction::Mean => 1.0 / (numel as f32),
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
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_margin_ranking_backward_f32_run(
                    numel, mode, inv_n_or_one, margin, x1_ptr, x2_ptr, t_ptr, dy_ptr, dx1_ptr,
                    dx2_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_margin_ranking_backward_f16_run(
                    numel, mode, inv_n_or_one, margin, x1_ptr, x2_ptr, t_ptr, dy_ptr, dx1_ptr,
                    dx2_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_margin_ranking_backward_bf16_run(
                    numel, mode, inv_n_or_one, margin, x1_ptr, x2_ptr, t_ptr, dy_ptr, dx1_ptr,
                    dx2_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_margin_ranking_backward_f64_run(
                    numel, mode, inv_n_or_one, margin, x1_ptr, x2_ptr, t_ptr, dy_ptr, dx1_ptr,
                    dx2_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::MarginRankingLossBackwardPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}
