//! SmoothL1 loss plan — piecewise:
//! `0.5·(x/β)²·β` if `|x|<β`, else `|x| - 0.5·β`, where `x = pred - target`.
//!
//! BW: `dpred = (x/β)` if `|x|<β`, else `sign(x)`, then `· scale`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LossKind, LossReduction, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::common::{check_supported_dtype, map_status, unpack_workspace, validate_shape};

/// Descriptor for a SmoothL1 loss op.
#[derive(Copy, Clone, Debug)]
pub struct SmoothL1LossDescriptor<const N: usize> {
    /// Input / target tensor shape (must match).
    pub input_shape: [i32; N],
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Smoothing parameter (PyTorch default 1.0).
    pub beta: f32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a SmoothL1 FW launch.
pub struct SmoothL1LossArgs<'a, T: Element, const N: usize> {
    /// Predictions.
    pub pred: TensorRef<'a, T, N>,
    /// Targets.
    pub target: TensorRef<'a, T, N>,
    /// Output.
    pub out: TensorMut<'a, T, N>,
}

/// SmoothL1 forward plan.
pub struct SmoothL1LossPlan<T: Element, const N: usize> {
    desc: SmoothL1LossDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> SmoothL1LossPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &SmoothL1LossDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::SmoothL1LossPlan: descriptor element != T",
            ));
        }
        if !(desc.beta > 0.0 && desc.beta.is_finite()) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SmoothL1LossPlan: beta must be > 0 and finite",
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
            op: LossKind::SmoothL1 as u16,
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
        args: SmoothL1LossArgs<'_, T, N>,
    ) -> Result<()> {
        if args.pred.shape != self.desc.input_shape || args.target.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SmoothL1LossPlan: pred / target shape mismatch",
            ));
        }
        let numel = args.pred.numel();
        if numel == 0 {
            return Ok(());
        }
        let (ws_ptr, ws_bytes) = unpack_workspace(workspace, self.workspace_size())?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let pred_ptr = args.pred.data.as_raw().0 as *const c_void;
        let target_ptr = args.target.data.as_raw().0 as *const c_void;
        let out_ptr = args.out.data.as_raw().0 as *mut c_void;
        let mode = self.desc.reduction as i32;
        let beta = self.desc.beta;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_smooth_l1_f32_run(
                    numel, mode, beta, pred_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_smooth_l1_f16_run(
                    numel, mode, beta, pred_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_smooth_l1_bf16_run(
                    numel, mode, beta, pred_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_smooth_l1_f64_run(
                    numel, mode, beta, pred_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::SmoothL1LossPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}

// =============================================================================
// BACKWARD
// =============================================================================

/// Descriptor for a SmoothL1 backward op.
#[derive(Copy, Clone, Debug)]
pub struct SmoothL1LossBackwardDescriptor<const N: usize> {
    /// Input / target tensor shape (must match dpred).
    pub input_shape: [i32; N],
    /// Reduction mode used in the forward.
    pub reduction: LossReduction,
    /// Smoothing parameter (must match the forward's value).
    pub beta: f32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a SmoothL1 BW launch.
pub struct SmoothL1LossBackwardArgs<'a, T: Element, const N: usize> {
    /// Predictions (saved from FW).
    pub pred: TensorRef<'a, T, N>,
    /// Targets (saved from FW).
    pub target: TensorRef<'a, T, N>,
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// Gradient w.r.t. predictions.
    pub dpred: TensorMut<'a, T, N>,
}

/// SmoothL1 backward plan.
pub struct SmoothL1LossBackwardPlan<T: Element, const N: usize> {
    desc: SmoothL1LossBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> SmoothL1LossBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &SmoothL1LossBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::SmoothL1LossBackwardPlan: descriptor element != T",
            ));
        }
        if !(desc.beta > 0.0 && desc.beta.is_finite()) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SmoothL1LossBackwardPlan: beta must be > 0 and finite",
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
            op: LossKind::SmoothL1 as u16,
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
        args: SmoothL1LossBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        if args.pred.shape != self.desc.input_shape
            || args.target.shape != self.desc.input_shape
            || args.dpred.shape != self.desc.input_shape
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SmoothL1LossBackwardPlan: shape mismatch",
            ));
        }
        let numel = args.pred.numel();
        if numel == 0 {
            return Ok(());
        }
        let mode = self.desc.reduction as i32;
        let inv_n_or_one: f32 = match self.desc.reduction {
            LossReduction::None => 0.0,
            LossReduction::Mean => 1.0 / (numel as f32),
            LossReduction::Sum => 1.0,
        };
        let beta = self.desc.beta;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let pred_ptr = args.pred.data.as_raw().0 as *const c_void;
        let target_ptr = args.target.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dpred_ptr = args.dpred.data.as_raw().0 as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_smooth_l1_backward_f32_run(
                    numel, mode, inv_n_or_one, beta, pred_ptr, target_ptr, dy_ptr, dpred_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_smooth_l1_backward_f16_run(
                    numel, mode, inv_n_or_one, beta, pred_ptr, target_ptr, dy_ptr, dpred_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_smooth_l1_backward_bf16_run(
                    numel, mode, inv_n_or_one, beta, pred_ptr, target_ptr, dy_ptr, dpred_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_smooth_l1_backward_f64_run(
                    numel, mode, inv_n_or_one, beta, pred_ptr, target_ptr, dy_ptr, dpred_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::SmoothL1LossBackwardPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}
