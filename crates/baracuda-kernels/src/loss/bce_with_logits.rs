//! BCEWithLogits loss plan — numerically stable BCE for raw logits.
//!
//! FW: `term[i] = max(x, 0) - x · target + log(1 + exp(-|x|))` where
//! `x = logits[i]`. The kernel applies sigmoid internally; the caller
//! passes raw (un-sigmoid'd) logits.
//!
//! BW: `dlogits[i] = (sigmoid(x) - target) · scale`. Numerically stable
//! sigmoid avoids overflow at extreme logit magnitudes.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LossKind, LossReduction, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::common::{check_supported_dtype, map_status, unpack_workspace, validate_shape};

/// Descriptor for a BCEWithLogits loss op.
#[derive(Copy, Clone, Debug)]
pub struct BceWithLogitsLossDescriptor<const N: usize> {
    /// Logits / target tensor shape (must match).
    pub input_shape: [i32; N],
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a BCEWithLogits FW launch.
pub struct BceWithLogitsLossArgs<'a, T: Element, const N: usize> {
    /// Raw logits tensor (NOT pre-sigmoid'd).
    pub logits: TensorRef<'a, T, N>,
    /// Targets.
    pub target: TensorRef<'a, T, N>,
    /// Output: per-cell for None mode, scalar for Mean / Sum.
    pub out: TensorMut<'a, T, N>,
}

/// BCEWithLogits forward plan.
pub struct BceWithLogitsLossPlan<T: Element, const N: usize> {
    desc: BceWithLogitsLossDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> BceWithLogitsLossPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &BceWithLogitsLossDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::BceWithLogitsLossPlan: descriptor element != T",
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
            op: LossKind::BceWithLogits as u16,
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
        args: BceWithLogitsLossArgs<'_, T, N>,
    ) -> Result<()> {
        if args.logits.shape != self.desc.input_shape
            || args.target.shape != self.desc.input_shape
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BceWithLogitsLossPlan: logits / target shape mismatch",
            ));
        }
        let numel = args.logits.numel();
        if numel == 0 {
            return Ok(());
        }
        let (ws_ptr, ws_bytes) = unpack_workspace(workspace, self.workspace_size())?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let logits_ptr = args.logits.data.as_raw().0 as *const c_void;
        let target_ptr = args.target.data.as_raw().0 as *const c_void;
        let out_ptr = args.out.data.as_raw().0 as *mut c_void;
        let mode = self.desc.reduction as i32;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_bce_with_logits_f32_run(
                    numel, mode, logits_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_bce_with_logits_f16_run(
                    numel, mode, logits_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_bce_with_logits_bf16_run(
                    numel, mode, logits_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_bce_with_logits_f64_run(
                    numel, mode, logits_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::BceWithLogitsLossPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}

// =============================================================================
// BACKWARD
// =============================================================================

/// Descriptor for a BCEWithLogits backward op.
#[derive(Copy, Clone, Debug)]
pub struct BceWithLogitsLossBackwardDescriptor<const N: usize> {
    /// Logits / target tensor shape.
    pub input_shape: [i32; N],
    /// Reduction mode used in the forward.
    pub reduction: LossReduction,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a BCEWithLogits BW launch.
pub struct BceWithLogitsLossBackwardArgs<'a, T: Element, const N: usize> {
    /// Raw logits (saved from FW).
    pub logits: TensorRef<'a, T, N>,
    /// Targets (saved from FW).
    pub target: TensorRef<'a, T, N>,
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// Gradient w.r.t. logits.
    pub dlogits: TensorMut<'a, T, N>,
}

/// BCEWithLogits backward plan.
pub struct BceWithLogitsLossBackwardPlan<T: Element, const N: usize> {
    desc: BceWithLogitsLossBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> BceWithLogitsLossBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &BceWithLogitsLossBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::BceWithLogitsLossBackwardPlan: descriptor element != T",
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
            op: LossKind::BceWithLogits as u16,
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
        args: BceWithLogitsLossBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        if args.logits.shape != self.desc.input_shape
            || args.target.shape != self.desc.input_shape
            || args.dlogits.shape != self.desc.input_shape
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BceWithLogitsLossBackwardPlan: shape mismatch",
            ));
        }
        let numel = args.logits.numel();
        if numel == 0 {
            return Ok(());
        }
        let mode = self.desc.reduction as i32;
        let inv_n_or_one: f32 = match self.desc.reduction {
            LossReduction::None => 0.0,
            LossReduction::Mean => 1.0 / (numel as f32),
            LossReduction::Sum => 1.0,
        };
        let stream_ptr = stream.as_raw() as *mut c_void;
        let logits_ptr = args.logits.data.as_raw().0 as *const c_void;
        let target_ptr = args.target.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dlogits_ptr = args.dlogits.data.as_raw().0 as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_bce_with_logits_backward_f32_run(
                    numel, mode, inv_n_or_one, logits_ptr, target_ptr, dy_ptr, dlogits_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_bce_with_logits_backward_f16_run(
                    numel, mode, inv_n_or_one, logits_ptr, target_ptr, dy_ptr, dlogits_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_bce_with_logits_backward_bf16_run(
                    numel, mode, inv_n_or_one, logits_ptr, target_ptr, dy_ptr, dlogits_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_bce_with_logits_backward_f64_run(
                    numel, mode, inv_n_or_one, logits_ptr, target_ptr, dy_ptr, dlogits_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::BceWithLogitsLossBackwardPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}
