//! KLDiv loss plan — `y = mean(target·(log(target) - input))`.
//!
//! PyTorch convention: `input` is already log-prob. Cells with
//! `target == 0` contribute 0 (avoids `log(0)`). Today wired:
//! `KlDiv × {f32, f16, bf16, f64}` × `{None, Mean, Sum}`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LossKind, LossReduction, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::common::{check_supported_dtype, map_status, unpack_workspace, validate_shape};

/// Descriptor for a KLDiv loss op.
#[derive(Copy, Clone, Debug)]
pub struct KlDivLossDescriptor<const N: usize> {
    /// Input / target tensor shape (must match).
    pub input_shape: [i32; N],
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a KLDiv FW launch.
pub struct KlDivLossArgs<'a, T: Element, const N: usize> {
    /// Input tensor (log-probabilities — PyTorch convention).
    pub input: TensorRef<'a, T, N>,
    /// Targets tensor (probabilities).
    pub target: TensorRef<'a, T, N>,
    /// Output: per-cell for None mode, scalar for Mean / Sum.
    pub out: TensorMut<'a, T, N>,
}

/// KLDiv loss forward plan.
pub struct KlDivLossPlan<T: Element, const N: usize> {
    desc: KlDivLossDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> KlDivLossPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &KlDivLossDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::KlDivLossPlan: descriptor element != T",
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
            op: LossKind::KlDiv as u16,
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
        args: KlDivLossArgs<'_, T, N>,
    ) -> Result<()> {
        if args.input.shape != self.desc.input_shape || args.target.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::KlDivLossPlan: input / target shape mismatch",
            ));
        }
        let numel = args.input.numel();
        if numel == 0 {
            return Ok(());
        }
        let (ws_ptr, ws_bytes) = unpack_workspace(workspace, self.workspace_size())?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let target_ptr = args.target.data.as_raw().0 as *const c_void;
        let out_ptr = args.out.data.as_raw().0 as *mut c_void;
        let mode = self.desc.reduction as i32;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_kl_div_f32_run(
                    numel, mode, input_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_kl_div_f16_run(
                    numel, mode, input_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_kl_div_bf16_run(
                    numel, mode, input_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_kl_div_f64_run(
                    numel, mode, input_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::KlDivLossPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}

// =============================================================================
// BACKWARD
// =============================================================================

/// Descriptor for a KLDiv backward op.
#[derive(Copy, Clone, Debug)]
pub struct KlDivLossBackwardDescriptor<const N: usize> {
    /// Input / target tensor shape (must match dinput).
    pub input_shape: [i32; N],
    /// Reduction mode used in the forward.
    pub reduction: LossReduction,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a KLDiv BW launch.
pub struct KlDivLossBackwardArgs<'a, T: Element, const N: usize> {
    /// Targets (saved from FW).
    pub target: TensorRef<'a, T, N>,
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// Gradient w.r.t. input.
    pub dinput: TensorMut<'a, T, N>,
}

/// KLDiv backward plan.
pub struct KlDivLossBackwardPlan<T: Element, const N: usize> {
    desc: KlDivLossBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> KlDivLossBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &KlDivLossBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::KlDivLossBackwardPlan: descriptor element != T",
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
            op: LossKind::KlDiv as u16,
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
        args: KlDivLossBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        if args.target.shape != self.desc.input_shape || args.dinput.shape != self.desc.input_shape
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::KlDivLossBackwardPlan: shape mismatch",
            ));
        }
        let numel = args.target.numel();
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
        let target_ptr = args.target.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dinput_ptr = args.dinput.data.as_raw().0 as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_kl_div_backward_f32_run(
                    numel, mode, inv_n_or_one, target_ptr, dy_ptr, dinput_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_kl_div_backward_f16_run(
                    numel, mode, inv_n_or_one, target_ptr, dy_ptr, dinput_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_kl_div_backward_bf16_run(
                    numel, mode, inv_n_or_one, target_ptr, dy_ptr, dinput_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_kl_div_backward_f64_run(
                    numel, mode, inv_n_or_one, target_ptr, dy_ptr, dinput_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::KlDivLossBackwardPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}
