//! PoissonNLL loss plan.
//!
//! `log_input=true`  (PyTorch default): `y = mean(exp(input) - target · input)`.
//! `log_input=false`: `y = mean(input - target · log(input))`.
//!
//! BW:
//! - log_input=true:  `dinput = (exp(input) - target) · scale`
//! - log_input=false: `dinput = (1 - target/input) · scale`

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LossKind, LossReduction, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::common::{check_supported_dtype, map_status, unpack_workspace, validate_shape};

/// Descriptor for a PoissonNLL loss op.
#[derive(Copy, Clone, Debug)]
pub struct PoissonNllLossDescriptor<const N: usize> {
    /// Input / target tensor shape.
    pub input_shape: [i32; N],
    /// Reduction mode.
    pub reduction: LossReduction,
    /// If true, input is pre-log'd (PyTorch default).
    pub log_input: bool,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a PoissonNLL FW launch.
pub struct PoissonNllLossArgs<'a, T: Element, const N: usize> {
    /// Input (pre-log'd when `log_input == true`, raw rate otherwise).
    pub input: TensorRef<'a, T, N>,
    /// Targets.
    pub target: TensorRef<'a, T, N>,
    /// Output.
    pub out: TensorMut<'a, T, N>,
}

/// PoissonNLL forward plan.
pub struct PoissonNllLossPlan<T: Element, const N: usize> {
    desc: PoissonNllLossDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> PoissonNllLossPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &PoissonNllLossDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::PoissonNllLossPlan: descriptor element != T",
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
            op: LossKind::PoissonNll as u16,
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
        args: PoissonNllLossArgs<'_, T, N>,
    ) -> Result<()> {
        if args.input.shape != self.desc.input_shape || args.target.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PoissonNllLossPlan: input / target shape mismatch",
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
        let log_input_flag: i32 = if self.desc.log_input { 1 } else { 0 };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_poisson_nll_f32_run(
                    numel, mode, log_input_flag, input_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_poisson_nll_f16_run(
                    numel, mode, log_input_flag, input_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_poisson_nll_bf16_run(
                    numel, mode, log_input_flag, input_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_poisson_nll_f64_run(
                    numel, mode, log_input_flag, input_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::PoissonNllLossPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}

// =============================================================================
// BACKWARD
// =============================================================================

/// Descriptor for a PoissonNLL backward op.
#[derive(Copy, Clone, Debug)]
pub struct PoissonNllLossBackwardDescriptor<const N: usize> {
    /// Input / target tensor shape.
    pub input_shape: [i32; N],
    /// Reduction mode used in the forward.
    pub reduction: LossReduction,
    /// Must match the forward's value.
    pub log_input: bool,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a PoissonNLL BW launch.
pub struct PoissonNllLossBackwardArgs<'a, T: Element, const N: usize> {
    /// Input (saved from FW).
    pub input: TensorRef<'a, T, N>,
    /// Targets (saved from FW).
    pub target: TensorRef<'a, T, N>,
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// Gradient w.r.t. input.
    pub dinput: TensorMut<'a, T, N>,
}

/// PoissonNLL backward plan.
pub struct PoissonNllLossBackwardPlan<T: Element, const N: usize> {
    desc: PoissonNllLossBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> PoissonNllLossBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &PoissonNllLossBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::PoissonNllLossBackwardPlan: descriptor element != T",
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
            op: LossKind::PoissonNll as u16,
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
        args: PoissonNllLossBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        if args.input.shape != self.desc.input_shape
            || args.target.shape != self.desc.input_shape
            || args.dinput.shape != self.desc.input_shape
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PoissonNllLossBackwardPlan: shape mismatch",
            ));
        }
        let numel = args.input.numel();
        if numel == 0 {
            return Ok(());
        }
        let mode = self.desc.reduction as i32;
        let inv_n_or_one: f32 = match self.desc.reduction {
            LossReduction::None => 0.0,
            LossReduction::Mean => 1.0 / (numel as f32),
            LossReduction::Sum => 1.0,
        };
        let log_input_flag: i32 = if self.desc.log_input { 1 } else { 0 };
        let stream_ptr = stream.as_raw() as *mut c_void;
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let target_ptr = args.target.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dinput_ptr = args.dinput.data.as_raw().0 as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_poisson_nll_backward_f32_run(
                    numel, mode, inv_n_or_one, log_input_flag, input_ptr, target_ptr, dy_ptr,
                    dinput_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_poisson_nll_backward_f16_run(
                    numel, mode, inv_n_or_one, log_input_flag, input_ptr, target_ptr, dy_ptr,
                    dinput_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_poisson_nll_backward_bf16_run(
                    numel, mode, inv_n_or_one, log_input_flag, input_ptr, target_ptr, dy_ptr,
                    dinput_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_poisson_nll_backward_f64_run(
                    numel, mode, inv_n_or_one, log_input_flag, input_ptr, target_ptr, dy_ptr,
                    dinput_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::PoissonNllLossBackwardPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}
