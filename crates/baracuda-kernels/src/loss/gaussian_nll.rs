//! GaussianNLL loss plan.
//!
//! FW: `y = 0.5 · mean(log(max(var, eps)) + (input - target)² / max(var, eps))`.
//!
//! BW: `dinput = (input - target) / max(var, eps) · scale`. Only `dinput`
//! is computed; grad to `var` / `target` is not produced by default.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LossKind, LossReduction, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::common::{check_supported_dtype, map_status, unpack_workspace, validate_shape};

/// Descriptor for a GaussianNLL loss op.
#[derive(Copy, Clone, Debug)]
pub struct GaussianNllLossDescriptor<const N: usize> {
    /// Input / target / var tensor shape (all three match).
    pub input_shape: [i32; N],
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Numerical floor for `var` (PyTorch default 1e-6).
    pub eps: f32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a GaussianNLL FW launch.
pub struct GaussianNllLossArgs<'a, T: Element, const N: usize> {
    /// Mean predictions.
    pub input: TensorRef<'a, T, N>,
    /// Targets.
    pub target: TensorRef<'a, T, N>,
    /// Per-cell variance.
    pub var: TensorRef<'a, T, N>,
    /// Output.
    pub out: TensorMut<'a, T, N>,
}

/// GaussianNLL forward plan.
pub struct GaussianNllLossPlan<T: Element, const N: usize> {
    desc: GaussianNllLossDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> GaussianNllLossPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &GaussianNllLossDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::GaussianNllLossPlan: descriptor element != T",
            ));
        }
        if !(desc.eps > 0.0 && desc.eps.is_finite()) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GaussianNllLossPlan: eps must be > 0 and finite",
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
            op: LossKind::GaussianNll as u16,
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
        args: GaussianNllLossArgs<'_, T, N>,
    ) -> Result<()> {
        if args.input.shape != self.desc.input_shape
            || args.target.shape != self.desc.input_shape
            || args.var.shape != self.desc.input_shape
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GaussianNllLossPlan: input / target / var shape mismatch",
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
        let var_ptr = args.var.data.as_raw().0 as *const c_void;
        let out_ptr = args.out.data.as_raw().0 as *mut c_void;
        let mode = self.desc.reduction as i32;
        let eps = self.desc.eps;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_gaussian_nll_f32_run(
                    numel, mode, eps, input_ptr, target_ptr, var_ptr, out_ptr, ws_ptr, ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_gaussian_nll_f16_run(
                    numel, mode, eps, input_ptr, target_ptr, var_ptr, out_ptr, ws_ptr, ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_gaussian_nll_bf16_run(
                    numel, mode, eps, input_ptr, target_ptr, var_ptr, out_ptr, ws_ptr, ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_gaussian_nll_f64_run(
                    numel, mode, eps, input_ptr, target_ptr, var_ptr, out_ptr, ws_ptr, ws_bytes,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::GaussianNllLossPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}

// =============================================================================
// BACKWARD
// =============================================================================

/// Descriptor for a GaussianNLL backward op.
#[derive(Copy, Clone, Debug)]
pub struct GaussianNllLossBackwardDescriptor<const N: usize> {
    /// Input / target / var tensor shape.
    pub input_shape: [i32; N],
    /// Reduction mode used in the forward.
    pub reduction: LossReduction,
    /// Numerical floor for `var` (must match forward).
    pub eps: f32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a GaussianNLL BW launch.
pub struct GaussianNllLossBackwardArgs<'a, T: Element, const N: usize> {
    /// Input (saved from FW).
    pub input: TensorRef<'a, T, N>,
    /// Targets (saved from FW).
    pub target: TensorRef<'a, T, N>,
    /// Variance (saved from FW).
    pub var: TensorRef<'a, T, N>,
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// Gradient w.r.t. input.
    pub dinput: TensorMut<'a, T, N>,
}

/// GaussianNLL backward plan.
pub struct GaussianNllLossBackwardPlan<T: Element, const N: usize> {
    desc: GaussianNllLossBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> GaussianNllLossBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &GaussianNllLossBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::GaussianNllLossBackwardPlan: descriptor element != T",
            ));
        }
        if !(desc.eps > 0.0 && desc.eps.is_finite()) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GaussianNllLossBackwardPlan: eps must be > 0 and finite",
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
            op: LossKind::GaussianNll as u16,
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
        args: GaussianNllLossBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        if args.input.shape != self.desc.input_shape
            || args.target.shape != self.desc.input_shape
            || args.var.shape != self.desc.input_shape
            || args.dinput.shape != self.desc.input_shape
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GaussianNllLossBackwardPlan: shape mismatch",
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
        let eps = self.desc.eps;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let target_ptr = args.target.data.as_raw().0 as *const c_void;
        let var_ptr = args.var.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dinput_ptr = args.dinput.data.as_raw().0 as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_gaussian_nll_backward_f32_run(
                    numel, mode, inv_n_or_one, eps, input_ptr, target_ptr, var_ptr, dy_ptr,
                    dinput_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_gaussian_nll_backward_f16_run(
                    numel, mode, inv_n_or_one, eps, input_ptr, target_ptr, var_ptr, dy_ptr,
                    dinput_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_gaussian_nll_backward_bf16_run(
                    numel, mode, inv_n_or_one, eps, input_ptr, target_ptr, var_ptr, dy_ptr,
                    dinput_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_gaussian_nll_backward_f64_run(
                    numel, mode, inv_n_or_one, eps, input_ptr, target_ptr, var_ptr, dy_ptr,
                    dinput_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::GaussianNllLossBackwardPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}
