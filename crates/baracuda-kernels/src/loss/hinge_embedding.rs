//! HingeEmbedding loss plan — Tier-2 (Milestone 5.3).
//!
//! FW: `y = mean(input if t==1 else max(0, margin - input))`.
//! BW: `dinput[i] = sc if t==1 else (-sc if margin > input else 0)`.
//!
//! Target is `i64` ±1 indicator.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LossKind, LossReduction, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::common::{check_supported_dtype, map_status, unpack_workspace, validate_shape};

/// Descriptor for a HingeEmbedding loss op.
#[derive(Copy, Clone, Debug)]
pub struct HingeEmbeddingLossDescriptor<const N: usize> {
    /// Input shape (target must match).
    pub input_shape: [i32; N],
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Margin (PyTorch default 1.0).
    pub margin: f32,
    /// Element type of input / output.
    pub element: ElementKind,
}

/// Args bundle for a HingeEmbedding FW launch.
pub struct HingeEmbeddingLossArgs<'a, T: Element, const N: usize> {
    /// Input tensor.
    pub input: TensorRef<'a, T, N>,
    /// Target tensor (i64, ±1 indicator).
    pub target: TensorRef<'a, i64, N>,
    /// Output: per-cell for None mode, scalar for Mean / Sum.
    pub out: TensorMut<'a, T, N>,
}

/// HingeEmbedding forward plan.
pub struct HingeEmbeddingLossPlan<T: Element, const N: usize> {
    desc: HingeEmbeddingLossDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> HingeEmbeddingLossPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &HingeEmbeddingLossDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::HingeEmbeddingLossPlan: descriptor element != T",
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
            op: LossKind::HingeEmbedding as u16,
            element: T::KIND,
            aux_element: Some(ElementKind::I64),
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
        args: HingeEmbeddingLossArgs<'_, T, N>,
    ) -> Result<()> {
        if args.input.shape != self.desc.input_shape
            || args.target.shape != self.desc.input_shape
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HingeEmbeddingLossPlan: shape mismatch",
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
        let margin = self.desc.margin;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_hinge_embedding_f32_run(
                    numel, mode, margin, input_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_hinge_embedding_f16_run(
                    numel, mode, margin, input_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_hinge_embedding_bf16_run(
                    numel, mode, margin, input_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_hinge_embedding_f64_run(
                    numel, mode, margin, input_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::HingeEmbeddingLossPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}

// =============================================================================
// BACKWARD
// =============================================================================

/// Descriptor for a HingeEmbedding backward op.
#[derive(Copy, Clone, Debug)]
pub struct HingeEmbeddingLossBackwardDescriptor<const N: usize> {
    /// Input shape.
    pub input_shape: [i32; N],
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Margin.
    pub margin: f32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a HingeEmbedding BW launch.
pub struct HingeEmbeddingLossBackwardArgs<'a, T: Element, const N: usize> {
    /// Input saved from FW.
    pub input: TensorRef<'a, T, N>,
    /// Target saved from FW.
    pub target: TensorRef<'a, i64, N>,
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// Output gradient w.r.t. input.
    pub dinput: TensorMut<'a, T, N>,
}

/// HingeEmbedding backward plan.
pub struct HingeEmbeddingLossBackwardPlan<T: Element, const N: usize> {
    desc: HingeEmbeddingLossBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> HingeEmbeddingLossBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &HingeEmbeddingLossBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::HingeEmbeddingLossBackwardPlan: descriptor element != T",
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
            op: LossKind::HingeEmbedding as u16,
            element: T::KIND,
            aux_element: Some(ElementKind::I64),
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
        args: HingeEmbeddingLossBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        if args.input.shape != self.desc.input_shape
            || args.target.shape != self.desc.input_shape
            || args.dinput.shape != self.desc.input_shape
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HingeEmbeddingLossBackwardPlan: shape mismatch",
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
        let margin = self.desc.margin;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let target_ptr = args.target.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dinput_ptr = args.dinput.data.as_raw().0 as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_hinge_embedding_backward_f32_run(
                    numel, mode, inv_n_or_one, margin, input_ptr, target_ptr, dy_ptr, dinput_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_hinge_embedding_backward_f16_run(
                    numel, mode, inv_n_or_one, margin, input_ptr, target_ptr, dy_ptr, dinput_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_hinge_embedding_backward_bf16_run(
                    numel, mode, inv_n_or_one, margin, input_ptr, target_ptr, dy_ptr, dinput_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_hinge_embedding_backward_f64_run(
                    numel, mode, inv_n_or_one, margin, input_ptr, target_ptr, dy_ptr, dinput_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::HingeEmbeddingLossBackwardPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}
