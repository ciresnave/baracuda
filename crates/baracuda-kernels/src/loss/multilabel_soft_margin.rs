//! MultilabelSoftMargin loss plan — Tier-2 (Milestone 5.3).
//!
//! Per-row n: `-mean_c(target·log(σ(input)) + (1-target)·log(1-σ(input)))`.
//! Input `[N, C]`, target `[N, C]` `T`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LossKind, LossReduction, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::common::{check_supported_dtype, map_status, unpack_workspace};

/// Descriptor for a MultilabelSoftMargin loss op.
#[derive(Copy, Clone, Debug)]
pub struct MultilabelSoftMarginLossDescriptor {
    /// Number of rows N.
    pub n_rows: i32,
    /// Class extent C.
    pub class_extent: i32,
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a MultilabelSoftMargin FW launch.
pub struct MultilabelSoftMarginLossArgs<'a, T: Element> {
    /// Input tensor [N, C] (raw logits).
    pub input: TensorRef<'a, T, 2>,
    /// Target tensor [N, C] T (soft labels in [0, 1]).
    pub target: TensorRef<'a, T, 2>,
    /// Output.
    pub out: TensorMut<'a, T, 2>,
}

/// MultilabelSoftMargin forward plan.
pub struct MultilabelSoftMarginLossPlan<T: Element> {
    desc: MultilabelSoftMarginLossDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> MultilabelSoftMarginLossPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &MultilabelSoftMarginLossDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::MultilabelSoftMarginLossPlan: descriptor element != T",
            ));
        }
        if desc.n_rows < 0 || desc.class_extent < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::MultilabelSoftMarginLossPlan: n_rows / class_extent must be ≥ 0",
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
            op: LossKind::MultilabelSoftMargin as u16,
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
        args: MultilabelSoftMarginLossArgs<'_, T>,
    ) -> Result<()> {
        if self.desc.n_rows == 0 {
            return Ok(());
        }
        let (ws_ptr, ws_bytes) = unpack_workspace(workspace, self.workspace_size())?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let target_ptr = args.target.data.as_raw().0 as *const c_void;
        let out_ptr = args.out.data.as_raw().0 as *mut c_void;
        let mode = self.desc.reduction as i32;
        let n_rows = self.desc.n_rows as i64;
        let class_extent = self.desc.class_extent;
        let row_stride_in: i64 = class_extent as i64;
        let row_stride_tgt: i64 = class_extent as i64;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_multilabel_soft_margin_f32_run(
                    n_rows, class_extent, row_stride_in, row_stride_tgt, mode, input_ptr,
                    target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_multilabel_soft_margin_f16_run(
                    n_rows, class_extent, row_stride_in, row_stride_tgt, mode, input_ptr,
                    target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_multilabel_soft_margin_bf16_run(
                    n_rows, class_extent, row_stride_in, row_stride_tgt, mode, input_ptr,
                    target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_multilabel_soft_margin_f64_run(
                    n_rows, class_extent, row_stride_in, row_stride_tgt, mode, input_ptr,
                    target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::MultilabelSoftMarginLossPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}

// =============================================================================
// BACKWARD
// =============================================================================

/// Descriptor for a MultilabelSoftMargin backward op.
#[derive(Copy, Clone, Debug)]
pub struct MultilabelSoftMarginLossBackwardDescriptor {
    /// Number of rows N.
    pub n_rows: i32,
    /// Class extent C.
    pub class_extent: i32,
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a MultilabelSoftMargin BW launch.
pub struct MultilabelSoftMarginLossBackwardArgs<'a, T: Element> {
    /// Input saved from FW.
    pub input: TensorRef<'a, T, 2>,
    /// Target saved from FW.
    pub target: TensorRef<'a, T, 2>,
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, 2>,
    /// Output gradient w.r.t. input.
    pub dinput: TensorMut<'a, T, 2>,
}

/// MultilabelSoftMargin backward plan.
pub struct MultilabelSoftMarginLossBackwardPlan<T: Element> {
    desc: MultilabelSoftMarginLossBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> MultilabelSoftMarginLossBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &MultilabelSoftMarginLossBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::MultilabelSoftMarginLossBackwardPlan: descriptor element != T",
            ));
        }
        if desc.n_rows < 0 || desc.class_extent < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::MultilabelSoftMarginLossBackwardPlan: n_rows / class_extent must be ≥ 0",
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
            op: LossKind::MultilabelSoftMargin as u16,
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
        args: MultilabelSoftMarginLossBackwardArgs<'_, T>,
    ) -> Result<()> {
        if self.desc.n_rows == 0 {
            return Ok(());
        }
        let mode = self.desc.reduction as i32;
        let n_rows = self.desc.n_rows as i64;
        let class_extent = self.desc.class_extent;
        let inv_n_or_one: f32 = match self.desc.reduction {
            LossReduction::None => 0.0,
            LossReduction::Mean => 1.0 / (n_rows as f32),
            LossReduction::Sum => 1.0,
        };
        let stream_ptr = stream.as_raw() as *mut c_void;
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let target_ptr = args.target.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dinput_ptr = args.dinput.data.as_raw().0 as *mut c_void;
        let row_stride_in: i64 = class_extent as i64;
        let row_stride_tgt: i64 = class_extent as i64;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_multilabel_soft_margin_backward_f32_run(
                    n_rows, class_extent, row_stride_in, row_stride_tgt, mode, inv_n_or_one,
                    input_ptr, target_ptr, dy_ptr, dinput_ptr, core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_multilabel_soft_margin_backward_f16_run(
                    n_rows, class_extent, row_stride_in, row_stride_tgt, mode, inv_n_or_one,
                    input_ptr, target_ptr, dy_ptr, dinput_ptr, core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_multilabel_soft_margin_backward_bf16_run(
                    n_rows, class_extent, row_stride_in, row_stride_tgt, mode, inv_n_or_one,
                    input_ptr, target_ptr, dy_ptr, dinput_ptr, core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_multilabel_soft_margin_backward_f64_run(
                    n_rows, class_extent, row_stride_in, row_stride_tgt, mode, inv_n_or_one,
                    input_ptr, target_ptr, dy_ptr, dinput_ptr, core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::MultilabelSoftMarginLossBackwardPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}
