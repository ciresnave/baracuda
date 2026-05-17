//! MultiMargin loss plan — Tier-2 (Milestone 5.3).
//!
//! Per-row n: `Σ_{j != t_n} max(0, margin - input[n, t_n] + input[n, j])^p / C`.
//! Input `[N, C]`, target `[N]` i64 class indices.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LossKind, LossReduction, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::common::{check_supported_dtype, map_status, unpack_workspace};

/// Descriptor for a MultiMargin loss op.
#[derive(Copy, Clone, Debug)]
pub struct MultiMarginLossDescriptor {
    /// Number of rows N.
    pub n_rows: i32,
    /// Class extent C.
    pub class_extent: i32,
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Margin (PyTorch default 1.0).
    pub margin: f32,
    /// p-norm (PyTorch default 1.0).
    pub p_norm: f32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a MultiMargin FW launch.
pub struct MultiMarginLossArgs<'a, T: Element> {
    /// Input tensor [N, C].
    pub input: TensorRef<'a, T, 2>,
    /// Target tensor [N] i64 class indices (encoded as [N, 1]).
    pub target: TensorRef<'a, i64, 2>,
    /// Output.
    pub out: TensorMut<'a, T, 2>,
}

/// MultiMargin forward plan.
pub struct MultiMarginLossPlan<T: Element> {
    desc: MultiMarginLossDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> MultiMarginLossPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &MultiMarginLossDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::MultiMarginLossPlan: descriptor element != T",
            ));
        }
        if desc.n_rows < 0 || desc.class_extent < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::MultiMarginLossPlan: n_rows / class_extent must be ≥ 0",
            ));
        }
        if !(desc.p_norm > 0.0 && desc.p_norm.is_finite()) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::MultiMarginLossPlan: p_norm must be > 0",
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
            op: LossKind::MultiMargin as u16,
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
        args: MultiMarginLossArgs<'_, T>,
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
        let margin = self.desc.margin;
        let p_norm = self.desc.p_norm;
        let n_rows = self.desc.n_rows as i64;
        let class_extent = self.desc.class_extent;
        let row_stride: i64 = class_extent as i64;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_multi_margin_f32_run(
                    n_rows, class_extent, row_stride, mode, margin, p_norm, input_ptr, target_ptr,
                    out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_multi_margin_f16_run(
                    n_rows, class_extent, row_stride, mode, margin, p_norm, input_ptr, target_ptr,
                    out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_multi_margin_bf16_run(
                    n_rows, class_extent, row_stride, mode, margin, p_norm, input_ptr, target_ptr,
                    out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_multi_margin_f64_run(
                    n_rows, class_extent, row_stride, mode, margin, p_norm, input_ptr, target_ptr,
                    out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::MultiMarginLossPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}

// =============================================================================
// BACKWARD
// =============================================================================

/// Descriptor for a MultiMargin backward op.
#[derive(Copy, Clone, Debug)]
pub struct MultiMarginLossBackwardDescriptor {
    /// Number of rows N.
    pub n_rows: i32,
    /// Class extent C.
    pub class_extent: i32,
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Margin.
    pub margin: f32,
    /// p-norm.
    pub p_norm: f32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a MultiMargin BW launch.
pub struct MultiMarginLossBackwardArgs<'a, T: Element> {
    /// Input saved from FW.
    pub input: TensorRef<'a, T, 2>,
    /// Target saved from FW.
    pub target: TensorRef<'a, i64, 2>,
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, 2>,
    /// Output gradient w.r.t. input.
    pub dinput: TensorMut<'a, T, 2>,
}

/// MultiMargin backward plan.
pub struct MultiMarginLossBackwardPlan<T: Element> {
    desc: MultiMarginLossBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> MultiMarginLossBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &MultiMarginLossBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::MultiMarginLossBackwardPlan: descriptor element != T",
            ));
        }
        if desc.n_rows < 0 || desc.class_extent < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::MultiMarginLossBackwardPlan: n_rows / class_extent must be ≥ 0",
            ));
        }
        if !(desc.p_norm > 0.0 && desc.p_norm.is_finite()) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::MultiMarginLossBackwardPlan: p_norm must be > 0",
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
            op: LossKind::MultiMargin as u16,
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
        args: MultiMarginLossBackwardArgs<'_, T>,
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
        let margin = self.desc.margin;
        let p_norm = self.desc.p_norm;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let target_ptr = args.target.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dinput_ptr = args.dinput.data.as_raw().0 as *mut c_void;
        let row_stride: i64 = class_extent as i64;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_multi_margin_backward_f32_run(
                    n_rows, class_extent, row_stride, mode, inv_n_or_one, margin, p_norm,
                    input_ptr, target_ptr, dy_ptr, dinput_ptr, core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_multi_margin_backward_f16_run(
                    n_rows, class_extent, row_stride, mode, inv_n_or_one, margin, p_norm,
                    input_ptr, target_ptr, dy_ptr, dinput_ptr, core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_multi_margin_backward_bf16_run(
                    n_rows, class_extent, row_stride, mode, inv_n_or_one, margin, p_norm,
                    input_ptr, target_ptr, dy_ptr, dinput_ptr, core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_multi_margin_backward_f64_run(
                    n_rows, class_extent, row_stride, mode, inv_n_or_one, margin, p_norm,
                    input_ptr, target_ptr, dy_ptr, dinput_ptr, core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::MultiMarginLossBackwardPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}
