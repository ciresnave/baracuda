//! TripletMargin loss plan — Tier-2 (Milestone 5.3).
//!
//! Per-row: `max(0, ||a - p||_p - ||a - n||_p + margin)`.
//! 2-D input `[N, D]`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LossKind, LossReduction, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::common::{check_supported_dtype, map_status, unpack_workspace};

/// Descriptor for a TripletMargin loss op.
#[derive(Copy, Clone, Debug)]
pub struct TripletMarginLossDescriptor {
    /// Number of rows N.
    pub n_rows: i32,
    /// D extent.
    pub d_extent: i32,
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Margin (PyTorch default 1.0).
    pub margin: f32,
    /// p-norm (typically 2.0).
    pub p_norm: f32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a TripletMargin FW launch.
pub struct TripletMarginLossArgs<'a, T: Element> {
    /// Anchor tensor [N, D].
    pub anchor: TensorRef<'a, T, 2>,
    /// Positive tensor [N, D].
    pub positive: TensorRef<'a, T, 2>,
    /// Negative tensor [N, D].
    pub negative: TensorRef<'a, T, 2>,
    /// Output.
    pub out: TensorMut<'a, T, 2>,
}

/// TripletMargin forward plan.
pub struct TripletMarginLossPlan<T: Element> {
    desc: TripletMarginLossDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> TripletMarginLossPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &TripletMarginLossDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::TripletMarginLossPlan: descriptor element != T",
            ));
        }
        if desc.n_rows < 0 || desc.d_extent < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TripletMarginLossPlan: n_rows / d_extent must be ≥ 0",
            ));
        }
        if !(desc.p_norm > 0.0 && desc.p_norm.is_finite()) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TripletMarginLossPlan: p_norm must be > 0 and finite",
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
            op: LossKind::TripletMargin as u16,
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
        args: TripletMarginLossArgs<'_, T>,
    ) -> Result<()> {
        if self.desc.n_rows == 0 {
            return Ok(());
        }
        let (ws_ptr, ws_bytes) = unpack_workspace(workspace, self.workspace_size())?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let a_ptr = args.anchor.data.as_raw().0 as *const c_void;
        let p_ptr = args.positive.data.as_raw().0 as *const c_void;
        let n_ptr = args.negative.data.as_raw().0 as *const c_void;
        let out_ptr = args.out.data.as_raw().0 as *mut c_void;
        let mode = self.desc.reduction as i32;
        let margin = self.desc.margin;
        let p_norm = self.desc.p_norm;
        let n_rows = self.desc.n_rows as i64;
        let d_extent = self.desc.d_extent;
        let row_stride: i64 = d_extent as i64;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_triplet_margin_f32_run(
                    n_rows, d_extent, row_stride, mode, margin, p_norm, a_ptr, p_ptr, n_ptr,
                    out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_triplet_margin_f16_run(
                    n_rows, d_extent, row_stride, mode, margin, p_norm, a_ptr, p_ptr, n_ptr,
                    out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_triplet_margin_bf16_run(
                    n_rows, d_extent, row_stride, mode, margin, p_norm, a_ptr, p_ptr, n_ptr,
                    out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_triplet_margin_f64_run(
                    n_rows, d_extent, row_stride, mode, margin, p_norm, a_ptr, p_ptr, n_ptr,
                    out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::TripletMarginLossPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}

// =============================================================================
// BACKWARD
// =============================================================================

/// Descriptor for a TripletMargin backward op.
#[derive(Copy, Clone, Debug)]
pub struct TripletMarginLossBackwardDescriptor {
    /// Number of rows N.
    pub n_rows: i32,
    /// D extent.
    pub d_extent: i32,
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Margin.
    pub margin: f32,
    /// p-norm.
    pub p_norm: f32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a TripletMargin BW launch.
pub struct TripletMarginLossBackwardArgs<'a, T: Element> {
    /// Anchor saved from FW.
    pub anchor: TensorRef<'a, T, 2>,
    /// Positive saved from FW.
    pub positive: TensorRef<'a, T, 2>,
    /// Negative saved from FW.
    pub negative: TensorRef<'a, T, 2>,
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, 2>,
    /// Output gradient w.r.t. anchor.
    pub d_anchor: TensorMut<'a, T, 2>,
    /// Output gradient w.r.t. positive.
    pub d_positive: TensorMut<'a, T, 2>,
    /// Output gradient w.r.t. negative.
    pub d_negative: TensorMut<'a, T, 2>,
}

/// TripletMargin backward plan.
pub struct TripletMarginLossBackwardPlan<T: Element> {
    desc: TripletMarginLossBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> TripletMarginLossBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &TripletMarginLossBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::TripletMarginLossBackwardPlan: descriptor element != T",
            ));
        }
        if desc.n_rows < 0 || desc.d_extent < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TripletMarginLossBackwardPlan: n_rows / d_extent must be ≥ 0",
            ));
        }
        if !(desc.p_norm > 0.0 && desc.p_norm.is_finite()) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TripletMarginLossBackwardPlan: p_norm must be > 0",
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
            op: LossKind::TripletMargin as u16,
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
        args: TripletMarginLossBackwardArgs<'_, T>,
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
        let p_norm = self.desc.p_norm;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let a_ptr = args.anchor.data.as_raw().0 as *const c_void;
        let p_ptr = args.positive.data.as_raw().0 as *const c_void;
        let n_ptr = args.negative.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let da_ptr = args.d_anchor.data.as_raw().0 as *mut c_void;
        let dp_ptr = args.d_positive.data.as_raw().0 as *mut c_void;
        let dn_ptr = args.d_negative.data.as_raw().0 as *mut c_void;
        let row_stride: i64 = d_extent as i64;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_triplet_margin_backward_f32_run(
                    n_rows, d_extent, row_stride, mode, inv_n_or_one, margin, p_norm, a_ptr,
                    p_ptr, n_ptr, dy_ptr, da_ptr, dp_ptr, dn_ptr, core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_triplet_margin_backward_f16_run(
                    n_rows, d_extent, row_stride, mode, inv_n_or_one, margin, p_norm, a_ptr,
                    p_ptr, n_ptr, dy_ptr, da_ptr, dp_ptr, dn_ptr, core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_triplet_margin_backward_bf16_run(
                    n_rows, d_extent, row_stride, mode, inv_n_or_one, margin, p_norm, a_ptr,
                    p_ptr, n_ptr, dy_ptr, da_ptr, dp_ptr, dn_ptr, core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_triplet_margin_backward_f64_run(
                    n_rows, d_extent, row_stride, mode, inv_n_or_one, margin, p_norm, a_ptr,
                    p_ptr, n_ptr, dy_ptr, da_ptr, dp_ptr, dn_ptr, core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::TripletMarginLossBackwardPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}
