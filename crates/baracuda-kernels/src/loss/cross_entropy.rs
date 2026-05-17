//! CrossEntropyLoss plan — `y = NLLLoss(LogSoftmax(input), target)`.
//!
//! Fused into a single per-row two-pass kernel for numerical stability
//! (max subtraction, sum-of-exp, then `-log_softmax[target]`).
//!
//! Two target formats are supported via [`CrossEntropyTargetKind`]:
//! - [`CrossEntropyTargetKind::ClassIndex`] — target is `i64[N]` class
//!   indices. The original fused FW: `y[n] = -log_softmax(input)[n, t[n]]`.
//! - [`CrossEntropyTargetKind::SoftProb`] — target is `T[N, C]` soft
//!   probability tensor (same dtype as input). Used for label smoothing
//!   / distillation. Formula: `y[n] = -Σ_c target[n,c] · log_softmax(input)[n,c]`.
//!
//! Today wired: `CrossEntropy × {f32, f16, bf16, f64}` × `{None, Mean, Sum}`
//! × `{ClassIndex, SoftProb}`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, CrossEntropyTargetKind, Element, ElementKind, KernelSku, LossKind,
    LossReduction, MathPrecision, OpCategory, PlanPreference, PrecisionGuarantee, TensorMut,
    TensorRef, Workspace,
};

use super::common::{check_supported_dtype, map_status, unpack_workspace};

/// Descriptor for a CrossEntropyLoss op.
#[derive(Copy, Clone, Debug)]
pub struct CrossEntropyLossDescriptor {
    /// Number of rows / batch size.
    pub n_rows: i32,
    /// Number of classes.
    pub class_extent: i32,
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Target tensor format.
    pub target_kind: CrossEntropyTargetKind,
    /// Element type of input.
    pub element: ElementKind,
}

/// Args bundle for a CrossEntropyLoss FW launch.
///
/// Exactly one of `target` (class-index) or `soft_target` (soft probability)
/// must be `Some` to match the descriptor's `target_kind`.
pub struct CrossEntropyLossArgs<'a, T: Element> {
    /// Input logits `[n_rows, class_extent]`.
    pub input: TensorRef<'a, T, 2>,
    /// Target class indices `[n_rows]` — populated when `target_kind ==
    /// ClassIndex`.
    pub target: Option<TensorRef<'a, i64, 1>>,
    /// Soft-target probability `[n_rows, class_extent]` (dtype `T`) —
    /// populated when `target_kind == SoftProb`.
    pub soft_target: Option<TensorRef<'a, T, 2>>,
    /// Output: `[n_rows]` for None, scalar `[1]` for Mean / Sum.
    pub out: TensorMut<'a, T, 1>,
}

/// CrossEntropyLoss forward plan.
pub struct CrossEntropyLossPlan<T: Element> {
    desc: CrossEntropyLossDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> CrossEntropyLossPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &CrossEntropyLossDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::CrossEntropyLossPlan: descriptor element != T",
            ));
        }
        check_supported_dtype::<T>()?;
        if desc.n_rows < 0 || desc.class_extent < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CrossEntropyLossPlan: n_rows / class_extent must be \
                 non-negative",
            ));
        }
        if desc.class_extent < 1 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CrossEntropyLossPlan: class_extent must be ≥ 1",
            ));
        }
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
            op: LossKind::CrossEntropy as u16,
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

    /// Workspace size — `n_rows · sizeof(T)` for Mean/Sum; 0 for None.
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
        args: CrossEntropyLossArgs<'_, T>,
    ) -> Result<()> {
        let n_rows = self.desc.n_rows as i64;
        let class_extent = self.desc.class_extent;
        if n_rows == 0 {
            return Ok(());
        }
        if args.input.shape != [self.desc.n_rows, class_extent] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CrossEntropyLossPlan: input shape must be [n_rows, \
                 class_extent]",
            ));
        }
        let row_stride_input: i64 = args.input.stride[0];
        let (ws_ptr, ws_bytes) = unpack_workspace(workspace, self.workspace_size())?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let out_ptr = args.out.data.as_raw().0 as *mut c_void;
        let mode = self.desc.reduction as i32;

        let status = match self.desc.target_kind {
            CrossEntropyTargetKind::ClassIndex => {
                let target = args.target.ok_or(Error::InvalidProblem(
                    "baracuda-kernels::CrossEntropyLossPlan: target_kind=ClassIndex requires \
                     `target` arg",
                ))?;
                if target.shape != [self.desc.n_rows] {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::CrossEntropyLossPlan: target shape must be [n_rows]",
                    ));
                }
                let target_ptr = target.data.as_raw().0 as *const c_void;
                match T::KIND {
                    ElementKind::F32 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_loss_cross_entropy_f32_run(
                            n_rows, class_extent, row_stride_input, mode, input_ptr, target_ptr,
                            out_ptr, ws_ptr, ws_bytes, stream_ptr,
                        )
                    },
                    ElementKind::F16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_loss_cross_entropy_f16_run(
                            n_rows, class_extent, row_stride_input, mode, input_ptr, target_ptr,
                            out_ptr, ws_ptr, ws_bytes, stream_ptr,
                        )
                    },
                    ElementKind::Bf16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_loss_cross_entropy_bf16_run(
                            n_rows, class_extent, row_stride_input, mode, input_ptr, target_ptr,
                            out_ptr, ws_ptr, ws_bytes, stream_ptr,
                        )
                    },
                    ElementKind::F64 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_loss_cross_entropy_f64_run(
                            n_rows, class_extent, row_stride_input, mode, input_ptr, target_ptr,
                            out_ptr, ws_ptr, ws_bytes, stream_ptr,
                        )
                    },
                    _ => {
                        return Err(Error::Unsupported(
                            "baracuda-kernels::CrossEntropyLossPlan::run unwired dtype",
                        ));
                    }
                }
            }
            CrossEntropyTargetKind::SoftProb => {
                let soft = args.soft_target.ok_or(Error::InvalidProblem(
                    "baracuda-kernels::CrossEntropyLossPlan: target_kind=SoftProb requires \
                     `soft_target` arg",
                ))?;
                if soft.shape != [self.desc.n_rows, class_extent] {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::CrossEntropyLossPlan: soft_target shape must be \
                         [n_rows, class_extent]",
                    ));
                }
                let row_stride_target: i64 = soft.stride[0];
                let target_ptr = soft.data.as_raw().0 as *const c_void;
                match T::KIND {
                    ElementKind::F32 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_loss_cross_entropy_soft_f32_run(
                            n_rows, class_extent, row_stride_input, row_stride_target, mode,
                            input_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                        )
                    },
                    ElementKind::F16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_loss_cross_entropy_soft_f16_run(
                            n_rows, class_extent, row_stride_input, row_stride_target, mode,
                            input_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                        )
                    },
                    ElementKind::Bf16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_loss_cross_entropy_soft_bf16_run(
                            n_rows, class_extent, row_stride_input, row_stride_target, mode,
                            input_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                        )
                    },
                    ElementKind::F64 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_loss_cross_entropy_soft_f64_run(
                            n_rows, class_extent, row_stride_input, row_stride_target, mode,
                            input_ptr, target_ptr, out_ptr, ws_ptr, ws_bytes, stream_ptr,
                        )
                    },
                    _ => {
                        return Err(Error::Unsupported(
                            "baracuda-kernels::CrossEntropyLossPlan::run unwired dtype",
                        ));
                    }
                }
            }
        };
        map_status(status)
    }
}

// =============================================================================
// BACKWARD
// =============================================================================

/// Descriptor for a CrossEntropyLoss backward op.
#[derive(Copy, Clone, Debug)]
pub struct CrossEntropyLossBackwardDescriptor {
    /// Number of rows / batch size.
    pub n_rows: i32,
    /// Number of classes.
    pub class_extent: i32,
    /// Reduction mode used in the forward.
    pub reduction: LossReduction,
    /// Target tensor format (must match forward).
    pub target_kind: CrossEntropyTargetKind,
    /// Element type of input.
    pub element: ElementKind,
}

/// Args bundle for a CrossEntropyLoss BW launch.
pub struct CrossEntropyLossBackwardArgs<'a, T: Element> {
    /// Input logits (saved from FW).
    pub input: TensorRef<'a, T, 2>,
    /// Target class indices `[n_rows]` — populated when `target_kind ==
    /// ClassIndex`.
    pub target: Option<TensorRef<'a, i64, 1>>,
    /// Soft-target probability `[n_rows, class_extent]` — populated when
    /// `target_kind == SoftProb`.
    pub soft_target: Option<TensorRef<'a, T, 2>>,
    /// Upstream gradient: `[n_rows]` for None, scalar `[1]` for Mean / Sum.
    pub dy: TensorRef<'a, T, 1>,
    /// Gradient w.r.t. input `[n_rows, class_extent]`.
    pub dinput: TensorMut<'a, T, 2>,
}

/// CrossEntropyLoss backward plan.
pub struct CrossEntropyLossBackwardPlan<T: Element> {
    desc: CrossEntropyLossBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> CrossEntropyLossBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &CrossEntropyLossBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::CrossEntropyLossBackwardPlan: descriptor element != T",
            ));
        }
        check_supported_dtype::<T>()?;
        if desc.n_rows < 0 || desc.class_extent < 1 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CrossEntropyLossBackwardPlan: invalid shape parameters",
            ));
        }
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
            op: LossKind::CrossEntropy as u16,
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
        args: CrossEntropyLossBackwardArgs<'_, T>,
    ) -> Result<()> {
        let n_rows = self.desc.n_rows as i64;
        let class_extent = self.desc.class_extent;
        if n_rows == 0 {
            return Ok(());
        }
        if args.input.shape != [self.desc.n_rows, class_extent] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CrossEntropyLossBackwardPlan: input shape must be \
                 [n_rows, class_extent]",
            ));
        }
        if args.dinput.shape != [self.desc.n_rows, class_extent] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CrossEntropyLossBackwardPlan: dinput shape must be \
                 [n_rows, class_extent]",
            ));
        }
        let row_stride_input: i64 = args.input.stride[0];
        let mode = self.desc.reduction as i32;
        let inv_n_or_one: f32 = match self.desc.reduction {
            LossReduction::None => 0.0,
            LossReduction::Mean => 1.0 / (n_rows as f32),
            LossReduction::Sum => 1.0,
        };
        let stream_ptr = stream.as_raw() as *mut c_void;
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dinput_ptr = args.dinput.data.as_raw().0 as *mut c_void;

        let status = match self.desc.target_kind {
            CrossEntropyTargetKind::ClassIndex => {
                let target = args.target.ok_or(Error::InvalidProblem(
                    "baracuda-kernels::CrossEntropyLossBackwardPlan: target_kind=ClassIndex \
                     requires `target` arg",
                ))?;
                if target.shape != [self.desc.n_rows] {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::CrossEntropyLossBackwardPlan: target shape must be \
                         [n_rows]",
                    ));
                }
                let target_ptr = target.data.as_raw().0 as *const c_void;
                match T::KIND {
                    ElementKind::F32 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_loss_cross_entropy_backward_f32_run(
                            n_rows, class_extent, row_stride_input, mode, inv_n_or_one, input_ptr,
                            target_ptr, dy_ptr, dinput_ptr, core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::F16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_loss_cross_entropy_backward_f16_run(
                            n_rows, class_extent, row_stride_input, mode, inv_n_or_one, input_ptr,
                            target_ptr, dy_ptr, dinput_ptr, core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::Bf16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_loss_cross_entropy_backward_bf16_run(
                            n_rows, class_extent, row_stride_input, mode, inv_n_or_one, input_ptr,
                            target_ptr, dy_ptr, dinput_ptr, core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::F64 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_loss_cross_entropy_backward_f64_run(
                            n_rows, class_extent, row_stride_input, mode, inv_n_or_one, input_ptr,
                            target_ptr, dy_ptr, dinput_ptr, core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    _ => {
                        return Err(Error::Unsupported(
                            "baracuda-kernels::CrossEntropyLossBackwardPlan::run unwired dtype",
                        ));
                    }
                }
            }
            CrossEntropyTargetKind::SoftProb => {
                let soft = args.soft_target.ok_or(Error::InvalidProblem(
                    "baracuda-kernels::CrossEntropyLossBackwardPlan: target_kind=SoftProb \
                     requires `soft_target` arg",
                ))?;
                if soft.shape != [self.desc.n_rows, class_extent] {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::CrossEntropyLossBackwardPlan: soft_target shape \
                         must be [n_rows, class_extent]",
                    ));
                }
                let row_stride_target: i64 = soft.stride[0];
                let target_ptr = soft.data.as_raw().0 as *const c_void;
                match T::KIND {
                    ElementKind::F32 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_loss_cross_entropy_soft_backward_f32_run(
                            n_rows, class_extent, row_stride_input, row_stride_target, mode,
                            inv_n_or_one, input_ptr, target_ptr, dy_ptr, dinput_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::F16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_loss_cross_entropy_soft_backward_f16_run(
                            n_rows, class_extent, row_stride_input, row_stride_target, mode,
                            inv_n_or_one, input_ptr, target_ptr, dy_ptr, dinput_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::Bf16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_loss_cross_entropy_soft_backward_bf16_run(
                            n_rows, class_extent, row_stride_input, row_stride_target, mode,
                            inv_n_or_one, input_ptr, target_ptr, dy_ptr, dinput_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::F64 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_loss_cross_entropy_soft_backward_f64_run(
                            n_rows, class_extent, row_stride_input, row_stride_target, mode,
                            inv_n_or_one, input_ptr, target_ptr, dy_ptr, dinput_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    _ => {
                        return Err(Error::Unsupported(
                            "baracuda-kernels::CrossEntropyLossBackwardPlan::run unwired dtype",
                        ));
                    }
                }
            }
        };
        map_status(status)
    }
}
