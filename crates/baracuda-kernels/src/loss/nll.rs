//! NLLLoss plan — `y = -mean(input[target_idx[i]])` along the feature axis.
//!
//! Heterogeneous-dtype: input is `T`, target is `i64` class indices.
//! Today wired for 2-D input `[N, C]` with class axis = 1; higher-rank
//! input is reserved (would need a flatten-leading-dims preprocessing
//! step). Class-index target only; ignore-index masking is reserved.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LossKind, LossReduction, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::common::{check_supported_dtype, map_status, unpack_workspace};

/// Descriptor for an NLL loss op.
#[derive(Copy, Clone, Debug)]
pub struct NllLossDescriptor {
    /// Number of rows / batch size (extent along the non-class axis).
    pub n_rows: i32,
    /// Number of classes (extent along the class axis).
    pub class_extent: i32,
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Element type of input.
    pub element: ElementKind,
}

/// Args bundle for an NLL FW launch.
///
/// Input is rank-2 `[n_rows, class_extent]` (contiguous row-major).
/// Target is rank-1 `[n_rows]` of `i64`. For `None` mode, output is
/// rank-1 `[n_rows]`. For `Mean` / `Sum`, output is rank-1 `[1]`
/// (scalar) — wrapper accepts any TensorMut with numel ≥ 1.
pub struct NllLossArgs<'a, T: Element> {
    /// Input log-probabilities `[n_rows, class_extent]`.
    pub input: TensorRef<'a, T, 2>,
    /// Target class indices `[n_rows]`.
    pub target: TensorRef<'a, i64, 1>,
    /// Output: `[n_rows]` for None, scalar `[1]` for Mean / Sum.
    pub out: TensorMut<'a, T, 1>,
}

/// NLL loss forward plan.
pub struct NllLossPlan<T: Element> {
    desc: NllLossDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> NllLossPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &NllLossDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::NllLossPlan: descriptor element != T",
            ));
        }
        check_supported_dtype::<T>()?;
        if desc.n_rows < 0 || desc.class_extent < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::NllLossPlan: n_rows / class_extent must be non-negative",
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
            op: LossKind::Nll as u16,
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

    /// Workspace size in bytes — `n_rows · sizeof(T)` for Mean/Sum; 0 for None.
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
        args: NllLossArgs<'_, T>,
    ) -> Result<()> {
        let n_rows = self.desc.n_rows as i64;
        let class_extent = self.desc.class_extent;
        if n_rows == 0 {
            return Ok(());
        }
        if args.input.shape != [self.desc.n_rows, class_extent] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::NllLossPlan: input shape must be [n_rows, class_extent]",
            ));
        }
        if args.target.shape != [self.desc.n_rows] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::NllLossPlan: target shape must be [n_rows]",
            ));
        }
        // row_stride is the i64 stride of the leading (n_rows) axis.
        let row_stride_input: i64 = args.input.stride[0];
        let (ws_ptr, ws_bytes) = unpack_workspace(workspace, self.workspace_size())?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let target_ptr = args.target.data.as_raw().0 as *const c_void;
        let out_ptr = args.out.data.as_raw().0 as *mut c_void;
        let mode = self.desc.reduction as i32;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_nll_f32_run(
                    n_rows, class_extent, row_stride_input, mode, input_ptr, target_ptr,
                    out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_nll_f16_run(
                    n_rows, class_extent, row_stride_input, mode, input_ptr, target_ptr,
                    out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_nll_bf16_run(
                    n_rows, class_extent, row_stride_input, mode, input_ptr, target_ptr,
                    out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_nll_f64_run(
                    n_rows, class_extent, row_stride_input, mode, input_ptr, target_ptr,
                    out_ptr, ws_ptr, ws_bytes, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::NllLossPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}

// =============================================================================
// BACKWARD
// =============================================================================

/// Descriptor for an NLL backward op.
#[derive(Copy, Clone, Debug)]
pub struct NllLossBackwardDescriptor {
    /// Number of rows / batch size.
    pub n_rows: i32,
    /// Number of classes.
    pub class_extent: i32,
    /// Reduction mode used in the forward.
    pub reduction: LossReduction,
    /// Element type of input.
    pub element: ElementKind,
}

/// Args bundle for an NLL BW launch.
pub struct NllLossBackwardArgs<'a, T: Element> {
    /// Upstream gradient: `[n_rows]` for None, scalar `[1]` for Mean / Sum.
    pub dy: TensorRef<'a, T, 1>,
    /// Target class indices `[n_rows]`.
    pub target: TensorRef<'a, i64, 1>,
    /// Gradient w.r.t. input `[n_rows, class_extent]`. Launcher pre-zeros it.
    pub dinput: TensorMut<'a, T, 2>,
}

/// NLL backward plan.
pub struct NllLossBackwardPlan<T: Element> {
    desc: NllLossBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> NllLossBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &NllLossBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::NllLossBackwardPlan: descriptor element != T",
            ));
        }
        check_supported_dtype::<T>()?;
        if desc.n_rows < 0 || desc.class_extent < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::NllLossBackwardPlan: n_rows / class_extent must be \
                 non-negative",
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
            op: LossKind::Nll as u16,
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
        args: NllLossBackwardArgs<'_, T>,
    ) -> Result<()> {
        let n_rows = self.desc.n_rows as i64;
        let class_extent = self.desc.class_extent;
        if n_rows == 0 {
            return Ok(());
        }
        if args.target.shape != [self.desc.n_rows] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::NllLossBackwardPlan: target shape must be [n_rows]",
            ));
        }
        if args.dinput.shape != [self.desc.n_rows, class_extent] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::NllLossBackwardPlan: dinput shape must be [n_rows, \
                 class_extent]",
            ));
        }
        let row_stride_input: i64 = args.dinput.stride[0];
        let dinput_numel: i64 = (self.desc.n_rows as i64) * (class_extent as i64);
        let mode = self.desc.reduction as i32;
        let inv_n_or_one: f32 = match self.desc.reduction {
            LossReduction::None => 0.0,
            LossReduction::Mean => 1.0 / (n_rows as f32),
            LossReduction::Sum => 1.0,
        };
        let stream_ptr = stream.as_raw() as *mut c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let target_ptr = args.target.data.as_raw().0 as *const c_void;
        let dinput_ptr = args.dinput.data.as_raw().0 as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_nll_backward_f32_run(
                    n_rows, class_extent, row_stride_input, dinput_numel, mode, inv_n_or_one,
                    dy_ptr, target_ptr, dinput_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_nll_backward_f16_run(
                    n_rows, class_extent, row_stride_input, dinput_numel, mode, inv_n_or_one,
                    dy_ptr, target_ptr, dinput_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_nll_backward_bf16_run(
                    n_rows, class_extent, row_stride_input, dinput_numel, mode, inv_n_or_one,
                    dy_ptr, target_ptr, dinput_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_nll_backward_f64_run(
                    n_rows, class_extent, row_stride_input, dinput_numel, mode, inv_n_or_one,
                    dy_ptr, target_ptr, dinput_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::NllLossBackwardPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}
