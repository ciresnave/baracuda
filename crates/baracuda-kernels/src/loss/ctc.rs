//! CTCLoss plan — `torch.nn.CTCLoss` (Phase 5 final deferral).
//!
//! Connectionist Temporal Classification: sequence-to-sequence loss
//! where input and target sequences have different lengths. Computed
//! via forward dynamic programming on the CTC lattice — one CUDA block
//! per batch sample, threads cooperate on the per-time-step recurrence
//! over the extended target sequence of length `L = 2·S + 1` (blanks
//! inserted between every label).
//!
//! ## Inputs (FW)
//! - `log_probs`: `T[max_time, batch_size, num_classes]` — must be log-
//!   probabilities (typically `log_softmax(logits)` along the class axis).
//! - `targets`: `i64[batch_size, max_target_len]` — per-sample target
//!   class indices in `[0, num_classes)` excluding `blank`.
//! - `input_lengths`: `i64[batch_size]` — per-sample actual input
//!   timesteps (must be ≤ `max_time`).
//! - `target_lengths`: `i64[batch_size]` — per-sample actual target
//!   lengths (must be ≤ `max_target_len`).
//!
//! ## Outputs (FW)
//! - `loss`: `T[batch_size]` for [`LossReduction::None`]; `T[1]` for
//!   [`LossReduction::Mean`] / [`LossReduction::Sum`]. Mean reduction
//!   divides by `Σ target_lengths` (the standard CTC mean convention).
//! - `alpha`: workspace tensor of accumulator type
//!   (`f32` for `{f32, f16, bf16}`; `f64` for `f64`) shaped
//!   `[max_time, batch_size, 2·max_target_len + 1]`. Saved for BW reuse.
//!
//! ## Caps (trailblazer)
//! - `max_target_len ≤ 256` (so `L_max = 513` fits within a 1024-thread
//!   block with comfortable margin).
//! - `num_classes ≤ 32` (so the per-(t,n) gamma scatter fits in shared
//!   memory and the BW kernel's per-class write loop is short).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LossKind, LossReduction, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::common::{check_supported_dtype, map_status, unpack_workspace};

/// Descriptor for a CTCLoss FW op.
#[derive(Copy, Clone, Debug)]
pub struct CtcLossDescriptor {
    /// Maximum input timestep extent (T).
    pub max_time: i32,
    /// Batch size (N).
    pub batch_size: i32,
    /// Number of classes including the blank class (C). Must be ≤ 32.
    pub num_classes: i32,
    /// Maximum target length (S). Must be ≤ 256.
    pub max_target_len: i32,
    /// Index of the blank class in `[0, num_classes)`. PyTorch default 0.
    pub blank: i32,
    /// Reduction mode.
    pub reduction: LossReduction,
    /// If `true`, clamp infinite losses (typically when `input_lengths[n]`
    /// is too short for the lattice to be reachable) to zero and zero
    /// the matching gradient slice.
    pub zero_infinity: bool,
    /// Element type of `log_probs` / output.
    pub element: ElementKind,
}

/// Args bundle for a CTCLoss FW launch.
pub struct CtcLossArgs<'a, T: Element> {
    /// Log-probabilities `[max_time, batch_size, num_classes]`. Must be
    /// row-major contiguous along the class axis.
    pub log_probs: TensorRef<'a, T, 3>,
    /// Target indices `[batch_size, max_target_len]`. Row-major.
    pub targets: TensorRef<'a, i64, 2>,
    /// Per-sample input timestep counts `[batch_size]`.
    pub input_lengths: TensorRef<'a, i64, 1>,
    /// Per-sample target lengths `[batch_size]`.
    pub target_lengths: TensorRef<'a, i64, 1>,
    /// Output loss tensor. Shape `[batch_size]` for `None`, `[1]` for
    /// `Mean` / `Sum`. Wrapper accepts any TensorMut with numel ≥ that.
    pub loss: TensorMut<'a, T, 1>,
    /// Alpha workspace `[max_time, batch_size, L_max]` where
    /// `L_max = 2·max_target_len + 1`. Element type is `f32` for
    /// `{f32, f16, bf16}` and `f64` for `f64` (caller must allocate
    /// accordingly). Saved for BW reuse.
    pub alpha: TensorMut<'a, u8, 1>,
}

/// CTCLoss FW plan.
pub struct CtcLossPlan<T: Element> {
    desc: CtcLossDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> CtcLossPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &CtcLossDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::CtcLossPlan: descriptor element != T",
            ));
        }
        check_supported_dtype::<T>()?;
        if desc.max_time < 0
            || desc.batch_size < 0
            || desc.num_classes < 0
            || desc.max_target_len < 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CtcLossPlan: dimensions must be non-negative",
            ));
        }
        if desc.num_classes > 32 {
            return Err(Error::Unsupported(
                "baracuda-kernels::CtcLossPlan: num_classes > 32 not supported \
                 (trailblazer cap)",
            ));
        }
        if desc.max_target_len > 256 {
            return Err(Error::Unsupported(
                "baracuda-kernels::CtcLossPlan: max_target_len > 256 not \
                 supported (trailblazer cap)",
            ));
        }
        if desc.num_classes > 0 && (desc.blank < 0 || desc.blank >= desc.num_classes) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CtcLossPlan: blank must be in [0, num_classes)",
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
            op: LossKind::Ctc as u16,
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

    /// Required `alpha` workspace size in bytes.
    /// `max_time · batch_size · (2·max_target_len + 1) · sizeof(acc_type)`,
    /// where `acc_type = f32` for `{f32, f16, bf16}` and `f64` for `f64`.
    #[inline]
    pub fn alpha_workspace_size(&self) -> usize {
        let acc_size = if T::KIND == ElementKind::F64 { 8usize } else { 4usize };
        let l_max = (2 * self.desc.max_target_len as i64 + 1).max(1);
        (self.desc.max_time as usize)
            .saturating_mul(self.desc.batch_size as usize)
            .saturating_mul(l_max as usize)
            .saturating_mul(acc_size)
    }

    /// Aux workspace size (per-sample loss buffer): `batch_size ·
    /// sizeof(acc_type)`.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        let acc_size = if T::KIND == ElementKind::F64 { 8usize } else { 4usize };
        (self.desc.batch_size as usize).saturating_mul(acc_size)
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
        args: CtcLossArgs<'_, T>,
    ) -> Result<()> {
        if self.desc.batch_size == 0 || self.desc.max_time == 0 {
            return Ok(());
        }
        if args.log_probs.shape
            != [self.desc.max_time, self.desc.batch_size, self.desc.num_classes]
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CtcLossPlan: log_probs shape must be \
                 [max_time, batch_size, num_classes]",
            ));
        }
        if args.targets.shape != [self.desc.batch_size, self.desc.max_target_len] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CtcLossPlan: targets shape must be \
                 [batch_size, max_target_len]",
            ));
        }
        if args.input_lengths.shape != [self.desc.batch_size] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CtcLossPlan: input_lengths shape must be [batch_size]",
            ));
        }
        if args.target_lengths.shape != [self.desc.batch_size] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CtcLossPlan: target_lengths shape must be [batch_size]",
            ));
        }
        let needed_alpha = self.alpha_workspace_size();
        if (args.alpha.data.len()) < needed_alpha {
            return Err(Error::WorkspaceTooSmall {
                needed: needed_alpha,
                got: args.alpha.data.len(),
            });
        }
        let (ws_ptr, ws_bytes) = unpack_workspace(workspace, self.workspace_size())?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let log_probs_ptr = args.log_probs.data.as_raw().0 as *const c_void;
        let targets_ptr = args.targets.data.as_raw().0 as *const c_void;
        let input_lengths_ptr = args.input_lengths.data.as_raw().0 as *const c_void;
        let target_lengths_ptr = args.target_lengths.data.as_raw().0 as *const c_void;
        let alpha_ptr = args.alpha.data.as_raw().0 as *mut c_void;
        let out_ptr = args.loss.data.as_raw().0 as *mut c_void;
        let mode = self.desc.reduction as i32;
        let zinf = if self.desc.zero_infinity { 1 } else { 0 };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_ctc_f32_run(
                    self.desc.max_time,
                    self.desc.batch_size,
                    self.desc.num_classes,
                    self.desc.max_target_len,
                    self.desc.blank,
                    mode,
                    zinf,
                    log_probs_ptr,
                    targets_ptr,
                    input_lengths_ptr,
                    target_lengths_ptr,
                    alpha_ptr,
                    out_ptr,
                    ws_ptr,
                    ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_ctc_f16_run(
                    self.desc.max_time,
                    self.desc.batch_size,
                    self.desc.num_classes,
                    self.desc.max_target_len,
                    self.desc.blank,
                    mode,
                    zinf,
                    log_probs_ptr,
                    targets_ptr,
                    input_lengths_ptr,
                    target_lengths_ptr,
                    alpha_ptr,
                    out_ptr,
                    ws_ptr,
                    ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_ctc_bf16_run(
                    self.desc.max_time,
                    self.desc.batch_size,
                    self.desc.num_classes,
                    self.desc.max_target_len,
                    self.desc.blank,
                    mode,
                    zinf,
                    log_probs_ptr,
                    targets_ptr,
                    input_lengths_ptr,
                    target_lengths_ptr,
                    alpha_ptr,
                    out_ptr,
                    ws_ptr,
                    ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_ctc_f64_run(
                    self.desc.max_time,
                    self.desc.batch_size,
                    self.desc.num_classes,
                    self.desc.max_target_len,
                    self.desc.blank,
                    mode,
                    zinf,
                    log_probs_ptr,
                    targets_ptr,
                    input_lengths_ptr,
                    target_lengths_ptr,
                    alpha_ptr,
                    out_ptr,
                    ws_ptr,
                    ws_bytes,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::CtcLossPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}

// =============================================================================
// BACKWARD
// =============================================================================

/// Descriptor for a CTCLoss BW op.
#[derive(Copy, Clone, Debug)]
pub struct CtcLossBackwardDescriptor {
    /// Maximum input timestep extent (T).
    pub max_time: i32,
    /// Batch size (N).
    pub batch_size: i32,
    /// Number of classes (C). Must be ≤ 32.
    pub num_classes: i32,
    /// Maximum target length (S). Must be ≤ 256.
    pub max_target_len: i32,
    /// Index of the blank class.
    pub blank: i32,
    /// Reduction mode used in the forward.
    pub reduction: LossReduction,
    /// Whether infinite-loss samples were clamped to zero in FW.
    pub zero_infinity: bool,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a CTCLoss BW launch.
pub struct CtcLossBackwardArgs<'a, T: Element> {
    /// Same `log_probs` `[max_time, batch_size, num_classes]` as FW.
    pub log_probs: TensorRef<'a, T, 3>,
    /// `[batch_size, max_target_len]`.
    pub targets: TensorRef<'a, i64, 2>,
    /// `[batch_size]`.
    pub input_lengths: TensorRef<'a, i64, 1>,
    /// `[batch_size]`.
    pub target_lengths: TensorRef<'a, i64, 1>,
    /// Per-sample loss vector saved from FW. Element type is the acc type
    /// (`f32` for `{f32, f16, bf16}`; `f64` for `f64`). Shape `[batch_size]`.
    pub per_sample_loss: TensorRef<'a, u8, 1>,
    /// Alpha workspace from FW.
    pub alpha: TensorRef<'a, u8, 1>,
    /// Upstream loss gradient: `T[batch_size]` for `None`, `T[1]` for
    /// `Mean` / `Sum`.
    pub dloss: TensorRef<'a, T, 1>,
    /// Gradient w.r.t. `log_probs`: `T[max_time, batch_size, num_classes]`.
    /// Launcher pre-zeros it.
    pub dlog_probs: TensorMut<'a, T, 3>,
    /// Σ target_lengths from the FW pass (for Mean denominator). Caller-
    /// computed and passed in to avoid a device→host sync.
    pub mean_denom: i64,
}

/// CTCLoss BW plan.
pub struct CtcLossBackwardPlan<T: Element> {
    desc: CtcLossBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> CtcLossBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &CtcLossBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::CtcLossBackwardPlan: descriptor element != T",
            ));
        }
        check_supported_dtype::<T>()?;
        if desc.max_time < 0
            || desc.batch_size < 0
            || desc.num_classes < 0
            || desc.max_target_len < 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CtcLossBackwardPlan: dimensions must be non-negative",
            ));
        }
        if desc.num_classes > 32 {
            return Err(Error::Unsupported(
                "baracuda-kernels::CtcLossBackwardPlan: num_classes > 32 not supported",
            ));
        }
        if desc.max_target_len > 256 {
            return Err(Error::Unsupported(
                "baracuda-kernels::CtcLossBackwardPlan: max_target_len > 256 not supported",
            ));
        }
        if desc.num_classes > 0 && (desc.blank < 0 || desc.blank >= desc.num_classes) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CtcLossBackwardPlan: blank must be in [0, num_classes)",
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
            deterministic: false, // shared-mem atomicAdd for the γ scatter
        };
        let sku = KernelSku {
            category: OpCategory::Loss,
            op: LossKind::Ctc as u16,
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

    /// β-running-row workspace: `batch_size · (2·max_target_len + 1) ·
    /// sizeof(acc_type)`.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        let acc_size = if T::KIND == ElementKind::F64 { 8usize } else { 4usize };
        let l_max = (2 * self.desc.max_target_len as i64 + 1).max(1);
        (self.desc.batch_size as usize)
            .saturating_mul(l_max as usize)
            .saturating_mul(acc_size)
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
        args: CtcLossBackwardArgs<'_, T>,
    ) -> Result<()> {
        if self.desc.batch_size == 0 || self.desc.max_time == 0 {
            return Ok(());
        }
        if args.dlog_probs.shape
            != [self.desc.max_time, self.desc.batch_size, self.desc.num_classes]
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CtcLossBackwardPlan: dlog_probs shape must be \
                 [max_time, batch_size, num_classes]",
            ));
        }
        let (ws_ptr, ws_bytes) = unpack_workspace(workspace, self.workspace_size())?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let log_probs_ptr = args.log_probs.data.as_raw().0 as *const c_void;
        let targets_ptr = args.targets.data.as_raw().0 as *const c_void;
        let input_lengths_ptr = args.input_lengths.data.as_raw().0 as *const c_void;
        let target_lengths_ptr = args.target_lengths.data.as_raw().0 as *const c_void;
        let alpha_ptr = args.alpha.data.as_raw().0 as *const c_void;
        let per_sample_loss_ptr = args.per_sample_loss.data.as_raw().0 as *const c_void;
        let dloss_ptr = args.dloss.data.as_raw().0 as *const c_void;
        let dlog_probs_ptr = args.dlog_probs.data.as_raw().0 as *mut c_void;
        let mode = self.desc.reduction as i32;
        let zinf = if self.desc.zero_infinity { 1 } else { 0 };

        // inv_denom: factor applied to dloss to derive per-sample scale.
        //   None: 1.0 (per-sample dloss[n] is already the right scale)
        //   Sum:  1.0
        //   Mean: 1.0 / Σ target_lengths
        let inv_denom = match self.desc.reduction {
            LossReduction::Mean => {
                let d = args.mean_denom.max(1);
                1.0_f32 / (d as f32)
            }
            LossReduction::Sum | LossReduction::None => 1.0_f32,
        };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_ctc_backward_f32_run(
                    self.desc.max_time,
                    self.desc.batch_size,
                    self.desc.num_classes,
                    self.desc.max_target_len,
                    self.desc.blank,
                    mode,
                    zinf,
                    inv_denom,
                    log_probs_ptr,
                    targets_ptr,
                    input_lengths_ptr,
                    target_lengths_ptr,
                    alpha_ptr,
                    per_sample_loss_ptr,
                    dloss_ptr,
                    dlog_probs_ptr,
                    ws_ptr,
                    ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_ctc_backward_f16_run(
                    self.desc.max_time,
                    self.desc.batch_size,
                    self.desc.num_classes,
                    self.desc.max_target_len,
                    self.desc.blank,
                    mode,
                    zinf,
                    inv_denom,
                    log_probs_ptr,
                    targets_ptr,
                    input_lengths_ptr,
                    target_lengths_ptr,
                    alpha_ptr,
                    per_sample_loss_ptr,
                    dloss_ptr,
                    dlog_probs_ptr,
                    ws_ptr,
                    ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_ctc_backward_bf16_run(
                    self.desc.max_time,
                    self.desc.batch_size,
                    self.desc.num_classes,
                    self.desc.max_target_len,
                    self.desc.blank,
                    mode,
                    zinf,
                    inv_denom,
                    log_probs_ptr,
                    targets_ptr,
                    input_lengths_ptr,
                    target_lengths_ptr,
                    alpha_ptr,
                    per_sample_loss_ptr,
                    dloss_ptr,
                    dlog_probs_ptr,
                    ws_ptr,
                    ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_loss_ctc_backward_f64_run(
                    self.desc.max_time,
                    self.desc.batch_size,
                    self.desc.num_classes,
                    self.desc.max_target_len,
                    self.desc.blank,
                    mode,
                    zinf,
                    inv_denom,
                    log_probs_ptr,
                    targets_ptr,
                    input_lengths_ptr,
                    target_lengths_ptr,
                    alpha_ptr,
                    per_sample_loss_ptr,
                    dloss_ptr,
                    dlog_probs_ptr,
                    ws_ptr,
                    ws_bytes,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::CtcLossBackwardPlan::run unwired dtype",
                ));
            }
        };
        map_status(status)
    }
}
