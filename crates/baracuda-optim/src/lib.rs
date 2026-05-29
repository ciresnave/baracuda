//! # baracuda-optim
//!
//! Fused multi-tensor optimizer plans (Adam / LAMB / SGD) for the
//! baracuda CUDA stack. Built on the **`multi_tensor_apply` idiom**
//! vendored from [NVIDIA Apex](https://github.com/NVIDIA/apex)
//! (BSD-3-Clause).
//!
//! ## Why this crate
//!
//! baracuda's main facade (`baracuda-kernels`) ships **zero**
//! optimizers — it's a kernel substrate, not a training framework.
//! Phase 49 added this sibling crate to give downstream training stacks
//! a fast optimizer step. The crate boundary is deliberate:
//!
//! - Inference-only consumers (e.g. Fuel) **don't pay** the FFI surface
//!   cost — they simply don't depend on this crate.
//! - The `optim` cargo feature on `baracuda-kernels` re-exports these
//!   plans into the unified facade when a downstream wants the full
//!   training surface.
//!
//! ## The multi-tensor apply idiom
//!
//! Without multi-tensor apply, the optimizer step on a 32B-parameter
//! model would launch **one kernel per parameter tensor** — that's
//! ~10,000 launches at ~5µs of launch overhead each = 50ms of
//! pure kernel-launch latency PER STEP. Multi-tensor apply collapses
//! that to a single launch (or a small handful of multi-launch batches
//! when the per-launch tensor cap of 110 is exceeded) by packing all
//! per-tensor pointers + sizes into one parameter struct.
//!
//! At training time this is the difference between 5% and 50% of the
//! step being optimizer overhead. The smoke test
//! `multi_tensor_dispatch_smoke` measures the speedup directly.
//!
//! ## What's exposed
//!
//! | Type | Update rule | Param dtype |
//! |---|---|---|
//! | [`AdamStepPlan`] | classic Adam or AdamW (mode flag) | f32 / f16 / bf16 |
//! | [`LambStepPlan`] | LAMB (Adam + per-layer trust ratio) | f32 (Phase 49) |
//! | [`SgdStepPlan`]  | SGD + momentum + Nesterov + weight decay | f32 / f16 / bf16 |
//!
//! Mixed-precision wiring: when param/grad is f16/bf16, the moments
//! (exp_avg / exp_avg_sq / momentum_buf) **must** stay in f32 — half-
//! precision moments lose precision catastrophically.
//!
//! ## License attribution
//!
//! Apex sources are vendored verbatim under
//! `vendor/apex/` (BSD-3-Clause). See `vendor/apex/VENDOR.md` for the
//! provenance, the trimmed scope, and how the original
//! `multi_tensor_apply<T>` host-side launcher was replaced by a
//! PyTorch-free C-ABI shim in `csrc/baracuda_optim_shim.cu`.

#![deny(missing_docs)]

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_driver::{DeviceBuffer, Stream};
use baracuda_types::DeviceRepr;
use thiserror::Error;

// ============================================================================
// Error / Result
// ============================================================================

/// Error category surfaced by the optimizer plans.
///
/// `#[non_exhaustive]` per the baracuda Phase 28 audit — variants may
/// be added as the surface grows (e.g. mixed-precision overflow
/// reporting once GradScaler integration ships).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// Caller-side validation failed before launch.
    #[error("optim argument invalid: {0}")]
    InvalidArgument(&'static str),
    /// One of the pointer-array length invariants didn't match (e.g.
    /// `params.len() != grads.len()`).
    #[error("tensor-list length mismatch: {0}")]
    LengthMismatch(&'static str),
    /// A tensor-list was empty.
    #[error("empty tensor list — at least one tensor required")]
    EmptyTensorList,
    /// The vendored C shim returned a non-zero status code.
    #[error("optim launch failed (status {0})")]
    LaunchFailed(i32),
}

/// `Result` alias used throughout the crate.
pub type Result<T> = core::result::Result<T, Error>;

// ============================================================================
// FFI extern block — matches csrc/baracuda_optim_shim.cu signatures
// ============================================================================

unsafe extern "C" {
    fn baracuda_optim_adam_f32_run(
        n_tensors: i32,
        sizes: *const i32,
        param_ptrs: *const *mut c_void,
        grad_ptrs: *const *mut c_void,
        exp_avg_ptrs: *const *mut c_void,
        exp_avg_sq_ptrs: *const *mut c_void,
        step: i32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        bias_correction: i32,
        adamw_mode: i32,
        stream: *mut c_void,
    ) -> i32;

    fn baracuda_optim_adam_f16_run(
        n_tensors: i32,
        sizes: *const i32,
        param_ptrs: *const *mut c_void,
        grad_ptrs: *const *mut c_void,
        exp_avg_ptrs: *const *mut c_void,
        exp_avg_sq_ptrs: *const *mut c_void,
        step: i32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        bias_correction: i32,
        adamw_mode: i32,
        stream: *mut c_void,
    ) -> i32;

    fn baracuda_optim_adam_bf16_run(
        n_tensors: i32,
        sizes: *const i32,
        param_ptrs: *const *mut c_void,
        grad_ptrs: *const *mut c_void,
        exp_avg_ptrs: *const *mut c_void,
        exp_avg_sq_ptrs: *const *mut c_void,
        step: i32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        bias_correction: i32,
        adamw_mode: i32,
        stream: *mut c_void,
    ) -> i32;

    fn baracuda_optim_sgd_f32_run(
        n_tensors: i32,
        sizes: *const i32,
        param_ptrs: *const *mut c_void,
        grad_ptrs: *const *mut c_void,
        momentum_ptrs: *const *mut c_void,
        lr: f32,
        momentum: f32,
        dampening: f32,
        weight_decay: f32,
        nesterov: i32,
        first_run: i32,
        wd_after_momentum: i32,
        grad_scale: f32,
        stream: *mut c_void,
    ) -> i32;

    fn baracuda_optim_sgd_f16_run(
        n_tensors: i32,
        sizes: *const i32,
        param_ptrs: *const *mut c_void,
        grad_ptrs: *const *mut c_void,
        momentum_ptrs: *const *mut c_void,
        lr: f32,
        momentum: f32,
        dampening: f32,
        weight_decay: f32,
        nesterov: i32,
        first_run: i32,
        wd_after_momentum: i32,
        grad_scale: f32,
        stream: *mut c_void,
    ) -> i32;

    fn baracuda_optim_sgd_bf16_run(
        n_tensors: i32,
        sizes: *const i32,
        param_ptrs: *const *mut c_void,
        grad_ptrs: *const *mut c_void,
        momentum_ptrs: *const *mut c_void,
        lr: f32,
        momentum: f32,
        dampening: f32,
        weight_decay: f32,
        nesterov: i32,
        first_run: i32,
        wd_after_momentum: i32,
        grad_scale: f32,
        stream: *mut c_void,
    ) -> i32;

    fn baracuda_optim_lamb_f32_run(
        n_tensors: i32,
        sizes: *const i32,
        param_ptrs: *const *mut c_void,
        grad_ptrs: *const *mut c_void,
        exp_avg_ptrs: *const *mut c_void,
        exp_avg_sq_ptrs: *const *mut c_void,
        u_scratch_ptrs: *const *mut c_void,
        u_scratch_host_to_dev: *mut c_void,
        w_norm_dev: *mut c_void,
        u_norm_dev: *mut c_void,
        step: i32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        bias_correction: i32,
        adamw_mode: i32,
        global_grad_norm: f32,
        max_global_grad_norm: f32,
        lr_lower_bound: f32,
        lr_upper_bound: f32,
        stream: *mut c_void,
    ) -> i32;

    /// Apex chunk size — currently 2048 elements per block.
    fn baracuda_optim_chunk_size() -> i32;

    /// Max tensors fitted into one multi_tensor_apply launch (110 per
    /// Apex's default).
    fn baracuda_optim_max_tensors_per_launch() -> i32;

    /// Max chunk-blocks fitted into one launch (320 per Apex).
    fn baracuda_optim_max_blocks_per_launch() -> i32;
}

// ============================================================================
// TensorList — opaque handle wrapping per-tensor device pointers + sizes
// ============================================================================

/// A list of device tensors of the same dtype, packed for a single
/// `multi_tensor_apply` launch.
///
/// This is **not** the on-device `TensorListMetadata<N>` pack — that
/// lives on-stack inside the shim TU during each launch. `TensorList`
/// is the host-side staging buffer the Rust plans use to hand pointer
/// arrays to the FFI.
///
/// `TensorList::new` borrows the underlying buffers; it does not take
/// ownership. The plan's `step` call must outlive the borrow.
pub struct TensorList<'a, T: DeviceRepr> {
    sizes: Vec<i32>,
    ptrs: Vec<*mut c_void>,
    _marker: PhantomData<&'a T>,
}

// SAFETY: the underlying pointers point at DeviceBuffer-owned memory
// that outlives the borrow (`'a`). The pointers themselves are stable
// for the borrow lifetime.
unsafe impl<T: DeviceRepr + Send> Send for TensorList<'_, T> {}

impl<'a, T: DeviceRepr> TensorList<'a, T> {
    /// Build a tensor list from a slice of `DeviceBuffer<T>` references.
    ///
    /// Returns [`Error::EmptyTensorList`] if the input slice is empty.
    pub fn new(buffers: &[&'a DeviceBuffer<T>]) -> Result<Self> {
        if buffers.is_empty() {
            return Err(Error::EmptyTensorList);
        }
        let sizes: Vec<i32> = buffers
            .iter()
            .map(|b| {
                let n = b.len();
                debug_assert!(
                    n <= i32::MAX as usize,
                    "TensorList tensor element count {n} exceeds i32::MAX"
                );
                n as i32
            })
            .collect();
        let ptrs: Vec<*mut c_void> = buffers
            .iter()
            .map(|b| b.as_raw().0 as *mut c_void)
            .collect();
        Ok(Self {
            sizes,
            ptrs,
            _marker: PhantomData,
        })
    }

    /// Build a tensor list from a slice of mutable `DeviceBuffer<T>` references.
    /// Required for buffers the optimizer step writes (params, moments).
    pub fn new_mut(buffers: &mut [&'a mut DeviceBuffer<T>]) -> Result<Self> {
        if buffers.is_empty() {
            return Err(Error::EmptyTensorList);
        }
        let sizes: Vec<i32> = buffers.iter().map(|b| b.len() as i32).collect();
        let ptrs: Vec<*mut c_void> = buffers
            .iter()
            .map(|b| b.as_raw().0 as *mut c_void)
            .collect();
        Ok(Self {
            sizes,
            ptrs,
            _marker: PhantomData,
        })
    }

    /// Number of tensors in the list.
    #[inline]
    pub fn len(&self) -> usize {
        self.sizes.len()
    }

    /// True if there are zero tensors. (Constructor rejects this — kept
    /// for parity with `Vec::is_empty`.)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }

    /// Total element count summed across all tensors.
    pub fn total_elements(&self) -> usize {
        self.sizes.iter().map(|&n| n as usize).sum()
    }

    /// Internal: pointer to the size array (i32, length == self.len()).
    #[inline]
    fn sizes_ptr(&self) -> *const i32 {
        self.sizes.as_ptr()
    }

    /// Internal: pointer to the pointer array (`*mut c_void`, length == self.len()).
    #[inline]
    fn ptrs_ptr(&self) -> *const *mut c_void {
        self.ptrs.as_ptr()
    }

    /// Snapshot of the per-tensor device pointers (as `u64`), suitable
    /// for staging into a `DeviceBuffer<u64>` for kernels that read
    /// `void**`.
    pub fn pointer_snapshot_u64(&self) -> Vec<u64> {
        self.ptrs.iter().map(|&p| p as u64).collect()
    }
}

// ============================================================================
// MultiTensorApplyContext — host-side dispatch constants
// ============================================================================

/// Geometry constants exposed by the Apex `multi_tensor_apply`
/// scaffold. Read once from the FFI on construction; cheap to copy.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct MultiTensorApplyContext {
    /// Per-block element chunk (2048 in current Apex).
    pub chunk_size: i32,
    /// Max tensors fitted into one launch metadata pack (110).
    pub max_tensors_per_launch: i32,
    /// Max chunk-blocks fitted into one launch (320).
    pub max_blocks_per_launch: i32,
}

impl MultiTensorApplyContext {
    /// Read the constants from the vendored shim.
    pub fn fetch() -> Self {
        // SAFETY: the three FFI calls are pure constant-returns; no
        // pointer dereferences, no CUDA context required.
        unsafe {
            Self {
                chunk_size: baracuda_optim_chunk_size(),
                max_tensors_per_launch: baracuda_optim_max_tensors_per_launch(),
                max_blocks_per_launch: baracuda_optim_max_blocks_per_launch(),
            }
        }
    }
}

// ============================================================================
// Common config / option types
// ============================================================================

/// Adam weight-decay mode.
///
/// `Classic` folds the L2 weight-decay term into the gradient before
/// the Adam math. `Decoupled` (a.k.a. AdamW per Loshchilov & Hutter
/// 2017) applies decay directly to the weight, independent of the
/// adaptive learning rate — empirically better for transformers.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
#[non_exhaustive]
pub enum AdamMode {
    /// Classic Adam: `g <- g + weight_decay * w`.
    #[default]
    Classic,
    /// AdamW: `w <- w - lr * (update + weight_decay * w)`.
    Decoupled,
}

impl AdamMode {
    #[inline]
    fn to_ffi(self) -> i32 {
        match self {
            AdamMode::Classic => 0,
            AdamMode::Decoupled => 1,
        }
    }
}

/// Adam hyperparameters.
#[derive(Copy, Clone, Debug)]
pub struct AdamConfig {
    /// Learning rate.
    pub lr: f32,
    /// First-moment exponential decay rate (default 0.9).
    pub beta1: f32,
    /// Second-moment exponential decay rate (default 0.999).
    pub beta2: f32,
    /// Numerical stability term added to the denominator (default 1e-8).
    pub epsilon: f32,
    /// Weight decay coefficient (default 0.0). Interpretation depends on
    /// [`AdamConfig::mode`].
    pub weight_decay: f32,
    /// Whether to apply bias correction. When false, the caller is
    /// expected to pre-scale lr.
    pub bias_correction: bool,
    /// Adam vs AdamW (decoupled decay).
    pub mode: AdamMode,
}

impl Default for AdamConfig {
    /// PyTorch / Apex default hyperparameters.
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            bias_correction: true,
            mode: AdamMode::Classic,
        }
    }
}

/// SGD hyperparameters.
#[derive(Copy, Clone, Debug)]
pub struct SgdConfig {
    /// Learning rate.
    pub lr: f32,
    /// Momentum coefficient (0 = vanilla SGD, no momentum buffer used).
    pub momentum: f32,
    /// Dampening on the gradient term inside the momentum buffer.
    pub dampening: f32,
    /// Weight decay coefficient (0 = no decay).
    pub weight_decay: f32,
    /// Nesterov accelerated gradient. Requires `momentum > 0`.
    pub nesterov: bool,
    /// Apex flag: apply weight decay AFTER the momentum buffer update
    /// rather than to the raw gradient. Matches the
    /// `momentum/weight_decay` ordering used in some pretraining
    /// recipes; off by default for PyTorch parity.
    pub weight_decay_after_momentum: bool,
    /// Mixed-precision gradient scale (1.0 = no scale; GradScaler users
    /// pass `1.0 / scale_factor`).
    pub grad_scale: f32,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            weight_decay_after_momentum: false,
            grad_scale: 1.0,
        }
    }
}

/// LAMB hyperparameters.
#[derive(Copy, Clone, Debug)]
pub struct LambConfig {
    /// Learning rate.
    pub lr: f32,
    /// First-moment decay rate.
    pub beta1: f32,
    /// Second-moment decay rate.
    pub beta2: f32,
    /// Numerical stability.
    pub epsilon: f32,
    /// Weight decay coefficient.
    pub weight_decay: f32,
    /// Bias correction.
    pub bias_correction: bool,
    /// Adam vs AdamW inner update mode.
    pub mode: AdamMode,
    /// LAMB global-gradient-norm clip threshold. If the precomputed
    /// `global_grad_norm` exceeds this, every gradient is pre-scaled.
    /// Set to `f32::INFINITY` (or any large value) to disable.
    pub max_global_grad_norm: f32,
    /// Lower bound on `trust_ratio` (typically 0.0).
    pub trust_lr_lower_bound: f32,
    /// Upper bound on `trust_ratio` (typically 10.0; some recipes use
    /// 1e10 for "effectively unbounded").
    pub trust_lr_upper_bound: f32,
}

impl Default for LambConfig {
    /// You et al. 2019 reference defaults.
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-6,
            weight_decay: 0.01,
            bias_correction: true,
            mode: AdamMode::Decoupled,
            max_global_grad_norm: 1.0,
            trust_lr_lower_bound: 0.0,
            trust_lr_upper_bound: 10.0,
        }
    }
}

// ============================================================================
// Adam — dtype-templated plan
// ============================================================================

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for half::f16 {}
    impl Sealed for half::bf16 {}
}

/// Sealed trait identifying which Adam FFI entry point services a given
/// parameter dtype. Implemented for `f32`, `half::f16`, `half::bf16`.
pub trait AdamParamDtype: sealed::Sealed + DeviceRepr {
    #[doc(hidden)]
    fn adam_ffi() -> AdamFfi;
}

#[doc(hidden)]
pub struct AdamFfi(
    pub  unsafe extern "C" fn(
        i32,
        *const i32,
        *const *mut c_void,
        *const *mut c_void,
        *const *mut c_void,
        *const *mut c_void,
        i32,
        f32,
        f32,
        f32,
        f32,
        f32,
        i32,
        i32,
        *mut c_void,
    ) -> i32,
);

impl AdamParamDtype for f32 {
    fn adam_ffi() -> AdamFfi {
        AdamFfi(baracuda_optim_adam_f32_run)
    }
}

impl AdamParamDtype for half::f16 {
    fn adam_ffi() -> AdamFfi {
        AdamFfi(baracuda_optim_adam_f16_run)
    }
}

impl AdamParamDtype for half::bf16 {
    fn adam_ffi() -> AdamFfi {
        AdamFfi(baracuda_optim_adam_bf16_run)
    }
}

/// Adam fused multi-tensor optimizer step.
///
/// Mixed precision: when `T` is `f16` / `bf16`, params + grads use the
/// half dtype while `exp_avg` / `exp_avg_sq` MUST be `f32` (the FFI
/// rejects anything else — the half-precision Adam moment is too
/// noisy to be useful).
#[derive(Clone, Debug)]
pub struct AdamStepPlan<T: AdamParamDtype> {
    cfg: AdamConfig,
    _marker: PhantomData<T>,
}

impl<T: AdamParamDtype> AdamStepPlan<T> {
    /// Build a plan with the given hyperparameters.
    pub fn new(cfg: AdamConfig) -> Self {
        Self {
            cfg,
            _marker: PhantomData,
        }
    }

    /// Read-only view of the active config.
    pub fn config(&self) -> &AdamConfig {
        &self.cfg
    }

    /// Run one optimizer step where every tensor list has the same
    /// dtype `T`. Use this for pure-f32 Adam.
    ///
    /// - `params`, `grads`, `exp_avg`, `exp_avg_sq`: dtype `T`.
    /// - `step_index`: 1-based step counter used for bias correction
    ///   (`1 - beta^step`).
    pub fn step(
        &self,
        params: &TensorList<'_, T>,
        grads: &TensorList<'_, T>,
        exp_avg: &TensorList<'_, T>,
        exp_avg_sq: &TensorList<'_, T>,
        step_index: i32,
        stream: &Stream,
    ) -> Result<()> {
        validate_quadruple(params, grads, exp_avg, exp_avg_sq)?;
        let n_tensors = params.len() as i32;
        // SAFETY: pointer arrays are valid for the launch duration
        // (the TensorList borrows outlive this synchronous call); stream
        // is borrowed from a live Stream; the FFI does host-side
        // validation on n_tensors > 0.
        let status = unsafe {
            let f = T::adam_ffi().0;
            f(
                n_tensors,
                params.sizes_ptr(),
                params.ptrs_ptr(),
                grads.ptrs_ptr(),
                exp_avg.ptrs_ptr(),
                exp_avg_sq.ptrs_ptr(),
                step_index,
                self.cfg.lr,
                self.cfg.beta1,
                self.cfg.beta2,
                self.cfg.epsilon,
                self.cfg.weight_decay,
                if self.cfg.bias_correction { 1 } else { 0 },
                self.cfg.mode.to_ffi(),
                stream.as_raw() as *mut c_void,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(Error::LaunchFailed(status))
        }
    }

    /// Run one step with an explicit f32 moment-buffer tensor list
    /// (use this on the half-precision plans — `AdamStepPlan::<f16>`
    /// / `AdamStepPlan::<bf16>`).
    pub fn step_with_f32_state(
        &self,
        params: &TensorList<'_, T>,
        grads: &TensorList<'_, T>,
        exp_avg: &TensorList<'_, f32>,
        exp_avg_sq: &TensorList<'_, f32>,
        step_index: i32,
        stream: &Stream,
    ) -> Result<()> {
        if params.len() != grads.len()
            || params.len() != exp_avg.len()
            || params.len() != exp_avg_sq.len()
        {
            return Err(Error::LengthMismatch(
                "params/grads/exp_avg/exp_avg_sq tensor counts differ",
            ));
        }
        let n_tensors = params.len() as i32;
        let status = unsafe {
            let f = T::adam_ffi().0;
            f(
                n_tensors,
                params.sizes_ptr(),
                params.ptrs_ptr(),
                grads.ptrs_ptr(),
                exp_avg.ptrs_ptr(),
                exp_avg_sq.ptrs_ptr(),
                step_index,
                self.cfg.lr,
                self.cfg.beta1,
                self.cfg.beta2,
                self.cfg.epsilon,
                self.cfg.weight_decay,
                if self.cfg.bias_correction { 1 } else { 0 },
                self.cfg.mode.to_ffi(),
                stream.as_raw() as *mut c_void,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(Error::LaunchFailed(status))
        }
    }
}

// ============================================================================
// SGD
// ============================================================================

/// Sealed trait identifying which SGD FFI entry point services a given
/// parameter dtype.
pub trait SgdParamDtype: sealed::Sealed + DeviceRepr {
    #[doc(hidden)]
    fn sgd_ffi() -> SgdFfi;
}

#[doc(hidden)]
pub struct SgdFfi(
    pub  unsafe extern "C" fn(
        i32,
        *const i32,
        *const *mut c_void,
        *const *mut c_void,
        *const *mut c_void,
        f32,
        f32,
        f32,
        f32,
        i32,
        i32,
        i32,
        f32,
        *mut c_void,
    ) -> i32,
);

impl SgdParamDtype for f32 {
    fn sgd_ffi() -> SgdFfi {
        SgdFfi(baracuda_optim_sgd_f32_run)
    }
}

impl SgdParamDtype for half::f16 {
    fn sgd_ffi() -> SgdFfi {
        SgdFfi(baracuda_optim_sgd_f16_run)
    }
}

impl SgdParamDtype for half::bf16 {
    fn sgd_ffi() -> SgdFfi {
        SgdFfi(baracuda_optim_sgd_bf16_run)
    }
}

/// SGD fused multi-tensor optimizer step.
#[derive(Clone, Debug)]
pub struct SgdStepPlan<T: SgdParamDtype> {
    cfg: SgdConfig,
    _marker: PhantomData<T>,
}

impl<T: SgdParamDtype> SgdStepPlan<T> {
    /// Build a plan with the given hyperparameters.
    pub fn new(cfg: SgdConfig) -> Self {
        Self {
            cfg,
            _marker: PhantomData,
        }
    }

    /// Active config (read-only).
    pub fn config(&self) -> &SgdConfig {
        &self.cfg
    }

    /// Run one optimizer step.
    ///
    /// `first_step` must be `true` on the very first call after the
    /// momentum buffer is allocated (uninitialized memory). Subsequent
    /// calls pass `false` — the buffer holds the previous step's
    /// velocity.
    pub fn step(
        &self,
        params: &TensorList<'_, T>,
        grads: &TensorList<'_, T>,
        momentum: &TensorList<'_, T>,
        first_step: bool,
        stream: &Stream,
    ) -> Result<()> {
        if params.len() != grads.len() || params.len() != momentum.len() {
            return Err(Error::LengthMismatch(
                "params/grads/momentum tensor counts differ",
            ));
        }
        let n_tensors = params.len() as i32;
        let status = unsafe {
            let f = T::sgd_ffi().0;
            f(
                n_tensors,
                params.sizes_ptr(),
                params.ptrs_ptr(),
                grads.ptrs_ptr(),
                momentum.ptrs_ptr(),
                self.cfg.lr,
                self.cfg.momentum,
                self.cfg.dampening,
                self.cfg.weight_decay,
                if self.cfg.nesterov { 1 } else { 0 },
                if first_step { 1 } else { 0 },
                if self.cfg.weight_decay_after_momentum {
                    1
                } else {
                    0
                },
                self.cfg.grad_scale,
                stream.as_raw() as *mut c_void,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(Error::LaunchFailed(status))
        }
    }

    /// f32-state variant for mixed-precision SGD (f16/bf16 params +
    /// grads with f32 momentum buffer).
    pub fn step_with_f32_momentum(
        &self,
        params: &TensorList<'_, T>,
        grads: &TensorList<'_, T>,
        momentum: &TensorList<'_, f32>,
        first_step: bool,
        stream: &Stream,
    ) -> Result<()> {
        if params.len() != grads.len() || params.len() != momentum.len() {
            return Err(Error::LengthMismatch(
                "params/grads/momentum tensor counts differ",
            ));
        }
        let n_tensors = params.len() as i32;
        let status = unsafe {
            let f = T::sgd_ffi().0;
            f(
                n_tensors,
                params.sizes_ptr(),
                params.ptrs_ptr(),
                grads.ptrs_ptr(),
                momentum.ptrs_ptr(),
                self.cfg.lr,
                self.cfg.momentum,
                self.cfg.dampening,
                self.cfg.weight_decay,
                if self.cfg.nesterov { 1 } else { 0 },
                if first_step { 1 } else { 0 },
                if self.cfg.weight_decay_after_momentum {
                    1
                } else {
                    0
                },
                self.cfg.grad_scale,
                stream.as_raw() as *mut c_void,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(Error::LaunchFailed(status))
        }
    }
}

// ============================================================================
// LAMB — f32 only in Phase 49
// ============================================================================

/// LAMB fused multi-tensor optimizer step. f32-only in Phase 49.
///
/// LAMB (You et al. 2019) is Adam + per-layer adaptive learning rate
/// scaling by the trust ratio `||w|| / ||u||`. Critical for large-batch
/// LLM pretraining (BERT was the canonical first user; T5/PaLM also).
///
/// ## Numerical caveats (from Apex's known edge cases)
///
/// - **Zero-norm fallback**: `||w|| == 0` or `||u|| == 0` ⇒
///   `trust_ratio = 1.0` (vanilla Adam). Triggers on freshly-
///   initialized layers and on zero-gradient layers respectively.
/// - **`atomicAdd` race for L2 norms**: the per-tensor `||w||²` and
///   `||u||²` are accumulated across all chunk-blocks via `atomicAdd`
///   on f32 device memory. Different scheduling orders give different
///   1–2-ulp results — LAMB is documented-robust to this; the deltas
///   are below the trust-ratio's threshold of meaningful change.
/// - **Bias correction interaction**: when `bias_correction = false`,
///   the caller must pre-scale `lr` exactly as in Adam.
/// - **Decoupled vs classic decay**: AdamW-mode (`mode = Decoupled`)
///   is the LAMB paper's recommended pairing; classic L2 decay is
///   numerically distinct and used less often.
///
/// ## Workspace contract
///
/// LAMB needs caller-supplied scratch:
/// - A per-parameter-tensor scratch buffer for the Adam update `u_t`
///   (same shape + dtype as params), passed as a [`TensorList`].
/// - A small device array of pointers indexing into those scratches
///   (a `DeviceBuffer<u64>` of length `n_tensors`, staged with
///   [`TensorList::pointer_snapshot_u64`]).
/// - Two `f32[num_tensors]` device buffers for the L2-norm
///   accumulators (`w_norm` and `u_norm`).
#[derive(Clone, Debug)]
pub struct LambStepPlan {
    cfg: LambConfig,
}

impl LambStepPlan {
    /// Build a plan with the given hyperparameters.
    pub fn new(cfg: LambConfig) -> Self {
        Self { cfg }
    }

    /// Active config (read-only).
    pub fn config(&self) -> &LambConfig {
        &self.cfg
    }

    /// Run one LAMB step.
    ///
    /// - `params`, `grads`, `exp_avg`, `exp_avg_sq`: f32 device tensors.
    /// - `u_scratch_tensors`: per-tensor Adam-update scratch; one
    ///   buffer per param tensor, matching shape and dtype.
    /// - `u_scratch_ptr_dev`: a device-resident `*mut *mut c_void` of
    ///   length `params.len()`, populated with the per-tensor scratch
    ///   pointers. Stage with [`TensorList::pointer_snapshot_u64`] →
    ///   `DeviceBuffer::from_slice` before the call.
    /// - `w_norm_dev` / `u_norm_dev`: device `f32[params.len()]`
    ///   scratch (the shim zeroes them internally each launch).
    /// - `global_grad_norm`: precomputed L2 norm of all gradients
    ///   (caller computes via baracuda-kernels reduce); used for the
    ///   global clip when it exceeds `cfg.max_global_grad_norm`.
    /// - `step_index`: 1-based step counter.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        params: &TensorList<'_, f32>,
        grads: &TensorList<'_, f32>,
        exp_avg: &TensorList<'_, f32>,
        exp_avg_sq: &TensorList<'_, f32>,
        u_scratch_tensors: &TensorList<'_, f32>,
        u_scratch_ptr_dev: &DeviceBuffer<u64>,
        w_norm_dev: &mut DeviceBuffer<f32>,
        u_norm_dev: &mut DeviceBuffer<f32>,
        step_index: i32,
        global_grad_norm: f32,
        stream: &Stream,
    ) -> Result<()> {
        validate_quadruple(params, grads, exp_avg, exp_avg_sq)?;
        if params.len() != u_scratch_tensors.len() {
            return Err(Error::LengthMismatch(
                "params and u_scratch_tensors have different counts",
            ));
        }
        if u_scratch_ptr_dev.len() < params.len() {
            return Err(Error::InvalidArgument(
                "u_scratch_ptr_dev shorter than params count",
            ));
        }
        if w_norm_dev.len() < params.len() || u_norm_dev.len() < params.len() {
            return Err(Error::InvalidArgument(
                "w_norm_dev / u_norm_dev shorter than params count",
            ));
        }
        let n_tensors = params.len() as i32;
        let status = unsafe {
            baracuda_optim_lamb_f32_run(
                n_tensors,
                params.sizes_ptr(),
                params.ptrs_ptr(),
                grads.ptrs_ptr(),
                exp_avg.ptrs_ptr(),
                exp_avg_sq.ptrs_ptr(),
                u_scratch_tensors.ptrs_ptr(),
                u_scratch_ptr_dev.as_raw().0 as *mut c_void,
                w_norm_dev.as_raw().0 as *mut c_void,
                u_norm_dev.as_raw().0 as *mut c_void,
                step_index,
                self.cfg.lr,
                self.cfg.beta1,
                self.cfg.beta2,
                self.cfg.epsilon,
                self.cfg.weight_decay,
                if self.cfg.bias_correction { 1 } else { 0 },
                self.cfg.mode.to_ffi(),
                global_grad_norm,
                self.cfg.max_global_grad_norm,
                self.cfg.trust_lr_lower_bound,
                self.cfg.trust_lr_upper_bound,
                stream.as_raw() as *mut c_void,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(Error::LaunchFailed(status))
        }
    }

    /// Workspace byte count for the LAMB step given `n_tensors` and the
    /// total element count of all params combined. Useful for
    /// pre-allocating the scratch buffers.
    ///
    /// Breakdown:
    /// - `n_tensors * 8` bytes for the device-side scratch-pointer
    ///   array (u64 per pointer).
    /// - `n_tensors * 4` bytes for `w_norm_dev`.
    /// - `n_tensors * 4` bytes for `u_norm_dev`.
    /// - `total_elements * 4` bytes for the per-tensor `u_scratch`
    ///   buffers (the caller usually allocates each scratch alongside
    ///   its param — this is the upper bound).
    pub fn workspace_bytes(n_tensors: usize, total_elements: usize) -> usize {
        n_tensors * 8 + n_tensors * 4 + n_tensors * 4 + total_elements * 4
    }
}

// ============================================================================
// Shared helpers
// ============================================================================

fn validate_quadruple<T: DeviceRepr>(
    a: &TensorList<'_, T>,
    b: &TensorList<'_, T>,
    c: &TensorList<'_, T>,
    d: &TensorList<'_, T>,
) -> Result<()> {
    if a.len() != b.len() || a.len() != c.len() || a.len() != d.len() {
        return Err(Error::LengthMismatch(
            "the four tensor lists must have the same length",
        ));
    }
    // Per-tensor element-count check.
    for i in 0..a.len() {
        let s = a.sizes[i];
        if b.sizes[i] != s || c.sizes[i] != s || d.sizes[i] != s {
            return Err(Error::LengthMismatch(
                "per-tensor element counts differ across the four lists",
            ));
        }
    }
    Ok(())
}
