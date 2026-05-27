//! CTCLoss — cuDNN-backed sibling of [`super::ctc::CtcLossPlan`].
//!
//! cuDNN exposes a single fused FW+BW CTC entry point
//! ([`baracuda_kernels_sys::cudnnCTCLoss`]) that consumes log-probs
//! and produces both per-sample losses and per-element gradients in
//! one shot. This plan wraps that entry point so Fuel's autotuner
//! can race it against the bespoke `CtcLossPlan` per-shape /
//! per-dtype.
//!
//! ## Convention parity with the bespoke plan
//!
//! The bespoke plan takes pre-log-softmaxed `log_probs`; cuDNN
//! supports the same convention via
//! [`baracuda_kernels_sys::CUDNN_LOSS_NORMALIZATION_SOFTMAX`] (=
//! "input is log-probs, cuDNN softmaxes internally"). We pin that
//! mode so the caller-visible input semantics match.
//!
//! ## Differences from the bespoke plan
//!
//! 1. **Host-side label arrays.** cuDNN's API takes `labels`,
//!    `label_lengths`, and `input_lengths` as *host* `i32` arrays
//!    (not device pointers, unlike the bespoke plan's `i64` device
//!    tensors). Callers stage their per-batch counts host-side.
//! 2. **Fused FW+BW.** A single `run` call produces both the
//!    per-sample loss vector and the full gradient tensor. There is
//!    no separate FW / BW split — `grads` is non-optional in this
//!    wrapper (cuDNN does support a null `grads` for FW-only, but
//!    Fuel always wants the gradient too, so we keep the API tight).
//! 3. **No `zero_infinity` / `reduction` modes.** cuDNN's CTC entry
//!    point always produces per-sample losses (no mean / sum reduce)
//!    and does not zero out infinite-loss samples. Callers wanting
//!    those behaviors layer them on the cuDNN output (a separate
//!    reduction kernel + mask) — keeping the cuDNN path thin
//!    matches the bespoke surface's reduction-as-separate-step
//!    spirit on the BW side.
//! 4. **f32 / f64 only.** cuDNN's CTC kernel does not accept f16 /
//!    bf16 log-probs. The bespoke plan covers those dtypes; this
//!    plan rejects them at `select`.
//! 5. **No bit-stability guarantee.** Even with
//!    `CUDNN_CTC_LOSS_ALGO_DETERMINISTIC` cuDNN does not publish a
//!    bit-stable contract across runs (the algorithm is single-run
//!    deterministic but internal reduction orderings may shift
//!    between cuDNN versions).

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    cudnnCTCLoss, cudnnCTCLossDescriptor_t, cudnnCreate, cudnnCreateCTCLossDescriptor,
    cudnnCreateTensorDescriptor, cudnnDestroy, cudnnDestroyCTCLossDescriptor,
    cudnnDestroyTensorDescriptor, cudnnGetCTCLossWorkspaceSize, cudnnHandle_t,
    cudnnSetCTCLossDescriptorEx, cudnnSetStream, cudnnSetTensorNdDescriptor,
    cudnnTensorDescriptor_t, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC,
    CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC, CUDNN_DATA_DOUBLE, CUDNN_DATA_FLOAT,
    CUDNN_LOSS_NORMALIZATION_SOFTMAX, CUDNN_NOT_PROPAGATE_NAN,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LossKind, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a cuDNN-backed CTC-loss op.
///
/// Shape semantics match the bespoke [`super::ctc::CtcLossDescriptor`]:
/// log-probs are `[max_input_length, batch, num_classes]` row-major,
/// `blank_index` is the index of the blank class in `[0,
/// num_classes)`, and `num_classes` includes the blank.
#[derive(Copy, Clone, Debug)]
pub struct CtcLossCudnnDescriptor {
    /// Batch size `B`.
    pub batch: i32,
    /// Maximum input timestep extent `T`.
    pub max_input_length: i32,
    /// Number of classes (including the blank class) `C`.
    pub num_classes: i32,
    /// Index of the blank class. Typically `0`. PyTorch default.
    pub blank_index: i32,
    /// Element dtype. Must be `F32` or `F64` — cuDNN's CTC kernel
    /// does not accept f16 / bf16.
    pub element: ElementKind,
    /// If `true`, request cuDNN's deterministic algorithm
    /// ([`CUDNN_CTC_LOSS_ALGO_DETERMINISTIC`]). Otherwise the
    /// non-deterministic algorithm is used (faster on large batches
    /// but uses atomicAdd reductions).
    pub deterministic: bool,
}

/// Args bundle for a fused cuDNN CTC FW+BW launch.
pub struct CtcLossCudnnArgs<'a, T: Element> {
    /// Log-probabilities `[T, B, C]` device tensor. cuDNN applies an
    /// internal log-softmax to recover the normalization (matching
    /// the bespoke plan's input convention).
    pub log_probs: TensorRef<'a, T, 3>,
    /// Host-side concatenated label sequences. Length equals
    /// `Σ label_lengths`.
    pub labels: &'a [i32],
    /// Host-side per-batch label lengths. Length equals `B`.
    pub label_lengths: &'a [i32],
    /// Host-side per-batch input lengths (actual `T` per sample, may
    /// be `≤ max_input_length`). Length equals `B`.
    pub input_lengths: &'a [i32],
    /// Per-batch loss vector `[B]` device tensor.
    pub costs: TensorMut<'a, T, 1>,
    /// Gradients w.r.t. `log_probs`, same shape as `log_probs`
    /// (`[T, B, C]`). Fully overwritten by the kernel.
    pub grads: TensorMut<'a, T, 3>,
}

/// cuDNN-backed CTC-loss plan. Owns one cuDNN handle, two tensor
/// descriptors (probs / grads — same shape, kept distinct so cuDNN's
/// API contract is honored), one CTC-loss descriptor, and a
/// workspace-size cache. Lifecycle parity with
/// [`super::super::conv::Conv2dPlan`].
pub struct CtcLossCudnnPlan<T: Element> {
    desc: CtcLossCudnnDescriptor,
    sku: KernelSku,
    handle: Cell<cudnnHandle_t>,
    probs_desc: Cell<cudnnTensorDescriptor_t>,
    grads_desc: Cell<cudnnTensorDescriptor_t>,
    ctc_desc: Cell<cudnnCTCLossDescriptor_t>,
    workspace_bytes: Cell<usize>,
    /// `true` once `workspace_bytes` has been populated by a
    /// [`Self::query_workspace_size`] call. We need this in addition
    /// to "size != 0" because a degenerate problem could legitimately
    /// return `0` bytes from cuDNN.
    workspace_queried: Cell<bool>,
    _marker: PhantomData<T>,
}

impl<T: Element> CtcLossCudnnPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &CtcLossCudnnDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::CtcLossCudnnPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::CtcLossCudnnPlan: cuDNN CTC supports f32 / f64 only \
                 (f16 / bf16 are bespoke-plan-only)",
            ));
        }
        if desc.batch < 0 || desc.max_input_length < 0 || desc.num_classes < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CtcLossCudnnPlan: dimensions must be non-negative",
            ));
        }
        if desc.num_classes > 0
            && (desc.blank_index < 0 || desc.blank_index >= desc.num_classes)
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CtcLossCudnnPlan: blank_index must be in [0, num_classes)",
            ));
        }

        let math_precision = match T::KIND {
            ElementKind::F64 => MathPrecision::F64,
            _ => MathPrecision::F32,
        };
        let accumulator = match T::KIND {
            ElementKind::F64 => ElementKind::F64,
            _ => ElementKind::F32,
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator,
            // cuDNN doesn't publish a bit-stable contract across runs
            // even for the deterministic algo.
            bit_stable_on_same_hardware: false,
            // Only the deterministic algo is determinism-safe; the
            // non-deterministic algo uses atomic adds.
            deterministic: desc.deterministic,
        };
        let sku = KernelSku {
            category: OpCategory::Loss,
            op: LossKind::Ctc as u16,
            element: T::KIND,
            aux_element: None,
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::Cudnn,
            precision_guarantee,
        };

        Ok(Self {
            desc: *desc,
            sku,
            handle: Cell::new(core::ptr::null_mut()),
            probs_desc: Cell::new(core::ptr::null_mut()),
            grads_desc: Cell::new(core::ptr::null_mut()),
            ctc_desc: Cell::new(core::ptr::null_mut()),
            workspace_bytes: Cell::new(0),
            workspace_queried: Cell::new(false),
            _marker: PhantomData,
        })
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

    /// Cached workspace size in bytes (`0` before the first
    /// [`Self::query_workspace_size`] call — but note that a
    /// degenerate problem may legitimately return `0` bytes; use
    /// [`Self::workspace_size_queried`] to disambiguate).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        self.workspace_bytes.get()
    }

    /// `true` once a workspace-size query has populated
    /// [`Self::workspace_size`].
    #[inline]
    pub fn workspace_size_queried(&self) -> bool {
        self.workspace_queried.get()
    }

    /// Materialize the cuDNN handle + descriptors and query the
    /// workspace size cuDNN's CTC fused kernel needs for this
    /// descriptor + the supplied per-batch label / input lengths.
    /// Caches the result.
    ///
    /// Note: cuDNN's workspace-size query *depends on the per-batch
    /// label content* (not just the shape), since the internal
    /// scratch is sized against the extended-target lattice. The
    /// caller must therefore pass the same `labels` / `label_lengths`
    /// / `input_lengths` arrays they will pass to [`Self::run`].
    pub fn query_workspace_size(
        &self,
        stream: &Stream,
        labels: &[i32],
        label_lengths: &[i32],
        input_lengths: &[i32],
    ) -> Result<usize> {
        self.check_host_arrays(labels, label_lengths, input_lengths)?;
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        self.ensure_descriptors()?;
        let algo = if self.desc.deterministic {
            CUDNN_CTC_LOSS_ALGO_DETERMINISTIC
        } else {
            CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC
        };
        let mut bytes: usize = 0;
        let status = unsafe {
            cudnnGetCTCLossWorkspaceSize(
                h,
                self.probs_desc.get(),
                self.grads_desc.get(),
                labels.as_ptr(),
                label_lengths.as_ptr(),
                input_lengths.as_ptr(),
                algo,
                self.ctc_desc.get(),
                &mut bytes as *mut usize,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        self.workspace_bytes.set(bytes);
        self.workspace_queried.set(true);
        Ok(bytes)
    }

    /// Run the fused CTC FW+BW pass. Writes `args.costs` and
    /// `args.grads`.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: CtcLossCudnnArgs<'_, T>,
    ) -> Result<()> {
        self.check_args(&args)?;
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        self.ensure_descriptors()?;

        let needed = if self.workspace_queried.get() {
            self.workspace_bytes.get()
        } else {
            self.query_workspace_size(
                stream,
                args.labels,
                args.label_lengths,
                args.input_lengths,
            )?
        };
        let (ws_ptr, _ws_bytes) = unpack_workspace(workspace, needed)?;

        let algo = if self.desc.deterministic {
            CUDNN_CTC_LOSS_ALGO_DETERMINISTIC
        } else {
            CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC
        };
        let status = unsafe {
            cudnnCTCLoss(
                h,
                self.probs_desc.get(),
                args.log_probs.data.as_raw().0 as *const c_void,
                args.labels.as_ptr(),
                args.label_lengths.as_ptr(),
                args.input_lengths.as_ptr(),
                args.costs.data.as_raw().0 as *mut c_void,
                self.grads_desc.get(),
                args.grads.data.as_raw().0 as *mut c_void,
                algo,
                self.ctc_desc.get(),
                ws_ptr,
                needed,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Internal: lazy handle + descriptors
    // ------------------------------------------------------------------

    fn ensure_handle(&self) -> Result<cudnnHandle_t> {
        let h = self.handle.get();
        if !h.is_null() {
            return Ok(h);
        }
        // Retry under transient cuDNN init failure. When many test
        // processes start in parallel (cargo test --workspace launches
        // dozens of binaries concurrently), `cudnnCreate` intermittently
        // returns CUDNN_STATUS_NOT_INITIALIZED (1001) — the library
        // races on first-process-touch driver init. Mirror Phase 30's
        // cuBLAS retry: five linear-backoff retries (50ms, 100ms, ...,
        // 250ms; worst-case ~750ms; one-time-per-plan cost).
        let mut last_status = 0;
        for attempt in 0..5 {
            let mut handle: cudnnHandle_t = core::ptr::null_mut();
            let status = unsafe { cudnnCreate(&mut handle as *mut _) };
            if status == 0 {
                self.handle.set(handle);
                return Ok(handle);
            }
            last_status = status;
            std::thread::sleep(std::time::Duration::from_millis(
                50 * (attempt as u64 + 1),
            ));
        }
        Err(Error::CutlassInternal(-last_status))
    }

    fn bind_stream(&self, h: cudnnHandle_t, stream: &Stream) -> Result<()> {
        let status = unsafe { cudnnSetStream(h, stream.as_raw() as *mut c_void) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }

    /// Allocate (once) and populate the two tensor descriptors plus
    /// the CTC-loss descriptor. Idempotent.
    fn ensure_descriptors(&self) -> Result<()> {
        if !self.ctc_desc.get().is_null() {
            return Ok(());
        }
        let dt = cudnn_dtype::<T>();
        let comp_dt = if matches!(T::KIND, ElementKind::F64) {
            CUDNN_DATA_DOUBLE
        } else {
            CUDNN_DATA_FLOAT
        };
        // [T, B, C] dims; contiguous (T-major) strides match
        // `contiguous_stride([T, B, C])`.
        let dims: [i32; 3] = [
            self.desc.max_input_length,
            self.desc.batch,
            self.desc.num_classes,
        ];
        let strides: [i32; 3] = [
            self.desc.batch * self.desc.num_classes,
            self.desc.num_classes,
            1,
        ];

        // probs descriptor.
        let mut pd: cudnnTensorDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateTensorDescriptor(&mut pd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let status = unsafe {
            cudnnSetTensorNdDescriptor(pd, dt, 3, dims.as_ptr(), strides.as_ptr())
        };
        if status != 0 {
            unsafe {
                let _ = cudnnDestroyTensorDescriptor(pd);
            }
            return Err(Error::CutlassInternal(-status));
        }
        self.probs_desc.set(pd);

        // grads descriptor — same shape but distinct handle (cuDNN
        // takes two descriptor args).
        let mut gd: cudnnTensorDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateTensorDescriptor(&mut gd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let status = unsafe {
            cudnnSetTensorNdDescriptor(gd, dt, 3, dims.as_ptr(), strides.as_ptr())
        };
        if status != 0 {
            unsafe {
                let _ = cudnnDestroyTensorDescriptor(gd);
            }
            return Err(Error::CutlassInternal(-status));
        }
        self.grads_desc.set(gd);

        // CTC-loss descriptor.
        let mut cd: cudnnCTCLossDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateCTCLossDescriptor(&mut cd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let status = unsafe {
            cudnnSetCTCLossDescriptorEx(
                cd,
                comp_dt,
                CUDNN_LOSS_NORMALIZATION_SOFTMAX,
                CUDNN_NOT_PROPAGATE_NAN,
            )
        };
        if status != 0 {
            unsafe {
                let _ = cudnnDestroyCTCLossDescriptor(cd);
            }
            return Err(Error::CutlassInternal(-status));
        }
        self.ctc_desc.set(cd);

        Ok(())
    }

    // ------------------------------------------------------------------
    // Internal: arg validation
    // ------------------------------------------------------------------

    fn check_args(&self, args: &CtcLossCudnnArgs<'_, T>) -> Result<()> {
        let probs_shape = [
            self.desc.max_input_length,
            self.desc.batch,
            self.desc.num_classes,
        ];
        if args.log_probs.shape != probs_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CtcLossCudnnPlan: log_probs shape != \
                 [max_input_length, batch, num_classes]",
            ));
        }
        if args.grads.shape != probs_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CtcLossCudnnPlan: grads shape != log_probs shape",
            ));
        }
        if args.costs.shape != [self.desc.batch] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CtcLossCudnnPlan: costs shape != [batch]",
            ));
        }
        self.check_host_arrays(args.labels, args.label_lengths, args.input_lengths)
    }

    fn check_host_arrays(
        &self,
        labels: &[i32],
        label_lengths: &[i32],
        input_lengths: &[i32],
    ) -> Result<()> {
        let b = self.desc.batch as usize;
        if label_lengths.len() != b {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CtcLossCudnnPlan: label_lengths.len() != batch",
            ));
        }
        if input_lengths.len() != b {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CtcLossCudnnPlan: input_lengths.len() != batch",
            ));
        }
        let total: i64 = label_lengths.iter().map(|&v| v as i64).sum();
        if (labels.len() as i64) != total {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CtcLossCudnnPlan: labels.len() != Σ label_lengths",
            ));
        }
        for &v in input_lengths {
            if v < 0 || v > self.desc.max_input_length {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::CtcLossCudnnPlan: input_lengths[i] out of \
                     [0, max_input_length]",
                ));
            }
        }
        for &v in label_lengths {
            if v < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::CtcLossCudnnPlan: label_lengths[i] < 0",
                ));
            }
        }
        for &v in labels {
            if v < 0 || v >= self.desc.num_classes {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::CtcLossCudnnPlan: labels[i] out of [0, num_classes)",
                ));
            }
        }
        Ok(())
    }
}

impl<T: Element> Drop for CtcLossCudnnPlan<T> {
    fn drop(&mut self) {
        let cd = self.ctc_desc.get();
        if !cd.is_null() {
            unsafe {
                let _ = cudnnDestroyCTCLossDescriptor(cd);
            }
            self.ctc_desc.set(core::ptr::null_mut());
        }
        let gd = self.grads_desc.get();
        if !gd.is_null() {
            unsafe {
                let _ = cudnnDestroyTensorDescriptor(gd);
            }
            self.grads_desc.set(core::ptr::null_mut());
        }
        let pd = self.probs_desc.get();
        if !pd.is_null() {
            unsafe {
                let _ = cudnnDestroyTensorDescriptor(pd);
            }
            self.probs_desc.set(core::ptr::null_mut());
        }
        let h = self.handle.get();
        if !h.is_null() {
            unsafe {
                let _ = cudnnDestroy(h);
            }
            self.handle.set(core::ptr::null_mut());
        }
    }
}

// ----- helpers --------------------------------------------------------

#[inline]
fn cudnn_dtype<T: Element>() -> i32 {
    match T::KIND {
        ElementKind::F32 => CUDNN_DATA_FLOAT,
        ElementKind::F64 => CUDNN_DATA_DOUBLE,
        _ => unreachable!("CtcLossCudnnPlan::select gates on F32 / F64"),
    }
}

/// Same contract as the conv module's local helper: `None` is valid
/// iff `needed == 0`, otherwise `Borrowed(s)` must have `s.len() >=
/// needed`.
fn unpack_workspace(workspace: Workspace<'_>, needed: usize) -> Result<(*mut c_void, usize)> {
    match workspace {
        Workspace::None => {
            if needed == 0 {
                Ok((core::ptr::null_mut(), 0))
            } else {
                Err(Error::WorkspaceTooSmall { needed, got: 0 })
            }
        }
        Workspace::Borrowed(slice) => {
            let got = slice.len();
            if got < needed {
                return Err(Error::WorkspaceTooSmall { needed, got });
            }
            Ok((slice.as_raw().0 as *mut c_void, got))
        }
    }
}
