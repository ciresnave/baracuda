//! Phase 47 — Fused Linear Cross-Entropy plan.
//!
//! Math / algorithm credit: LinkedIn Liger-Kernel (BSD-2-Clause),
//! [`fused_linear_cross_entropy.py`][liger-flce]. Triton-original by
//! Pin-Lun Hsu et al. (LinkedIn, 2024). This module is a clean-room
//! CUDA port — no Liger source is vendored.
//!
//! [liger-flce]: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py
//!
//! ## What it does
//!
//! Fuses the **last linear layer projection** (`logits = input @
//! weight^T`) **with the cross-entropy loss** into a single chunked
//! kernel, eliminating the `[batch×seq, vocab]` logits tensor that
//! the standard `Linear + CrossEntropy` pair materializes. At
//! Llama-3-class vocab (128K) and a 16K-token batch, that saves
//! **5–10 GiB of activation memory** — exactly the cliff that
//! prevents long-context LLM training on commodity GPUs.
//!
//! ## Algorithm
//!
//! `input` is `[BT, H]` (hidden states), `weight` is `[V, H]`
//! (lm_head weight, NOT yet transposed). The outer Rust loop tiles
//! over `BT` in chunks of `chunk_size` rows. Per chunk:
//!
//! 1. **Forward GEMM** — `logits_chunk = input_chunk @ weight^T`,
//!    shape `[chunk_size, V]`, stored in a single scratch buffer
//!    reused across chunks (this is the only `O(chunk_size · V)`
//!    allocation; `chunk_size` is sized so this fits in the same
//!    footprint as one `[BT, H]` activation).
//! 2. **Fused softmax + CE + gradient** — bespoke kernel
//!    `loss_flce_per_row_*_run` writes `(softmax - one_hot) ·
//!    scale` back over the same logits buffer (it's now
//!    `grad_logits_chunk`) and accumulates the per-row loss into a
//!    fp32 accumulator.
//! 3. **Backward GEMM #1** — `grad_input[chunk] = grad_logits_chunk
//!    @ weight`. Direct write into the caller's `grad_input`
//!    tensor.
//! 4. **Backward GEMM #2** — `grad_weight += grad_logits_chunk^T @
//!    input_chunk` (accumulating; β=1 across chunks).
//!
//! After the loop:
//! 5. Finalize: scalar reduce the per-row fp32 loss to `out` (with
//!    divide-by-N for `Mean`, or just sum for `Sum`, or per-row cast
//!    for `None`).
//!
//! `chunk_size` mirrors Liger's heuristic:
//! `chunk_size = next_pow2(ceildiv(BT, ceildiv(V, H)))`, capped to
//! 2048 (the value Liger empirically settled on for occupancy).
//!
//! ## Backward pass
//!
//! Because the FW pass already produces `grad_input` and
//! `grad_weight` for the case `dy = 1.0` (the typical "CE is the
//! last layer" case), the BW pass just multiplies the saved
//! gradients by the upstream scalar `dy`. This matches Liger's
//! design — `LigerFusedLinearCrossEntropyFunction.backward` is
//! exactly two `element_mul_kernel` launches.
//!
//! ## Numerical contract
//!
//! Loss is fp32-accumulated across the per-chunk fused step and the
//! scalar finalize. f16 / bf16 paths use fp32 math for the
//! softmax / exp / log, matching the precision contract of
//! [`crate::CrossEntropyLossPlan`]. Results are not bit-equivalent
//! to the unfused `Linear + CE` reference because the chunked GEMM
//! has a different reduction order than the un-chunked GEMM; the
//! error bound is approximately `K · eps · vocab` (with `K` the
//! hidden dim, `eps = 2^-23` for fp32 / `2^-10` for fp16 /
//! `2^-7` for bf16) — equivalent to any reduction-order
//! difference. The smoke tests use this bound.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::{Context, DeviceBuffer, Stream};
use baracuda_kernels_sys::{
    cublasCreate_v2, cublasDestroy_v2, cublasGemmEx, cublasHandle_t, cublasSetStream_v2,
    CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT, CUBLAS_OP_N, CUBLAS_OP_T,
    CUDA_R_16BF, CUDA_R_16F, CUDA_R_32F, CUDA_R_64F,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LossKind, LossReduction, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::common::check_supported_dtype;

/// The PyTorch / Liger default ignore_index sentinel: a target value
/// of `-100` marks "skip this token in the loss".
pub const FLCE_DEFAULT_IGNORE_INDEX: i64 = -100;

/// Maximum `chunk_size` row count for the chunked outer loop.
///
/// Liger uses `2048` (= `MAX_FUSED_SIZE / 32`, where the `/32` is for
/// the Triton block-size headroom). We use the same cap.
const MAX_CHUNK_ROWS: i32 = 2048;

/// Descriptor for the Fused Linear Cross-Entropy forward op.
///
/// `BT` is "batch × sequence" rows of hidden states, `H` is hidden
/// dim, `V` is vocab. The `weight` tensor is the lm_head weight as
/// stored by PyTorch's `nn.Linear` — shape `[V, H]` (the transpose
/// of what GEMM needs; we fold the transpose into the GEMM layout).
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct FusedLinearCrossEntropyDescriptor {
    /// `BT` — batch × sequence rows of hidden states.
    pub bt: i32,
    /// `H` — hidden dimension.
    pub h: i32,
    /// `V` — vocab size / number of classes.
    pub v: i32,
    /// Reduction mode.
    pub reduction: LossReduction,
    /// Element dtype of input / weight / out.
    pub element: ElementKind,
    /// Class index value that marks "skip this token". PyTorch
    /// default is `-100` (see [`FLCE_DEFAULT_IGNORE_INDEX`]).
    pub ignore_index: i64,
}

impl FusedLinearCrossEntropyDescriptor {
    /// Constructor with PyTorch / Liger defaults
    /// (`reduction = Mean`, `ignore_index = -100`).
    #[inline]
    pub fn new(bt: i32, h: i32, v: i32, element: ElementKind) -> Self {
        Self {
            bt,
            h,
            v,
            reduction: LossReduction::Mean,
            element,
            ignore_index: FLCE_DEFAULT_IGNORE_INDEX,
        }
    }

    /// Builder: set the reduction mode.
    #[inline]
    #[must_use]
    pub fn with_reduction(mut self, reduction: LossReduction) -> Self {
        self.reduction = reduction;
        self
    }

    /// Builder: set the ignore_index.
    #[inline]
    #[must_use]
    pub fn with_ignore_index(mut self, ignore_index: i64) -> Self {
        self.ignore_index = ignore_index;
        self
    }
}

/// Args for a [`FusedLinearCrossEntropyPlan::run`] launch.
///
/// `grad_input` and `grad_weight` are **always written** if `Some`
/// — they're free given the BW-during-FW design. Pass `None` for
/// either to skip the corresponding GEMM (inference-only path).
pub struct FusedLinearCrossEntropyArgs<'a, T: Element> {
    /// Hidden states. Row-major `[BT, H]`, contiguous.
    pub input: TensorRef<'a, T, 2>,
    /// LM-head weight. Row-major `[V, H]`, contiguous.
    pub weight: TensorRef<'a, T, 2>,
    /// Target class indices. `i64[BT]`. Values outside `[0, V)`
    /// other than `ignore_index` are silently zeroed (matches the
    /// `CrossEntropyLossPlan` contract).
    pub target: TensorRef<'a, i64, 1>,
    /// Output loss. Shape `[BT]` for `None` reduction, `[1]` for
    /// `Mean` / `Sum`.
    pub out: TensorMut<'a, T, 1>,
    /// Gradient w.r.t. input. Same shape as `input`. Caller owns
    /// the buffer; this plan writes it. Pass `None` to skip the
    /// `grad_input` GEMM (inference / eval).
    pub grad_input: Option<TensorMut<'a, T, 2>>,
    /// Gradient w.r.t. weight. Same shape as `weight`. **Must be
    /// pre-zeroed** by the caller (accumulating GEMM across chunks).
    /// Pass `None` to skip the `grad_weight` GEMM (frozen lm_head).
    pub grad_weight: Option<TensorMut<'a, T, 2>>,
}

/// Plan for Fused Linear Cross-Entropy.
///
/// Owns a lazy cuBLAS handle (`!Sync` / `!Send`); destroyed on `Drop`.
pub struct FusedLinearCrossEntropyPlan<T: Element> {
    desc: FusedLinearCrossEntropyDescriptor,
    sku: KernelSku,
    chunk_size: i32,
    handle: Cell<cublasHandle_t>,
    _marker: PhantomData<T>,
}

impl<T: Element> FusedLinearCrossEntropyPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &FusedLinearCrossEntropyDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::FusedLinearCrossEntropyPlan: descriptor.element != T",
            ));
        }
        check_supported_dtype::<T>()?;
        if desc.bt < 0 || desc.h < 1 || desc.v < 1 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FusedLinearCrossEntropyPlan: bt must be ≥ 0; h, v must be ≥ 1",
            ));
        }
        let chunk_size = pick_chunk_size(desc.bt, desc.h, desc.v);
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: if T::KIND == ElementKind::F64 {
                ElementKind::F64
            } else {
                ElementKind::F32
            },
            // The chunked GEMM has a different reduction order than the
            // un-chunked reference, so we cannot promise bit-stability
            // here. Deterministic per launch though (no atomicAdd).
            bit_stable_on_same_hardware: false,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Loss,
            op: LossKind::FusedLinearCrossEntropy as u16,
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
            chunk_size,
            handle: Cell::new(core::ptr::null_mut()),
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

    /// Chunk size picked at `select` time.
    ///
    /// Inspect-only — used by tests + by the memory-savings smoke
    /// test to assert the per-chunk logits buffer is small.
    #[inline]
    pub fn chunk_size(&self) -> i32 {
        self.chunk_size
    }

    /// Total workspace bytes required at `run` time.
    ///
    /// Currently returns **0** — the plan allocates the per-chunk
    /// scratch buffers internally via `DeviceBuffer::zeros` (same
    /// pattern as `KthvaluePlan`). A future phase could thread these
    /// through the public `Workspace` so callers can pool the
    /// allocation; for Phase 47 the simpler internal-alloc path is
    /// preferred. Reported size if you want to pre-allocate
    /// externally:
    ///   - `logits_scratch[chunk_size * V]` of `T`
    ///   - `loss_1d[BT]` of `f32`
    ///   - `count[1]` of `i64`
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Reports the **conceptual** scratch the plan needs internally
    /// (informational; not enforced by `run`). Useful for callers
    /// wanting to assert against the "no logits materialization"
    /// invariant.
    #[inline]
    pub fn conceptual_scratch_bytes(&self) -> usize {
        let elem_t = core::mem::size_of::<T>();
        let logits_bytes = (self.chunk_size as usize) * (self.desc.v as usize) * elem_t;
        let loss_bytes = (self.desc.bt as usize) * core::mem::size_of::<f32>();
        logits_bytes + loss_bytes + 8 // 8 = sizeof(i64) for count
    }

    fn ensure_handle(&self) -> Result<cublasHandle_t> {
        let h = self.handle.get();
        if !h.is_null() {
            return Ok(h);
        }
        let mut handle: cublasHandle_t = core::ptr::null_mut();
        // 5x linear-backoff retry — see Phase 35 memory entry for the
        // parallel-init race that motivated this in cuBLAS handle creation.
        let mut last_status = 0;
        for attempt in 0..5 {
            let status = unsafe { cublasCreate_v2(&mut handle as *mut _) };
            if status == 0 {
                last_status = 0;
                break;
            }
            last_status = status;
            std::thread::sleep(std::time::Duration::from_millis(
                10u64 * (attempt as u64 + 1),
            ));
        }
        if last_status != 0 {
            return Err(Error::CutlassInternal(-last_status));
        }
        self.handle.set(handle);
        Ok(handle)
    }

    fn bind_stream(&self, h: cublasHandle_t, stream: &Stream) -> Result<()> {
        let status = unsafe { cublasSetStream_v2(h, stream.as_raw() as *mut c_void) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: FusedLinearCrossEntropyArgs<'_, T>,
    ) -> Result<()> {
        // ---- Shape validation -------------------------------------
        let bt = self.desc.bt;
        let h = self.desc.h;
        let v = self.desc.v;
        if args.input.shape != [bt, h] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FusedLinearCrossEntropyPlan: input shape != [bt, h]",
            ));
        }
        if args.weight.shape != [v, h] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FusedLinearCrossEntropyPlan: weight shape != [v, h]",
            ));
        }
        if args.target.shape != [bt] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FusedLinearCrossEntropyPlan: target shape != [bt]",
            ));
        }
        if !args.input.is_contiguous() || !args.weight.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::FusedLinearCrossEntropyPlan: input and weight must be \
                 contiguous (Phase 47 v1 limitation)",
            ));
        }
        // Output shape per reduction mode.
        let expected_out_n = match self.desc.reduction {
            LossReduction::None => bt,
            LossReduction::Mean | LossReduction::Sum => 1,
        };
        if args.out.shape != [expected_out_n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FusedLinearCrossEntropyPlan: out shape mismatch (expected \
                 [BT] for None or [1] for Mean/Sum)",
            ));
        }
        if let Some(ref gi) = args.grad_input {
            if gi.shape != [bt, h] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::FusedLinearCrossEntropyPlan: grad_input shape != [bt, h]",
                ));
            }
        }
        if let Some(ref gw) = args.grad_weight {
            if gw.shape != [v, h] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::FusedLinearCrossEntropyPlan: grad_weight shape != [v, h]",
                ));
            }
        }

        if bt == 0 {
            return Ok(());
        }

        // ---- Allocate the per-chunk scratch (logits_chunk) + loss_1d ----
        let ctx = stream.context();
        let chunk_size = self.chunk_size;
        let logits_elems = (chunk_size as usize) * (v as usize);
        let mut logits_scratch: DeviceBuffer<T> =
            DeviceBuffer::zeros(ctx, logits_elems).map_err(|_| {
                Error::InvalidProblem(
                    "baracuda-kernels::FusedLinearCrossEntropyPlan: logits scratch alloc failed",
                )
            })?;
        let mut loss_1d: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, bt as usize).map_err(|_| {
            Error::InvalidProblem(
                "baracuda-kernels::FusedLinearCrossEntropyPlan: loss_1d alloc failed",
            )
        })?;

        // ---- Compute N_non_ignore on host (CPU pass over target) ----
        // Triton-Liger does it on-device with `.sum().item()`, which is
        // a sync. We do it on host via a single D2H of the target
        // buffer, which is equivalent latency at typical BT (a few
        // hundred microseconds) and avoids the bespoke reduction
        // kernel. Could be optimized to a fused device-side count
        // in a future phase; the mean denominator just needs to be
        // available before the per-chunk fused kernel launches (so
        // it can pre-fold `1/N` into the gradient).
        let n_non_ignore = self.count_non_ignore(ctx, stream, &args.target)?;
        if n_non_ignore == 0 {
            // Every token is ignored. Zero the output and any
            // gradients; nothing to compute.
            self.zero_outputs(stream, &args)?;
            return Ok(());
        }

        // ---- Per-chunk scale_per_row (folded into gradient) -----------
        //
        //   None mode:    each row's gradient should equal (softmax - one_hot)
        //                 (caller multiplies by per-token upstream dy later).
        //                 -> scale_per_row = 1.0
        //   Mean mode:    grad = (softmax - one_hot) / N_non_ignore
        //                 -> scale_per_row = 1 / N_non_ignore
        //   Sum mode:     grad = (softmax - one_hot)
        //                 -> scale_per_row = 1.0
        let scale_per_row: f32 = match self.desc.reduction {
            LossReduction::Mean => 1.0f32 / (n_non_ignore as f32),
            LossReduction::None | LossReduction::Sum => 1.0f32,
        };

        // ---- cuBLAS handle bound to the stream -------------------------
        let handle = self.ensure_handle()?;
        self.bind_stream(handle, stream)?;

        // ---- Chunked outer loop ---------------------------------------
        let chunk_size_u = chunk_size as i32;
        let n_chunks = (bt + chunk_size_u - 1) / chunk_size_u;

        // Raw pointers for the GEMM dispatch — kept here so the loop
        // body stays readable.
        let input_ptr_base = args.input.data.as_raw().0 as *const c_void;
        let weight_ptr = args.weight.data.as_raw().0 as *const c_void;
        let target_ptr = args.target.data.as_raw().0 as *const c_void;
        let logits_ptr = logits_scratch.as_slice_mut().as_raw().0 as *mut c_void;
        let loss_1d_ptr = loss_1d.as_slice_mut().as_raw().0 as *mut c_void;
        let grad_input_ptr_base = args
            .grad_input
            .as_ref()
            .map(|gi| gi.data.as_raw().0 as *mut c_void)
            .unwrap_or(core::ptr::null_mut());
        let grad_weight_ptr = args
            .grad_weight
            .as_ref()
            .map(|gw| gw.data.as_raw().0 as *mut c_void)
            .unwrap_or(core::ptr::null_mut());

        let elem_t = core::mem::size_of::<T>() as isize;
        let input_row_stride_elems = args.input.stride[0] as isize;
        let grad_input_row_stride_elems = args
            .grad_input
            .as_ref()
            .map(|gi| gi.stride[0] as isize)
            .unwrap_or(0);

        for chunk_id in 0..n_chunks {
            let start = chunk_id * chunk_size_u;
            let end = core::cmp::min((chunk_id + 1) * chunk_size_u, bt);
            let n_rows = end - start;
            if n_rows == 0 {
                break;
            }

            // ---- 1. logits_chunk = input_chunk @ weight^T -------------
            //
            // input_chunk: row-major [n_rows, H], leading dim = H.
            // weight:      row-major [V, H], leading dim = H.
            // logits:      row-major [n_rows, V], leading dim = V.
            //
            // cuBLAS is column-major. We want to compute the row-major
            // product `D = A · B^T` where A is row-major [n, H] and B is
            // row-major [V, H] (so we need B^T which is [H, V]).
            //
            // Row→col view: `row[r, c]` storage == `col[c, r]` storage.
            //   A (row [n, H]) → A_col [H, n], lda = H.
            //   B (row [V, H]) → B_col [H, V], ldb = H.
            //   D (row [n, V]) → D_col [V, n], ldc = V.
            // We want D_col[V, n] = B_col^T [V, H] · A_col [H, n].
            //
            // In cuBLAS terms `C = α op(A_cublas) · op(B_cublas) + β C`:
            //   A_cublas = weight (storage B_col [H, V]) with transa=OP_T
            //     → op(A_cublas)[V, H].
            //   B_cublas = input (storage A_col [H, n]) with transb=OP_N
            //     → op(B_cublas)[H, n].
            //   m = V, n = n_rows, k = H, lda = H, ldb = H, ldc = V.
            //
            // (Consolidation-pass note: this used to read OP_N / OP_T
            // for transa / transb — the comment block above was correct
            // about the math but the cuBLAS arg names had been swapped,
            // which fired `CUBLAS_STATUS_INVALID_VALUE` whenever V > H
            // because lda < m. Fixed by Consolidation Agent C.)
            let input_chunk_ptr = unsafe {
                (input_ptr_base as *const u8)
                    .offset(start as isize * input_row_stride_elems * elem_t)
                    as *const c_void
            };
            let alpha_f32 = 1.0f32;
            let beta_zero_f32 = 0.0f32;
            let alpha_f64 = 1.0f64;
            let beta_zero_f64 = 0.0f64;
            self.gemm_ex(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                v,
                n_rows,
                h,
                if T::KIND == ElementKind::F64 {
                    &alpha_f64 as *const f64 as *const c_void
                } else {
                    &alpha_f32 as *const f32 as *const c_void
                },
                weight_ptr,
                v,           // m marker (informational)
                h as i32,    // lda (storage leading-dim of weight col-view [H, V])
                input_chunk_ptr,
                h as i32,    // ldb (storage leading-dim of input col-view [H, n])
                if T::KIND == ElementKind::F64 {
                    &beta_zero_f64 as *const f64 as *const c_void
                } else {
                    &beta_zero_f32 as *const f32 as *const c_void
                },
                logits_ptr,
                v,           // ldc
            )?;

            // ---- 2. Fused per-row softmax + CE + gradient ------------
            //
            // Compute the loss_1d slice pointer (offset by `start` rows).
            let loss_1d_chunk_ptr = unsafe {
                (loss_1d_ptr as *mut u8).offset(start as isize * core::mem::size_of::<f32>() as isize)
                    as *mut c_void
            };
            let target_chunk_ptr = unsafe {
                (target_ptr as *const u8)
                    .offset(start as isize * core::mem::size_of::<i64>() as isize)
                    as *const c_void
            };
            let row_stride_logits = v as i64;
            let status = unsafe {
                match T::KIND {
                    ElementKind::F32 => baracuda_kernels_sys::baracuda_kernels_loss_flce_per_row_f32_run(
                        n_rows, v, row_stride_logits, self.desc.ignore_index, scale_per_row,
                        logits_ptr, target_chunk_ptr, loss_1d_chunk_ptr,
                        stream.as_raw() as *mut c_void,
                    ),
                    ElementKind::F16 => baracuda_kernels_sys::baracuda_kernels_loss_flce_per_row_f16_run(
                        n_rows, v, row_stride_logits, self.desc.ignore_index, scale_per_row,
                        logits_ptr, target_chunk_ptr, loss_1d_chunk_ptr,
                        stream.as_raw() as *mut c_void,
                    ),
                    ElementKind::Bf16 => baracuda_kernels_sys::baracuda_kernels_loss_flce_per_row_bf16_run(
                        n_rows, v, row_stride_logits, self.desc.ignore_index, scale_per_row,
                        logits_ptr, target_chunk_ptr, loss_1d_chunk_ptr,
                        stream.as_raw() as *mut c_void,
                    ),
                    ElementKind::F64 => baracuda_kernels_sys::baracuda_kernels_loss_flce_per_row_f64_run(
                        n_rows, v, row_stride_logits, self.desc.ignore_index, scale_per_row,
                        logits_ptr, target_chunk_ptr, loss_1d_chunk_ptr,
                        stream.as_raw() as *mut c_void,
                    ),
                    _ => return Err(Error::Unsupported(
                        "baracuda-kernels::FusedLinearCrossEntropyPlan::run unwired dtype",
                    )),
                }
            };
            if status != 0 {
                return Err(Error::CutlassInternal(status));
            }

            // ---- 3. grad_input[chunk] = grad_logits_chunk @ weight ----
            //
            // Row→col view:
            //   grad_logits_chunk (row [n, V]) → col [V, n], ld = V.
            //   weight            (row [V, H]) → col [H, V], ld = H.
            //   grad_input_chunk  (row [n, H]) → col [H, n], ld = H.
            //
            // grad_input_col[H, n] = weight_col[H, V] · grad_logits_col[V, n]
            // → transa = OP_N (weight_col is already [H, V]),
            //   transb = OP_N (grad_logits_col is already [V, n]),
            //   m = H, n = n_rows, k = V,
            //   lda = H, ldb = V, ldc = H.
            //
            // (Consolidation-pass note: previous code already had the
            // correct transa/transb for this call but the comment block
            // was confused — clarified above so the next reviewer can
            // skip the derivation.)
            if let Some(_) = args.grad_input.as_ref() {
                let grad_input_chunk_ptr = unsafe {
                    (grad_input_ptr_base as *mut u8)
                        .offset(start as isize * grad_input_row_stride_elems * elem_t)
                        as *mut c_void
                };
                self.gemm_ex(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    h,
                    n_rows,
                    v,
                    if T::KIND == ElementKind::F64 {
                        &alpha_f64 as *const f64 as *const c_void
                    } else {
                        &alpha_f32 as *const f32 as *const c_void
                    },
                    weight_ptr,
                    h,           // m marker
                    h as i32,    // lda (storage of weight col-view [H, V])
                    logits_ptr,
                    v as i32,    // ldb (storage of grad_logits col-view [V, n])
                    if T::KIND == ElementKind::F64 {
                        &beta_zero_f64 as *const f64 as *const c_void
                    } else {
                        &beta_zero_f32 as *const f32 as *const c_void
                    },
                    grad_input_chunk_ptr,
                    h as i32,    // ldc
                )?;
            }

            // ---- 4. grad_weight += grad_logits_chunk^T @ input_chunk ----
            //
            // grad_logits_chunk: row-major [n, V] = col-major [V, n], ld = V.
            // input_chunk:       row-major [n, H] = col-major [H, n], ld = H.
            // grad_weight:       row-major [V, H] = col-major [H, V], ld = H.
            //
            // We want grad_weight[V, H] += grad_logits^T @ input
            //   = grad_logits[BT, V]^T @ input[BT, H]  shape [V, H].
            //
            // In col-major (transposed):
            //   D[H, V] += input^T_col [H, n] · grad_logits_col [V, n]^T
            //
            // Concretely D = col-major [H, V] with ld=H:
            //   m = H, n = V, k = n_rows.
            //   transa = N on input (col-major [H, n], lda = H).
            //   transb = T on grad_logits (col-major [V, n], transpose to [n, V], ldb = V).
            // Output: D col-major [H, V] = row-major [V, H], ldc = H. β = 1 (accumulate).
            if let Some(_) = args.grad_weight.as_ref() {
                let beta_one_f32 = 1.0f32;
                let beta_one_f64 = 1.0f64;
                self.gemm_ex(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    h,
                    v,
                    n_rows,
                    if T::KIND == ElementKind::F64 {
                        &alpha_f64 as *const f64 as *const c_void
                    } else {
                        &alpha_f32 as *const f32 as *const c_void
                    },
                    input_chunk_ptr,
                    h,            // m
                    h as i32,     // lda
                    logits_ptr,
                    v as i32,     // ldb
                    if T::KIND == ElementKind::F64 {
                        &beta_one_f64 as *const f64 as *const c_void
                    } else {
                        &beta_one_f32 as *const f32 as *const c_void
                    },
                    grad_weight_ptr,
                    h as i32,     // ldc
                )?;
            }
        }

        // ---- 5. Finalize: loss_1d → scalar / per-row out ----------------
        let out_ptr = args.out.data.as_raw().0 as *mut c_void;
        let bt_i64 = bt as i64;
        let status = match self.desc.reduction {
            LossReduction::None => unsafe {
                match T::KIND {
                    ElementKind::F32 => baracuda_kernels_sys::baracuda_kernels_loss_flce_per_row_cast_f32_run(
                        bt_i64, loss_1d_ptr as *const c_void, out_ptr, stream.as_raw() as *mut c_void),
                    ElementKind::F16 => baracuda_kernels_sys::baracuda_kernels_loss_flce_per_row_cast_f16_run(
                        bt_i64, loss_1d_ptr as *const c_void, out_ptr, stream.as_raw() as *mut c_void),
                    ElementKind::Bf16 => baracuda_kernels_sys::baracuda_kernels_loss_flce_per_row_cast_bf16_run(
                        bt_i64, loss_1d_ptr as *const c_void, out_ptr, stream.as_raw() as *mut c_void),
                    ElementKind::F64 => baracuda_kernels_sys::baracuda_kernels_loss_flce_per_row_cast_f64_run(
                        bt_i64, loss_1d_ptr as *const c_void, out_ptr, stream.as_raw() as *mut c_void),
                    _ => return Err(Error::Unsupported("unwired dtype")),
                }
            },
            LossReduction::Mean | LossReduction::Sum => {
                let denom_inv = match self.desc.reduction {
                    LossReduction::Mean => 1.0f32 / (n_non_ignore as f32),
                    _ => 1.0f32,
                };
                unsafe {
                    match T::KIND {
                        ElementKind::F32 => baracuda_kernels_sys::baracuda_kernels_loss_flce_scalar_finalize_f32_run(
                            bt_i64, denom_inv, loss_1d_ptr as *const c_void, out_ptr, stream.as_raw() as *mut c_void),
                        ElementKind::F16 => baracuda_kernels_sys::baracuda_kernels_loss_flce_scalar_finalize_f16_run(
                            bt_i64, denom_inv, loss_1d_ptr as *const c_void, out_ptr, stream.as_raw() as *mut c_void),
                        ElementKind::Bf16 => baracuda_kernels_sys::baracuda_kernels_loss_flce_scalar_finalize_bf16_run(
                            bt_i64, denom_inv, loss_1d_ptr as *const c_void, out_ptr, stream.as_raw() as *mut c_void),
                        ElementKind::F64 => baracuda_kernels_sys::baracuda_kernels_loss_flce_scalar_finalize_f64_run(
                            bt_i64, denom_inv, loss_1d_ptr as *const c_void, out_ptr, stream.as_raw() as *mut c_void),
                        _ => return Err(Error::Unsupported("unwired dtype")),
                    }
                }
            }
        };
        if status != 0 {
            return Err(Error::CutlassInternal(status));
        }
        Ok(())
    }

    /// Forwarding helper that dispatches to `cublasGemmEx` with the
    /// right `cudaDataType` and `cublasComputeType_t` tags for `T`.
    #[allow(clippy::too_many_arguments)]
    fn gemm_ex(
        &self,
        handle: cublasHandle_t,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const c_void,
        a: *const c_void,
        _m_marker: i32, // shape-redundant, kept for call-site clarity
        lda: i32,
        b: *const c_void,
        ldb: i32,
        beta: *const c_void,
        c: *mut c_void,
        ldc: i32,
    ) -> Result<()> {
        let (data_type, compute_type) = match T::KIND {
            ElementKind::F16 => (CUDA_R_16F, CUBLAS_COMPUTE_32F),
            ElementKind::Bf16 => (CUDA_R_16BF, CUBLAS_COMPUTE_32F),
            ElementKind::F32 => (CUDA_R_32F, CUBLAS_COMPUTE_32F),
            ElementKind::F64 => (CUDA_R_64F, CUBLAS_COMPUTE_64F),
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::FusedLinearCrossEntropyPlan::gemm_ex: unwired dtype",
                ))
            }
        };
        let status = unsafe {
            cublasGemmEx(
                handle, transa, transb, m, n, k,
                alpha, a, data_type, lda, b, data_type, ldb, beta, c, data_type, ldc,
                compute_type, CUBLAS_GEMM_DEFAULT,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }

    /// Count `target[i] != ignore_index` via a small device-side
    /// reduction kernel + D2H of the single i64 result.
    ///
    /// Forces a host-sync on `stream` (because we read the count back
    /// before launching the per-chunk fused step that uses
    /// `1/N` as its `scale_per_row`). This matches Liger's
    /// `.sum().item()` sync point.
    fn count_non_ignore(
        &self,
        ctx: &Context,
        stream: &Stream,
        target: &TensorRef<'_, i64, 1>,
    ) -> Result<usize> {
        let bt = self.desc.bt;
        if bt == 0 {
            return Ok(0);
        }
        let mut count_dev: DeviceBuffer<i64> = DeviceBuffer::zeros(ctx, 1).map_err(|_| {
            Error::InvalidProblem(
                "baracuda-kernels::FusedLinearCrossEntropyPlan: count buffer alloc failed",
            )
        })?;
        let status = unsafe {
            baracuda_kernels_sys::baracuda_kernels_loss_flce_count_non_ignore_run(
                bt,
                self.desc.ignore_index,
                target.data.as_raw().0 as *const c_void,
                count_dev.as_slice_mut().as_raw().0 as *mut c_void,
                stream.as_raw() as *mut c_void,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(status));
        }
        // D2H read forces the count to be available before we use it
        // as `1/N` in the per-chunk kernel launch parameters.
        let mut host = [0i64; 1];
        count_dev.copy_to_host(&mut host).map_err(|_| {
            Error::InvalidProblem(
                "baracuda-kernels::FusedLinearCrossEntropyPlan: count D2H failed",
            )
        })?;
        Ok(host[0] as usize)
    }

    /// Zero out `out` (and grad_input / grad_weight if present) for the
    /// degenerate all-ignored case.
    fn zero_outputs(
        &self,
        stream: &Stream,
        args: &FusedLinearCrossEntropyArgs<'_, T>,
    ) -> Result<()> {
        use baracuda_driver::memory::memset_u8_async;
        let out_bytes = args.out.numel() as usize * core::mem::size_of::<T>();
        if out_bytes > 0 {
            memset_u8_async(args.out.data.as_raw(), 0, out_bytes, stream).map_err(|_| {
                Error::InvalidProblem(
                    "baracuda-kernels::FusedLinearCrossEntropyPlan: zero out failed",
                )
            })?;
        }
        if let Some(ref gi) = args.grad_input {
            let bytes = gi.numel() as usize * core::mem::size_of::<T>();
            if bytes > 0 {
                memset_u8_async(gi.data.as_raw(), 0, bytes, stream).map_err(|_| {
                    Error::InvalidProblem(
                        "baracuda-kernels::FusedLinearCrossEntropyPlan: zero grad_input failed",
                    )
                })?;
            }
        }
        // grad_weight is pre-zeroed by the caller contract; nothing to do.
        let _ = args.grad_weight.as_ref();
        Ok(())
    }
}

impl<T: Element> Drop for FusedLinearCrossEntropyPlan<T> {
    fn drop(&mut self) {
        let h = self.handle.get();
        if !h.is_null() {
            unsafe {
                cublasDestroy_v2(h);
            }
            self.handle.set(core::ptr::null_mut());
        }
    }
}

/// Pick the per-chunk row count.
///
/// Mirrors Liger's heuristic verbatim:
///   `inc_factor = ceildiv(V, H)`
///   `chunk_size = next_pow2(ceildiv(BT, inc_factor))`
/// capped at [`MAX_CHUNK_ROWS`].
fn pick_chunk_size(bt: i32, h: i32, v: i32) -> i32 {
    if bt <= 0 {
        return 1;
    }
    let inc_factor = (v + h - 1) / h;
    let raw = (bt + inc_factor - 1) / inc_factor;
    let pw2 = next_pow2_i32(raw);
    core::cmp::min(pw2, MAX_CHUNK_ROWS).max(1)
}

fn next_pow2_i32(x: i32) -> i32 {
    if x <= 1 {
        return 1;
    }
    let mut v = (x - 1) as u32;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    (v + 1) as i32
}

// =============================================================================
// BACKWARD
// =============================================================================

/// Descriptor for the FLCE backward op.
///
/// `BT`, `H`, `V` match the descriptor used at FW time. Both
/// `grad_input` and `grad_weight` were already populated by the FW
/// plan; backward just multiplies them by `dy_scalar`.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct FusedLinearCrossEntropyBackwardDescriptor {
    /// Number of rows of hidden states.
    pub bt: i32,
    /// Hidden dim.
    pub h: i32,
    /// Vocab.
    pub v: i32,
    /// Element dtype.
    pub element: ElementKind,
}

impl FusedLinearCrossEntropyBackwardDescriptor {
    /// Constructor.
    #[inline]
    pub fn new(bt: i32, h: i32, v: i32, element: ElementKind) -> Self {
        Self { bt, h, v, element }
    }
}

/// Args for [`FusedLinearCrossEntropyBackwardPlan::run`].
///
/// Note: unlike most BW plans, `dy` here is a **host f32 scalar** (not
/// a device tensor). This matches the typical "CE is the last layer"
/// case where the upstream gradient is `1.0`; Liger handles the
/// general case via a kernel-side `.item()` sync, which we avoid by
/// pushing the scalar through the API. The vast majority of callers
/// will pass `1.0` and hit the fast path.
pub struct FusedLinearCrossEntropyBackwardArgs<'a, T: Element> {
    /// Upstream gradient scalar (host-side f32). `1.0` is the common
    /// "CE is the last layer" case — pass that to hit the fast path
    /// (no in-place scale launches).
    pub dy_scalar: f32,
    /// Saved `grad_input` from the FW pass — gets multiplied in place
    /// by `dy_scalar`.
    pub grad_input: Option<TensorMut<'a, T, 2>>,
    /// Saved `grad_weight` from the FW pass — gets multiplied in
    /// place by `dy_scalar`.
    pub grad_weight: Option<TensorMut<'a, T, 2>>,
}

/// Backward plan for FLCE. Just multiplies the saved gradients by
/// the upstream `dy` scalar.
pub struct FusedLinearCrossEntropyBackwardPlan<T: Element> {
    desc: FusedLinearCrossEntropyBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> FusedLinearCrossEntropyBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &FusedLinearCrossEntropyBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::FusedLinearCrossEntropyBackwardPlan: descriptor.element != T",
            ));
        }
        check_supported_dtype::<T>()?;
        if desc.bt < 0 || desc.h < 1 || desc.v < 1 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FusedLinearCrossEntropyBackwardPlan: invalid shape",
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
            op: LossKind::FusedLinearCrossEntropy as u16,
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

    /// Workspace bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Launch.
    ///
    /// Issues two in-place scale launches (one for `grad_input`, one
    /// for `grad_weight`). Fast-path when `dy_scalar == 1.0` — emits
    /// no kernels.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: FusedLinearCrossEntropyBackwardArgs<'_, T>,
    ) -> Result<()> {
        let dy_scalar_f32 = args.dy_scalar;

        // Fast path: dy == 1.0 — no scale needed.
        if dy_scalar_f32 == 1.0 {
            return Ok(());
        }

        // Two in-place scale launches.
        if let Some(ref gi) = args.grad_input {
            let numel = gi.numel();
            let status = unsafe {
                match T::KIND {
                    ElementKind::F32 => baracuda_kernels_sys::baracuda_kernels_loss_flce_inplace_scale_f32_run(
                        numel, dy_scalar_f32, gi.data.as_raw().0 as *mut c_void,
                        stream.as_raw() as *mut c_void),
                    ElementKind::F16 => baracuda_kernels_sys::baracuda_kernels_loss_flce_inplace_scale_f16_run(
                        numel, dy_scalar_f32, gi.data.as_raw().0 as *mut c_void,
                        stream.as_raw() as *mut c_void),
                    ElementKind::Bf16 => baracuda_kernels_sys::baracuda_kernels_loss_flce_inplace_scale_bf16_run(
                        numel, dy_scalar_f32, gi.data.as_raw().0 as *mut c_void,
                        stream.as_raw() as *mut c_void),
                    ElementKind::F64 => baracuda_kernels_sys::baracuda_kernels_loss_flce_inplace_scale_f64_run(
                        numel, dy_scalar_f32, gi.data.as_raw().0 as *mut c_void,
                        stream.as_raw() as *mut c_void),
                    _ => return Err(Error::Unsupported("unwired dtype")),
                }
            };
            if status != 0 {
                return Err(Error::CutlassInternal(status));
            }
        }
        if let Some(ref gw) = args.grad_weight {
            let numel = gw.numel();
            let status = unsafe {
                match T::KIND {
                    ElementKind::F32 => baracuda_kernels_sys::baracuda_kernels_loss_flce_inplace_scale_f32_run(
                        numel, dy_scalar_f32, gw.data.as_raw().0 as *mut c_void,
                        stream.as_raw() as *mut c_void),
                    ElementKind::F16 => baracuda_kernels_sys::baracuda_kernels_loss_flce_inplace_scale_f16_run(
                        numel, dy_scalar_f32, gw.data.as_raw().0 as *mut c_void,
                        stream.as_raw() as *mut c_void),
                    ElementKind::Bf16 => baracuda_kernels_sys::baracuda_kernels_loss_flce_inplace_scale_bf16_run(
                        numel, dy_scalar_f32, gw.data.as_raw().0 as *mut c_void,
                        stream.as_raw() as *mut c_void),
                    ElementKind::F64 => baracuda_kernels_sys::baracuda_kernels_loss_flce_inplace_scale_f64_run(
                        numel, dy_scalar_f32, gw.data.as_raw().0 as *mut c_void,
                        stream.as_raw() as *mut c_void),
                    _ => return Err(Error::Unsupported("unwired dtype")),
                }
            };
            if status != 0 {
                return Err(Error::CutlassInternal(status));
            }
        }

        Ok(())
    }
}

// =============================================================================
// Pure-host unit tests for the chunk_size heuristic.
// =============================================================================

#[cfg(test)]
mod chunk_size_tests {
    use super::*;

    #[test]
    fn llama3_class_picks_2048() {
        // Liger's example: BT=4096*4, V=32000, H=4096 -> inc=8, chunk=2048.
        let cs = pick_chunk_size(4096 * 4, 4096, 32000);
        assert_eq!(cs, 2048);
    }

    #[test]
    fn small_problem_caps_at_bt() {
        // Tiny problem: BT=128, V=1000, H=4096 -> inc=1, raw=128, pw2=128.
        let cs = pick_chunk_size(128, 4096, 1000);
        assert_eq!(cs, 128);
    }

    #[test]
    fn empty_bt() {
        let cs = pick_chunk_size(0, 128, 256);
        assert_eq!(cs, 1);
    }

    #[test]
    fn vocab_128k_llama3() {
        // BT=16K, V=128K, H=4096 -> inc=32, raw=512, pw2=512.
        let cs = pick_chunk_size(16384, 4096, 128 * 1024);
        assert_eq!(cs, 512);
    }
}
