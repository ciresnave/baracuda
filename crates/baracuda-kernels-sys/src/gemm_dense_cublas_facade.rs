//! Phase 74 — dense floating-point GEMM FFI facade (cuBLAS-backed).
//!
//! Closes the Fuel 2026-06-10 ask ("dense FP GEMM family in
//! baracuda-kernels"): the last non-baracuda CUDA surface in Fuel is
//! its own cuBLAS MatMul wrapper (`matmul_via_cublas`), which can't
//! retire until baracuda exposes an equivalent flat C entry point.
//! These symbols are that entry point. The matching Rust plan is
//! `baracuda_kernels::DenseGemmPlan`.
//!
//! ## Coverage
//!
//! `baracuda_kernels_gemm_dense_{f32, f64, f16, bf16}_{run,
//! can_implement, workspace_size}` — 12 symbols. One symbol family
//! handles both the single GEMM and the strided-batched case: `batch`
//! is a first-class dimension (`batch == 1` routes to `cublasGemmEx`,
//! `batch > 1` to `cublasGemmStridedBatchedEx`).
//!
//! ## Problem semantics (row-major)
//!
//! `D[g] = α · A[g] · B[g] + β · D[g]` for each batch slot
//! `g ∈ [0, batch)`, with `A: [M, K]`, `B: [K, N]`, `D: [M, N]`.
//! There is no separate `C` operand: `β ≠ 0` accumulates into the
//! existing contents of `D` (read-modify-write, matching cuBLAS's
//! in-place `C` contract).
//!
//! ## Layout (`layout` parameter)
//!
//! | value | name | A storage          | B storage          | ld minimums              |
//! |-------|------|--------------------|--------------------|--------------------------|
//! | `0`   | RRR  | row-major `[M,K]`  | row-major `[K,N]`  | `lda≥K, ldb≥N, ldd≥N`    |
//! | `1`   | RCR  | row-major `[M,K]`  | col-major `[K,N]`  | `lda≥K, ldb≥K, ldd≥N`    |
//! | `2`   | CRR  | col-major `[M,K]`  | row-major `[K,N]`  | `lda≥M, ldb≥N, ldd≥N`    |
//!
//! `D` is always row-major `[M, N]`. Leading dimensions are in
//! **elements** and may exceed their minimum — this is the contract
//! that lets callers pass row-slice / column-slice views of larger
//! tensors without materializing a contiguous copy (the
//! BERT / SD-CLIP / Qwen2-MoE non-contiguous matmul cases from Fuel's
//! audit). RCR is the "transposed weight" (`x · Wᵀ`) case; CRR is the
//! "transposed activation" (`xᵀ · dy`, grad-weight) case. All three
//! map to a `transa`/`transb` pair on the cuBLAS column-major API —
//! no transpose is materialized.
//!
//! ## Batch contract
//!
//! `stride_a` / `stride_b` / `stride_d` are in **elements**, applied as
//! `ptr + g * stride`. `stride_a` / `stride_b` may be `0` to broadcast
//! one matrix across all slots. `stride_d` must be non-zero when
//! `batch > 1` (overlapping outputs race). Strides are ignored when
//! `batch == 1`.
//!
//! ## Dtype / accumulation
//!
//! | symbol | storage | compute (accumulator) | α/β type |
//! |--------|---------|-----------------------|----------|
//! | `_f32`  | `CUDA_R_32F`  | `CUBLAS_COMPUTE_32F` (IEEE binary32) | `f32` |
//! | `_f64`  | `CUDA_R_64F`  | `CUBLAS_COMPUTE_64F` | `f64` |
//! | `_f16`  | `CUDA_R_16F`  | `CUBLAS_COMPUTE_32F` | `f32` |
//! | `_bf16` | `CUDA_R_16BF` | `CUBLAS_COMPUTE_32F` | `f32` |
//!
//! The f32 path uses cuBLAS's **default math mode** — full IEEE 754
//! binary32 multiply-add, NOT TF32 (this differs from the CUTLASS
//! `GemmPlan<f32>` SKU, which routes through TF32 tensor cores).
//! Caveat: the process-wide `NVIDIA_TF32_OVERRIDE=1` environment
//! variable forces TF32 inside cuBLAS's default math mode — the
//! facade does not (and cannot cheaply) defend against it; don't set
//! it if you rely on the binary32 guarantee. f16 / bf16 accumulate in
//! f32, matching the reduce family's convention. Run-to-run
//! determinism follows cuBLAS's guarantee, INCLUDING its condition:
//! bitwise-reproducible for identical (shape, dtype, arch, SM count,
//! toolkit version) launches **only while a single CUDA stream is
//! active**. Concurrent GEMMs on multiple streams may select
//! different internal implementations run-to-run; the pooled-handle
//! design shares cuBLAS's default workspace pool across streams, so
//! this facade cannot promise multi-stream reproducibility.
//!
//! ## Handle lifecycle — pooled, NOT transient
//!
//! Unlike the cuSOLVER / cuFFT / cuRAND facades (transient handle per
//! call), this facade keeps a small lock-free **pool** of cuBLAS
//! handles keyed by the calling thread's current CUDA context. GEMM
//! sits on the model hot path (every MatMul node, every decode step);
//! a per-call `cublasCreate`/`cublasDestroy` pair costs hundreds of
//! microseconds AND hides a device-synchronizing `cudaFree` in the
//! destroy — unacceptable per-launch overhead. The pool:
//!
//! - holds up to `POOL_SLOTS` (8) `(context, handle)` pairs,
//!   process-wide;
//! - take/put are CAS-based (no locks, `no_std`-compatible);
//! - a taken handle is exclusively owned for the duration of one call
//!   (cuBLAS handles are not concurrently shareable — the stream
//!   binding is per-handle state);
//! - the stream is re-bound via `cublasSetStream_v2` on every call;
//! - if all slots are taken or foreign-context, the call falls back to
//!   a transient create/destroy (correct, just slower);
//! - pooled handles live until process exit (bounded leak of ≤
//!   `POOL_SLOTS` handles — same trade-off as the Phase 30
//!   thread-local cache in `baracuda-cutlass`).
//!
//! No-current-context policy: when `cuCtxGetCurrent` reports no
//! context at call entry (a fresh thread's first CUDA touch), the
//! context key is re-queried AFTER handle creation — cuBLAS's runtime
//! initialization binds the primary context, so the refreshed key is
//! the real context address and the handle pools correctly. If even
//! the re-query reports no context, the handle is destroyed instead
//! of pooled (transient call) — handles are never parked under the
//! no-context sentinel, which would strand slots and risk
//! cross-context revival.
//!
//! Hazard (shared with Phase 30): if a CUDA context is destroyed and a
//! NEW context is later allocated at the same address, a stale pooled
//! handle could be revived against the new context. Don't destroy
//! contexts mid-process while continuing to call these symbols from
//! other contexts at the same address; tear-down-then-exit is fine.
//!
//! ## Stream capture
//!
//! No capture-mode special-casing at this layer: behavior under
//! `cudaStreamBeginCapture` follows cuBLAS's own rules (capturable on
//! CUDA 12 provided the handle's workspace isn't shared with another
//! concurrently-capturing stream). Callers that need a
//! capture-guaranteed path should drive the CUTLASS `GemmPlan`, which
//! auto-falls-back under capture.
//!
//! ## Workspace
//!
//! `workspace` / `workspace_bytes` on `*_run` are **reserved and
//! ignored** (present for binding-table shape uniformity); cuBLAS
//! manages its own per-handle workspace internally. `*_workspace_size`
//! always returns `0`.
//!
//! ## Status codes
//!
//! Same convention as the rest of `baracuda-kernels-sys`: `0` success,
//! `2` invalid problem (bad extents / layout tag / leading dims /
//! strides / null pointers), `5` internal cuBLAS error. Empty problems
//! (`m == 0 || n == 0 || batch == 0`) return `0` without launching.
//! `k == 0` IS launched (BLAS semantics: `D = β · D`).

#![allow(clippy::too_many_arguments)]

use core::ffi::c_void;
use core::ptr;
use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use super::{
    cublasCreate_v2, cublasDestroy_v2, cublasGemmEx, cublasGemmStridedBatchedEx, cublasHandle_t,
    cublasSetStream_v2, CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT, CUBLAS_OP_N,
    CUBLAS_OP_T, CUDA_R_16BF, CUDA_R_16F, CUDA_R_32F, CUDA_R_64F,
};

unsafe extern "C" {
    /// `cuCtxGetCurrent` (CUDA **driver** API; `libcuda` is already on
    /// this crate's link line for the bespoke-kernel loaders). Used
    /// only as the handle-pool cache key — never to mutate context
    /// state.
    fn cuCtxGetCurrent(ctx: *mut *mut c_void) -> i32;
}

// =============================================================================
// Status codes
// =============================================================================

const OK: i32 = 0;
const INVALID: i32 = 2;
const INTERNAL: i32 = 5;

// =============================================================================
// Layout tags (must stay in sync with `baracuda_kernels::DenseGemmLayout`)
// =============================================================================

const LAYOUT_RRR: i32 = 0;
const LAYOUT_RCR: i32 = 1;
const LAYOUT_CRR: i32 = 2;

// =============================================================================
// Handle pool
// =============================================================================

/// Pool capacity. 8 covers (threads actively launching GEMMs) ×
/// (live contexts) for every workload we've seen; overflow degrades
/// to transient create/destroy, never to an error.
const POOL_SLOTS: usize = 8;

/// One pool slot. `ctx == 0` means "unclaimed"; once claimed for a
/// context key the claim is permanent (slots are never re-keyed —
/// keys are context addresses, and a freed-then-reused context address
/// keying into an old slot is exactly the documented stale-handle
/// hazard, not something re-keying could fix).
struct Slot {
    ctx: AtomicUsize,
    handle: AtomicPtr<c_void>,
}

#[allow(clippy::declare_interior_mutable_const)]
const EMPTY_SLOT: Slot = Slot {
    ctx: AtomicUsize::new(0),
    handle: AtomicPtr::new(ptr::null_mut()),
};

static POOL: [Slot; POOL_SLOTS] = [EMPTY_SLOT; POOL_SLOTS];

/// Cache key for the calling thread's current CUDA context. Null /
/// no-context maps to `usize::MAX` so it can't collide with the
/// `0 == unclaimed` slot sentinel.
fn current_ctx_key() -> usize {
    let mut ctx: *mut c_void = ptr::null_mut();
    let st = unsafe { cuCtxGetCurrent(&mut ctx as *mut _) };
    if st != 0 || ctx.is_null() {
        usize::MAX
    } else {
        ctx as usize
    }
}

/// Take a pooled handle for `key`, or create a fresh one.
///
/// Creation retries up to 5× with a spin backoff: `cublasCreate_v2`
/// races on a shared driver-init resource when many processes start
/// concurrently (the `cargo test --workspace` harness) and returns a
/// transient ALLOC_FAILED / NOT_INITIALIZED — observed empirically in
/// Phase 30 and cleared reliably by a short backoff. `no_std` has no
/// sleep, so the backoff is a bounded spin.
unsafe fn take_handle(key: usize) -> Result<cublasHandle_t, i32> {
    for slot in POOL.iter() {
        if slot.ctx.load(Ordering::Acquire) == key {
            let h = slot.handle.swap(ptr::null_mut(), Ordering::AcqRel);
            if !h.is_null() {
                return Ok(h);
            }
        }
    }
    let mut handle: cublasHandle_t = ptr::null_mut();
    for attempt in 0..5u64 {
        // Backoff BETWEEN attempts only — a permanently-failing host
        // (no driver / no device) shouldn't pay a trailing spin after
        // the final attempt.
        if attempt > 0 {
            let spins = 4_000_000 * attempt;
            for _ in 0..spins {
                core::hint::spin_loop();
            }
        }
        let st = unsafe { cublasCreate_v2(&mut handle as *mut _) };
        if st == 0 && !handle.is_null() {
            return Ok(handle);
        }
        handle = ptr::null_mut();
    }
    Err(INTERNAL)
}

/// Return a handle to the pool, or destroy it if no slot is available.
///
/// Never parks under the no-context sentinel: a `usize::MAX`-keyed
/// slot could only be hit again by some OTHER thread that also has no
/// current context — whose eventual context may be a different one,
/// reviving a handle across contexts. Callers re-query the key after
/// handle creation (see `gemm_dense_run_impl`); if it is still
/// `usize::MAX` the handle is treated as transient and destroyed.
unsafe fn put_handle(key: usize, h: cublasHandle_t) {
    if key == usize::MAX {
        unsafe {
            let _ = cublasDestroy_v2(h);
        }
        return;
    }
    // Pass 1: a slot already claimed for this context with no parked
    // handle.
    for slot in POOL.iter() {
        if slot.ctx.load(Ordering::Acquire) == key
            && slot
                .handle
                .compare_exchange(ptr::null_mut(), h, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        {
            return;
        }
    }
    // Pass 2: claim a fresh slot. The handle CAS (not a plain store)
    // closes the race where another thread parks into our
    // freshly-claimed slot between the two operations.
    for slot in POOL.iter() {
        if slot
            .ctx
            .compare_exchange(0, key, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            if slot
                .handle
                .compare_exchange(ptr::null_mut(), h, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return;
            }
            break;
        }
    }
    // Pool exhausted (or our claimed slot was raced) — this call was
    // effectively transient. Destroy is the slow path (device sync
    // inside cuBLAS's cudaFree) but only happens under > POOL_SLOTS
    // concurrent callers per context.
    unsafe {
        let _ = cublasDestroy_v2(h);
    }
}

// =============================================================================
// Validation
// =============================================================================

/// Host-side shape/layout validation shared by `*_run` and
/// `*_can_implement`. Pointer nullness is checked separately in
/// `*_run` (after the empty-problem early-out).
fn validate(
    m: i32,
    n: i32,
    k: i32,
    batch: i32,
    layout: i32,
    lda: i64,
    ldb: i64,
    ldd: i64,
    stride_d: i64,
) -> i32 {
    if m < 0 || n < 0 || k < 0 || batch < 0 {
        return INVALID;
    }
    let (min_lda, min_ldb) = match layout {
        LAYOUT_RRR => (k as i64, n as i64),
        LAYOUT_RCR => (k as i64, k as i64),
        LAYOUT_CRR => (m as i64, n as i64),
        _ => return INVALID,
    };
    if lda < min_lda.max(1) || ldb < min_ldb.max(1) || ldd < (n as i64).max(1) {
        return INVALID;
    }
    // cuBLAS leading-dim parameters are i32.
    let i32_max = i32::MAX as i64;
    if lda > i32_max || ldb > i32_max || ldd > i32_max {
        return INVALID;
    }
    if batch > 1 && stride_d == 0 {
        return INVALID;
    }
    OK
}

// =============================================================================
// Shared launch path
// =============================================================================

/// Row-major problem → cuBLAS column-major call. Mapping (same trick
/// as `baracuda-cutlass`'s Phase 30 cuBLAS backend and Fuel's
/// `matmul_via_cublas`): `D_rm = A · B` ⇔ `D_cmᵀ = Bᵀ_cm · Aᵀ_cm`,
/// where the column-major view of row-major storage IS the transpose.
/// So cuBLAS sees `m' = N, n' = M, k' = K`, first operand = our B,
/// second operand = our A:
///
/// - RRR: `transa = N` (B storage cm-view = Bᵀ, `[N,K]`, ld = ldb),
///        `transb = N` (A storage cm-view = Aᵀ, `[K,M]`, ld = lda).
/// - RCR: B is stored col-major `[K,N]` so its cm-view is B itself —
///        `transa = T` recovers Bᵀ.
/// - CRR: A is stored col-major `[M,K]` so its cm-view is A itself —
///        `transb = T` recovers Aᵀ.
unsafe fn gemm_dense_run_impl(
    m: i32,
    n: i32,
    k: i32,
    batch: i32,
    layout: i32,
    alpha: *const c_void,
    beta: *const c_void,
    a: *const c_void,
    lda: i64,
    stride_a: i64,
    b: *const c_void,
    ldb: i64,
    stride_b: i64,
    d: *mut c_void,
    ldd: i64,
    stride_d: i64,
    data_type: i32,
    compute_type: i32,
    stream: *mut c_void,
) -> i32 {
    let st = validate(m, n, k, batch, layout, lda, ldb, ldd, stride_d);
    if st != OK {
        return st;
    }
    if m == 0 || n == 0 || batch == 0 {
        return OK;
    }
    if a.is_null() || b.is_null() || d.is_null() {
        return INVALID;
    }

    let (transa, transb) = match layout {
        LAYOUT_RRR => (CUBLAS_OP_N, CUBLAS_OP_N),
        LAYOUT_RCR => (CUBLAS_OP_T, CUBLAS_OP_N),
        LAYOUT_CRR => (CUBLAS_OP_N, CUBLAS_OP_T),
        // Unreachable: `validate` rejected every other tag.
        _ => return INVALID,
    };

    let key = current_ctx_key();
    let handle = match unsafe { take_handle(key) } {
        Ok(h) => h,
        Err(e) => return e,
    };
    // Fresh thread, first CUDA touch: there was no current context at
    // entry, but cuBLAS's runtime init inside `cublasCreate` binds the
    // primary context — re-query so the handle pools under its real
    // context key instead of the sentinel (see `put_handle`).
    let key = if key == usize::MAX { current_ctx_key() } else { key };
    let st = unsafe { cublasSetStream_v2(handle, stream) };
    if st != 0 {
        // Don't pool a handle whose state we couldn't establish.
        unsafe {
            let _ = cublasDestroy_v2(handle);
        }
        return INTERNAL;
    }

    // Operand swap: cuBLAS A' = our B, B' = our A, C' = our D.
    let status = if batch == 1 {
        unsafe {
            cublasGemmEx(
                handle,
                transa,
                transb,
                n,
                m,
                k,
                alpha,
                b,
                data_type,
                ldb as i32,
                a,
                data_type,
                lda as i32,
                beta,
                d,
                data_type,
                ldd as i32,
                compute_type,
                CUBLAS_GEMM_DEFAULT,
            )
        }
    } else {
        unsafe {
            cublasGemmStridedBatchedEx(
                handle,
                transa,
                transb,
                n,
                m,
                k,
                alpha,
                b,
                data_type,
                ldb as i32,
                stride_b,
                a,
                data_type,
                lda as i32,
                stride_a,
                beta,
                d,
                data_type,
                ldd as i32,
                stride_d,
                batch,
                compute_type,
                CUBLAS_GEMM_DEFAULT,
            )
        }
    };
    unsafe { put_handle(key, handle) };
    if status != 0 {
        INTERNAL
    } else {
        OK
    }
}

// =============================================================================
// Per-dtype symbol families
// =============================================================================

macro_rules! gemm_dense_family {
    (
        $run:ident,
        $can:ident,
        $ws:ident,
        $scalar:ty,
        $data_type:expr,
        $compute_type:expr,
        $dtype_doc:literal,
        $acc_doc:literal
    ) => {
        #[doc = concat!(
            "Dense ", $dtype_doc, " GEMM (cuBLAS-backed): ",
            "`D[g] = α · A[g] · B[g] + β · D[g]` for `g ∈ [0, batch)`, ",
            "accumulating in ", $acc_doc, ". Row-major problem; see the ",
            "module docs for the `layout` tag (0 = RRR, 1 = RCR, ",
            "2 = CRR), leading-dim minimums, and the batch-stride ",
            "contract (element strides; `stride_a`/`stride_b` may be 0 ",
            "to broadcast; strides ignored at `batch == 1`).",
        )]
        ///
        /// `workspace` / `workspace_bytes` are reserved and ignored
        /// (cuBLAS manages its own per-handle workspace).
        ///
        /// # Safety
        /// `a` / `b` / `d` are device pointers sized per the layout
        /// table (incl. `(batch-1) * stride` reach); `stream` is a
        /// valid CUDA stream of the calling thread's current context
        /// (or null for the legacy default stream).
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $run(
            m: i32,
            n: i32,
            k: i32,
            batch: i32,
            layout: i32,
            alpha: $scalar,
            beta: $scalar,
            a: *const c_void,
            lda: i64,
            stride_a: i64,
            b: *const c_void,
            ldb: i64,
            stride_b: i64,
            d: *mut c_void,
            ldd: i64,
            stride_d: i64,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            let _ = (workspace, workspace_bytes);
            let alpha_v: $scalar = alpha;
            let beta_v: $scalar = beta;
            unsafe {
                gemm_dense_run_impl(
                    m,
                    n,
                    k,
                    batch,
                    layout,
                    &alpha_v as *const $scalar as *const c_void,
                    &beta_v as *const $scalar as *const c_void,
                    a,
                    lda,
                    stride_a,
                    b,
                    ldb,
                    stride_b,
                    d,
                    ldd,
                    stride_d,
                    $data_type,
                    $compute_type,
                    stream,
                )
            }
        }

        #[doc = concat!(
            "Host-side validity check for [`", stringify!($run), "`]. ",
            "Validates extents, the `layout` tag, leading-dim minimums, ",
            "i32-fit of leading dims, and `stride_d != 0` at ",
            "`batch > 1`. `stride_a` / `stride_b` are accepted ",
            "unconditionally (any value, including 0-broadcast).",
        )]
        #[unsafe(no_mangle)]
        pub extern "C" fn $can(
            m: i32,
            n: i32,
            k: i32,
            batch: i32,
            layout: i32,
            lda: i64,
            ldb: i64,
            ldd: i64,
            _stride_a: i64,
            _stride_b: i64,
            stride_d: i64,
        ) -> i32 {
            validate(m, n, k, batch, layout, lda, ldb, ldd, stride_d)
        }

        #[doc = concat!(
            "Workspace query for [`", stringify!($run), "`]. Always ",
            "`0` — cuBLAS allocates its workspace internally per handle.",
        )]
        #[unsafe(no_mangle)]
        pub extern "C" fn $ws(_m: i32, _n: i32, _k: i32, _batch: i32, _layout: i32) -> usize {
            0
        }
    };
}

gemm_dense_family!(
    baracuda_kernels_gemm_dense_f32_run,
    baracuda_kernels_gemm_dense_f32_can_implement,
    baracuda_kernels_gemm_dense_f32_workspace_size,
    f32,
    CUDA_R_32F,
    CUBLAS_COMPUTE_32F,
    "f32",
    "IEEE binary32 (default math mode — NOT TF32)"
);

gemm_dense_family!(
    baracuda_kernels_gemm_dense_f64_run,
    baracuda_kernels_gemm_dense_f64_can_implement,
    baracuda_kernels_gemm_dense_f64_workspace_size,
    f64,
    CUDA_R_64F,
    CUBLAS_COMPUTE_64F,
    "f64",
    "f64"
);

gemm_dense_family!(
    baracuda_kernels_gemm_dense_f16_run,
    baracuda_kernels_gemm_dense_f16_can_implement,
    baracuda_kernels_gemm_dense_f16_workspace_size,
    f32,
    CUDA_R_16F,
    CUBLAS_COMPUTE_32F,
    "f16",
    "f32"
);

gemm_dense_family!(
    baracuda_kernels_gemm_dense_bf16_run,
    baracuda_kernels_gemm_dense_bf16_can_implement,
    baracuda_kernels_gemm_dense_bf16_workspace_size,
    f32,
    CUDA_R_16BF,
    CUBLAS_COMPUTE_32F,
    "bf16",
    "f32"
);
