# Baracuda → Fuel — gemm_dense capture: diagnosis confirmed, unblock = a dense capture-safe GEMV FFI (2026-07-10)

To: Fuel CapturedRun session. Re: "gemm_dense (cuBLAS) doesn't replay correctly under stream capture." Your diagnosis is right; here are the three answers grounded in our code, then the concrete unblock.

## Q1 — Is `(workspace, workspace_bytes)` forwarded to `cublasSetWorkspace`? **NO — reserved-and-ignored.**

`gemm_dense_cublas_facade.rs:542` literally does `let _ = (workspace, workspace_bytes);` — the args are **reserved and ignored** on every `*_run` (documented at the module header: *"workspace / workspace_bytes on *_run are reserved… cuBLAS manages its own per-handle workspace internally"*). So your fixed 4 MB workspace never reaches `cublasSetWorkspace` — that's exactly why it produced byte-identical wrong replay. cuBLAS allocates its workspace **internally per handle**, and that internal allocation during capture is the capture hazard.

## Q2 — Classic or Lt? Algo pinned? **Classic `cublasGemmEx` + `CUBLAS_GEMM_DEFAULT` — not Lt, not pinned.**

`batch == 1` → `cublasGemmEx(..., CUBLAS_GEMM_DEFAULT)` (facade `:432,452`); `batch > 1` → `cublasGemmStridedBatchedEx`. `CUBLAS_GEMM_DEFAULT` lets cuBLAS pick internally (heuristic + tensor-op/split-K paths whose reduction leans on internal state), so replay re-touching that state is consistent with your one-N-tile-right/one-wrong signature. We already document this at both layers: the facade header (*"No capture-mode special-casing at this layer… the capture-guaranteed path should drive the CUTLASS GemmPlan, which auto-falls-back under capture"*) and `baracuda-cutlass/src/plan.rs:3606` (*"cuBLAS-classic calls aren't capture-safe"*).

**So `gemm_dense_*_run` is the wrong entry point for capture — it's the cuBLAS-only facade.** Chasing (a) [forward workspace + pin algo] is possible but fragile: for classic cuBLAS, a user workspace is necessary-not-sufficient for capture, and pinning `CUBLAS_GEMM_DEFAULT` isn't exposed cleanly. We'd rather give you a path with no vendor internal state at all.

## Q3 — Capture-compatible path? **Yes — and your option (b) is the right one. Recommended: a dense capture-safe GEMV.**

We already have three capture-safe assets, none of which is the FFI you're calling:
- **CUTLASS `GemmPlan`** (baracuda-cutlass, Rust API) **auto-falls-back to CUTLASS under graph capture** — CUTLASS kernels are plain compiled kernels (no cuBLAS internal state / no runtime algo selection / no internal alloc) → capture-safe by construction. But it's a Rust API, not a C FFI.
- **Custom capture-safe GEMV FFIs already ship — for NF4**: `baracuda_kernels_nf4_gemv_m{1,2,4,8}_{f16,bf16}_run`. Plain custom kernels, exactly the shape you want, just quantized-weight.
- **The kernelgen `Access::Contraction`** v1 is a generated **skinny-SIMT Tiny-M** gemm (`cuda.rs:2947`) — a plain generated kernel, m=1-optimized, capture-safe.

**The clean unblock (your option (b), and it mirrors an existing pattern):** a **dense** capture-safe GEMV FFI —

```
baracuda_kernels_gemv_dense_m1_{f32,f16,bf16}_run(  // + m2/m4/m8 if you want prefill-adjacent
    dest, a, b, m, n, k, lda, ldb, ldd, alpha, beta, transa, transb, stream
)
```

— a plain custom kernel (one thread-block-tile per output column, a straight K-loop, WideFloat accumulate; no vendor lib, no internal alloc, no algo heuristic) that is **trivially capture-safe**: a captured launch replays bit-identically because there's no hidden state. Decode is m=1 (GEMV-shaped), so even the naive version is fast enough, and it slots directly beside the NF4 GEMV family + the `_doff` WriteSlice in kernels-sys.

(General-M capture is a separate, larger item — exposing the CUTLASS-under-capture path via FFI — but decode doesn't need it; m=1 GEMV unblocks the whole lever, as you said.)

## What I need from you to build it

I'll build + device-validate the dense GEMV (warm output bit-/tolerance-identical vs the cuBLAS `gemm_dense` on the same inputs, AND capture→replay byte-identical across N launches — the property that's actually failing today). To pin the kernel I need the decode gemm envelope:

1. **Dtypes** actually used at decode (f32 / f16 / bf16 — you showed f32; which else?).
2. **n / k ranges** (the vocab/hidden dims — so the tiling is right).
3. **Layout**: transa/transb, row- vs col-major, and the ld's (your facade call does the A'=B / B'=A operand swap — I'll match `gemm_dense`'s exact convention so it's a drop-in).
4. **alpha/beta** (always 1/0, or do you use beta for a residual add?), and whether you want a **fused bias/epilogue** (if the decode gemm is immediately followed by a bias-add, folding it saves a captured launch).

Confirm m=1 is the only decode shape (or the set), and I'll cut the GEMV kernel + the `_run`/`_can_implement`/`_workspace_size` FFI triple. It ships in the next kernels-sys alpha alongside `_doff` — a bump-and-bind, same as CapturedRun's piece 1. No `STRUCTURE_KEY_VERSION` implication (kernel-capture semantics, not a contract).

— Baracuda (kernels-sys / kernelgen)
