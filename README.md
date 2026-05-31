# baracuda

![A great barracuda — the project's namesake, minus one letter.](https://raw.githubusercontent.com/ciresnave/baracuda/refs/heads/main/assets/barracuda.png)

> **About the name.** Yes, we know — it's spelled **barracuda** (two Rs). That
> name was taken on crates.io, so we dropped one R and kept swimming.

A unified Rust ML-op facade over the NVIDIA CUDA ecosystem.

![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)
![Status](https://img.shields.io/badge/status-alpha.61-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900)
![Tests](https://img.shields.io/badge/regression-2152%2F0-success)

## What baracuda is

baracuda is a Rust workspace that exposes every primitive an ML framework
expects — the union of PyTorch (`torch.*` + `nn.functional`) and JAX
(`jax.lax.*` + `jax.numpy.*`) — through a single `Plan`-based crate surface
called [`baracuda-kernels`]. Internally each plan dispatches to:

1. The appropriate NVIDIA-library wrapper crate (cuBLAS, cuDNN, cuFFT,
   cuSOLVER, cuRAND, cuSPARSE, cuTENSOR, NPP, CV-CUDA, CUTLASS) when one
   already covers the op well, or
2. A bespoke hand-rolled `.cu` kernel shipped in [`baracuda-kernels-sys`]
   when no NVIDIA library covers the op (or covers it poorly at the shapes
   that matter for modern transformer / vision / GNN workloads).

Callers import **one** crate (`baracuda-kernels`) and reach for **one** API
style. The dispatch decision — which is observable through
`Plan::sku()` for telemetry — is otherwise invisible. Switching from a
CUTLASS-backed SKU to a bespoke-backed SKU is a layout flag, not an import
change.

baracuda is for downstream Rust ML / inference / training frameworks that
need access to the full CUDA stack without re-vendoring it themselves. The
workspace also ships idiomatic stand-alone wrappers for every CUDA library
under `crates/baracuda-<lib>` if you want to skip the kernel facade and
talk to one library directly.

## Status

**In active development — alpha.61.** **2184+ GPU tests passing,
zero failures** across the 6 critical test crates on an RTX 4070
(sm_89) — Phase 60 lifted FA2 FW to the full Candle-fork-extended
9-head_dim set ({32, 64, 96, 128, 160, 192, 224, 256, 512}) via
12 new vendored `.cu` files + 32 FW smoke tests. Phase 61
(alpha.61, Fuel-ask) completes the 4-dtype matrix on the in-place
affine helper (`baracuda_kernels_affine_inplace_{bf16,f16}_run`)
and documents same-pointer aliasing safety as a stable public
contract on the unary / binary / ternary contig elementwise
trailblazers — unblocks Fuel's planned in-place op fanout (16+
families) with zero new baracuda symbols per family.
Phase 59a + 59b had added the full FA2 v2.8.3 surface (FW + BW
+ varlen across head_dims 32-256, GQA, ALiBi, sliding window,
softcap) plus 48 new smoke tests, closing Fuel's FA2-retirement
requirements. Phase 59c (consolidation pass, alpha.59) fixed a
pre-existing parallel-test race in the bespoke flash kernel's
SMEM-carveout call surfaced by Phase 59a's 5-head_dim fanout, plus
updated `flash_sdpa_backward_smoke` to force the bespoke backend on
f16/bf16 (Phase 59b made FA2 the new default BW backend, breaking
source-compat for the existing bespoke BW smoke tests).
Phase 42-44
add three opt-in backends (FA2, mHC.cu, ozIMMU); none are on the
default build path. Phase 44b internalized the ozIMMU sources
(clean-fork; cutf submodule retired; Linux + Windows both build).
Phase 49 adds the `baracuda-optim` sibling crate (Adam / LAMB / SGD
via vendored Apex `multi_tensor_apply`) — gated behind the `optim`
feature so inference-only consumers don't pay the FFI surface cost.
Phase 55 adds the `baracuda-transformer-engine` sibling crate
(NVIDIA TransformerEngine FP8 cast + delayed-scaling recipe,
Apache-2.0) — gated behind the `tensor_engine` feature. On Ada
(sm_89) the FP8 wins are bandwidth-saving only (KV cache, weights);
the recipe machinery is forward-compatible with Hopper / Blackwell
where the MMA throughput win also materializes.

Phase coverage (see [`ARCHITECTURE.md`](ARCHITECTURE.md) for the phase
matrix):

| Phase | Scope | Status |
| --- | --- | --- |
| 59a | FA2 FW expansion (alpha.59) — full upstream feature parity (head_dim fanout {32,64,96,192,256}; GQA; ALiBi; sliding window; softcap): vendored 20 new `.cu` files from Dao-AILab FA2 v2.8.3 (head_dims 32/64/96/192/256 × {fp16, bf16} × {causal, non-causal}) bringing the FW vendor coverage to the full upstream set of {32, 64, 96, 128, 192, 256}. Upstream FA2 v2.8.3 does NOT ship head_dims 160/224/512 — those are permanently out-of-scope (no source). Launcher (`kernels/attention/fa2_launcher.cu`) rewritten to dispatch all 6 supported head_dims via runtime switch. NEW `..._run_v2` + `..._can_implement_v2` FFI entry points (+4 symbols) carrying ALiBi slopes + per-head-or-per-batch layout selector + sliding window left/right bounds + Gemma-2-style softcap. v1 entry points preserved for backwards-compat. GQA-divisible head counts (`num_heads % num_heads_k == 0`) now accepted on the FA2 path. `FlashSdpaDescriptor` is now `#[non_exhaustive]` with `::new(...)` + chainable `with_window_size_left`/`with_window_size_right`/`with_softcap` builders (Phase 32 convention). `FlashSdpaArgs` gained `alibi_slopes: Option<TensorRef<f32, 2>>`. Bespoke backend rejects sliding-window/softcap/ALiBi at select-time with clear errors. ~33 descriptor + ~30 args callsites migrated to the builder pattern. 4 new smoke test files (26 new test functions): `fa2_hdim_fanout_smoke` (20), `fa2_gqa_smoke` (1), `fa2_alibi_smoke` (3), `fa2_sliding_window_smoke` (3), `fa2_softcap_smoke` (4). Out of scope (Phase 59b territory): BW path, varlen, split-KV. | done |
| 59c | Bespoke flash SMEM-carveout race fix + flash_sdpa_backward smoke test routing fix (alpha.59 consolidation pass): added `std::mutex`-serialized helper `set_dynamic_smem_serialized` around all `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` calls in `baracuda_flash_sdpa.cuh` + `baracuda_flash_sdpa_sm89.cuh` (5 call sites total: FW + BW dQ + BW dKdV + sm_89 FW + sm_89 strided FW). Pre-existing flake (root cause: Phase 6 / Milestone 6.6 host wrapper) that surfaced as `CutlassInternal(1001)` (= `cudaErrorMissingConfiguration`) at ~33% rate on Phase 59a's 20-test hdim fanout, specifically for d_k=96 + fp16 (smem ~50 KiB, just past the 48 KiB cudaFuncSetAttribute trigger). Confirmed fix via 3 stress runs after fix: 60/60 tests pass. Also fixed `flash_sdpa_backward_smoke`'s f16/bf16 paths to explicitly request `BackendKind::Bespoke` — Phase 59b made FA2 the default BW backend for f16/bf16 (more permissive heuristic), which broke source-compat for the existing bespoke BW smoke tests (they fed `lse: f16` not `lse_f32`). | done |
| 59b | FA2 BW + varlen (alpha.59; closes Fuel's FA2-retirement requirements): vendored 24 new BW `.cu` files (`flash_bwd_hdim{32,64,96,128,192,256}_{fp16,bf16}_{,causal}_sm80.cu` — full FA2 v2.8.3 BW set, mirrors 59a FW vendor 1:1) plus 3 new BW headers (`flash_bwd_kernel.h`, `flash_bwd_launch_template.h`, `flash_bwd_preprocess_kernel.h`). **Key finding**: varlen does NOT have a separate .cu file family upstream — FA2 v2.8.3 dispatches varlen via a runtime `cu_seqlens_q != nullptr` check inside the existing FW/BW launch templates, so the same per-(headdim, dtype, causal) instantiations serve dense and varlen callers. NEW `kernels/attention/fa2_backward_launcher.cu` (BW dispatch, supports dense + varlen via two `fill_*_params` helpers) + `fa2_varlen_launcher.cu` (varlen FW). +12 new FFI symbols (BW dense ×2 + can_implement ×2 + workspace_size; varlen FW ×2 + can_implement ×2 + lse_size; varlen BW ×2 + can_implement ×2 + workspace_size). API: `FlashSdpaBackwardDescriptor` is now `#[non_exhaustive]` with `::new(...)` + sliding-window/softcap builders. `FlashSdpaBackwardArgs` gained `lse_f32: Option<TensorRef<f32, 3>>` (FA2 stores LSE in f32 regardless of T) + `alibi_slopes`. `FlashSdpaBackwardPlan` extended with `BackendChoice::FlashAttentionV2` arm (additive — bespoke path source-compat preserved). NEW `FlashSdpaVarlenPlan` / `FlashSdpaVarlenBackwardPlan` plan families with packed-batch `[total_q, H, D]` layout + `cu_seqlens_q`/`cu_seqlens_k` index tensors + f32 LSE `[H, total_q + 128*B]`. BW workspace = `dq_accum + dsoftmax_d` (sizes via `..._backward_workspace_size`); launcher zero-fills via `cudaMemsetAsync`. Determinism: FA2 BW uses atomicAdd into dq_accum, so NOT bit-stable run-to-run (precision SKU tags this honestly). 2 new smoke test files: `fa2_backward_smoke.rs` (12 tests: workspace sizing + eligibility + e2e BW for d ∈ {64,128,192,256} × {f16,bf16} × {causal,non-causal}), `fa2_varlen_smoke.rs` (5 tests: plan selection, lse_size formula, varlen FW with 3 packed sequences, varlen BW with 2 sequences, varlen × GQA). | done |
| 60 | FA2 head_dim {160, 224, 512} FW expansion (alpha.60) — **corrects Phase 59a's incorrect "permanently out-of-scope" claim**. The Candle fork (`EricLBuehler/candle`) has carried hd160/192/224/256 since 2023-07 (PR #245 by Laurent Mazare); hd224 was restored by PR #2688 (Michael Feil, 2024-12-31); hd512 was added by PR #3417 (Eric Buehler, merged 2026-03-28 — adds the `cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)` SMEM opt-in path and updates the splitkv block-size formula). Phase 60 vendors the 12 missing FW `.cu` files from those PRs into baracuda's FA2 tree (8 hd160/224 from `EricLBuehler/candle@main`; 4 hd512 from `huggingface/candle@5430d32c`) plus the corresponding `flash_fwd_launch_template.h` + `static_switch.h` patches. **BW path NOT extended** — hd160/224 fall on FA2 BW kernel's `kBlockKSmem = (kHeadDim % 64 == 0) ? 64 : 32` constraint (BW atom_layout assumes 64); hd512 needs `kBlockM = 32` to fit any SMEM budget but BW kernel_traits static-asserts `kBlockM >= 64`. Upstream FA2 and the Candle fork ship no BW for these three either — limitation is fundamental to FA2's BW algorithm, not an oversight. Phase 60 attempted both paths; the experiment + reasoning is documented in `VENDOR.md`, in code comments at the dropped registration sites, and in `FA2_BW_SUPPORTED_HEAD_DIMS` (kept at `{32, 64, 96, 128, 192, 256}`). Callers needing BW at hd160/224/512 transparently fall back to the bespoke 3-kernel SDPA BW pipeline (the only path that was supporting them previously, anyway). 12 new FW smoke test functions in `fa2_hdim_fanout_smoke`. `FA2_SUPPORTED_HEAD_DIMS` (FW) lifted to `{32, 64, 96, 128, 160, 192, 224, 256, 512}` — full Candle-fork-extended set. | done |
| 61 | In-place op infrastructure completion + same-pointer aliasing contract (alpha.61, Fuel-ask) — 2 new bf16/f16 FFI symbols + docstring tightening. (1) `baracuda_kernels_affine_inplace_{bf16,f16}_run` complete the 4-dtype matrix on top of the alpha.60 f32/f64 in-place affine helper, with f32-scalar ABI matching the forward `affine_{bf16,f16}_run` convention (avoids passing `__nv_bfloat16`/`__half` by value through the C ABI). Kernels reuse the forward upcast-to-f32 / downcast-to-storage pattern from `affine_contig_kernel_{f16,bf16}`. Unblocks Fuel's `Op::AddScalar`/`Op::MulScalar` in-place rewrites + weight-decay scaling on bf16/f16 model weights without the previous Cast → Affine → Cast scratch-buffer round-trip. (2) Documented same-pointer aliasing safety as a stable public contract on the three contig elementwise trailblazers — `unary_neg_f32_run` (covers ~30 plain unary launchers + `unary_param_*` family across all dtypes via the existing "Same device-pointer contract..." inheritance line), `binary_add_f32_run` (covers ~20 binary launchers), `ternary_clamp_f32_run` (already documented since alpha.36). Unblocks Fuel's planned in-place expansion (16+ unary in-place op families + 4 binary in-place op families + ClampInplace + PowIInplace) with zero new baracuda symbols for the elementwise case — Fuel dispatches the forward symbol with `x_ptr == y_ptr` (or `a_ptr == y_ptr` for binary). Strided in-place variants (Phase 62 candidate) deferred — v1 contract from Fuel's executor is contiguous + zero-offset. | done |
| 0 | Crate scaffolding, shared type vocabulary | done |
| 1 | int8 GEMM RRR (Fuel-blocking, 18 SKUs) | done |
| 2 | FP8 / int4 / bin GEMM completion | done |
| 3 | Elementwise + shape / layout (Categories B, B', C, C', D, N) | done |
| 4 | Reductions + scans + random (Categories E, F, Q) | done |
| 5 | Normalization + softmax + loss (Categories G, H, R) | done |
| 6 | Attention + linalg + FFT (Categories K, Linalg, U) | done |
| 7 | Convolution + pooling + indexing + embedding + segment (Categories I, J, L, M, S) | done |
| 8 | Quantization helpers + GGUF + MoE (Category P, V) | done |
| 9 | Sort / topk / image / NMS (Categories O, T) | done |
| 10 | sm_89 (Ada Lovelace) tuning sweep | done |
| 11 | Fuel feedback integration (alpha.27) — ScalarType ergonomics, Conv/Pool fanout, GGUF Q8_K MMVQ, i64 indices, Sparsemax cap lift, atomicAdd-via-CAS, build-env probe | done |
| 12 | PowI + ArgMax/Min u32/i32 outputs (alpha.28) — `IndexOutputElement` sealed trait | done |
| 13 | WriteSlice + Contiguize + sub-byte casts + Triu/Tril (alpha.29) — KV-cache fast path, retires Fuel's D2H/CPU/H2D fallback, plus `DeviceBuffer::zero()` (alpha.30) | done |
| 14 | Strided FFI siblings (alpha.31) — Affine, PowI, Triu/Tril, RoPE+SDPA, GGUF MMVQ activation-strided + W byte offset; 56 new FFI symbols | done |
| 15 | Quick wins + correctness cleanup (alpha.32) — MMVQ alignment guard, OneHot/Nonzero i64 wrappers, MoE fixture race fix | done |
| 16 | Pool completion (alpha.33) — bit-exact adaptive pool {1,2,3}d, bespoke LpPool {1,2}d, bespoke FractionalMaxPool {2,3}d; 48 new FFI symbols | done |
| 17 | SDPA / attention completion (alpha.34) — Flash SDPA sm_89 strided FW + SDPA BW GQA-broadcast atomicAdd | done |
| 18 | Sub-byte / quantized completeness (alpha.35) — f16/bf16 activations for `GgufMmvqPlan` across all 11 block formats × contig + strided; 44 new FFI symbols | done |
| 19 | Fuel retirement asks (alpha.36) — pool/conv FFI facade for cuDNN-backed plans + Upsample Nearest2d + NEW im2col/im2col1d/col2im1d bespoke; vendored Fuel Q8_1 for inspection; 140 new FFI symbols. Surfaced 1.0-freeze prereq for broader library-backed FFI facade audit | done |
| 20 | MoE — Item 4 from Fuel retirement (alpha.37): batched MMVQ × N-experts (36 new FFI symbols across 11 GGUF block formats × 3 activation dtypes + 3 pure-FP); MoE absorb-and-expose proved to be a no-op (Fuel hadn't evolved their kernels since Phase 8.5 vendor; 5 baracuda-side symbols already match) + 2 direct-FFI smoke tests | done |
| 21 | Bilinear interpolate expansion (alpha.38): `align_corners` + scale-factor overrides + f16/bf16 fanout (FW+BW). Breaking change to existing f32/f64 signatures. | done |
| 22 | MMVQ ncols≥64 debug assertion + cuSOLVER FFI facade (alpha.39): 10 cuSOLVER-backed plan families (Cholesky, LU, QR+ormqr, SVD/SvdBatched/SvdaBatched, eigh real+complex, eig, lstsq, solve, inverse) wrapped behind ~50 flat C symbols in `baracuda-kernels-sys/src/cusolver_facade.rs`; closes the Phase 19 library-backed FFI facade gap for cuSOLVER. No feature gate (cuSOLVER ships with the CUDA toolkit). | done |
| 23 | cuFFT + cuRAND FFI facade (alpha.40): 6 cuFFT plan families (FFT 1d/Nd C2C, R2C, C2R) × c32/c64 + f32/f64 + 2 cuRAND families (Uniform, Normal) × f32/f64 = 32 flat C symbols in `baracuda-kernels-sys/src/{cufft,curand}_facade.rs`. cuSPARSE skipped — no baracuda-kernels plans wrap it today. | done |
| 24 | Cutlass GEMM re-export FFI facade (alpha.41): 210 trampolines (70 SKU families × {run, workspace_size, can_implement}) in `baracuda-kernels-sys/src/cutlass_reexport.rs` exposing the full Cutlass GEMM surface (fp16/bf16/tf32/f32_simt/f64/s8/u8 × {rcr, rrr} × {plain, bias, bias+relu/gelu/silu} + strided-batched fp16/bf16). cuTENSOR / NPP / CV-CUDA skipped — no baracuda-kernels plans wrap them. Completes the Phase 19 library-backed FFI facade 1.0-freeze prereq. | done |
| 25-26 | Segment/EmbeddingBag BW completion + BatchedOrmqrWy complex (alpha.42): 9 new Rust plans + 24 new FFI symbols for Segment Max/Min/Prod BW (sorted + unsorted, f32/f64), Unsorted Segment Prod FW (`atomicCAS`-retry mul), EmbeddingBag Max FW+BW (f32/f64/f16/bf16 × i32/i64). Plus BatchedOrmqrWy complex (Complex32, Complex64) via the bespoke WY-block kernels + cuBLAS C/Z gemmStridedBatched (4 new bespoke FFI + 2 cuBLAS symbols). | done |
| 27 | Q8_1 perf inspection (alpha.42 doc-only): Multi-M MMVQ opportunity identified, kept doc-only — bigger ROI than reformatting Q8_1. | done |
| 28 | API hygiene for 1.0 prep (alpha.43): new `KernelDtype` umbrella marker trait extending `Element`/`IntElement`/`FpElement`/`BinElement`; `#[non_exhaustive]` audit across the op-family `*Kind` enums + auxiliary tag enums + `Error` types. `ElementKind` / `LayoutSku` / `ArchSku` / `EpilogueKind` / `ActivationKind` / `Workspace` intentionally left exhaustive (hot-path-dispatched). | done |
| 29 | Cross-implementation benchmark suite (alpha.44): 10 new criterion+CUDA-event benches comparing baracuda against cuBLAS / cuDNN at LLM-typical shapes (GEMM f32/f16/bf16, MMVQ all qtypes, Softmax, LayerNorm, RMSNorm, Conv2d, MaxPool2d, Reductions, Elementwise, Flash SDPA+GQA). ~2,750 LOC of bench code + 13 bench binaries total. Critical finding: baracuda f16/bf16 GEMM is **2-4× slower than cuBLAS at M=1/M=32** (decode regime); validates the deferred Phase 27 multi-M MMVQ port. See [`BENCHMARKS.md`](crates/baracuda-kernels-bench/BENCHMARKS.md) for the methodology + sample run. | done |
| 30 | f16/bf16 GEMM cuBLAS fast-path (alpha.45): adds `PlanPreference::prefer_backend: Option<BackendKind>` + thread-local cuBLAS-handle cache to `GemmPlan`. Heuristic: cuBLAS for f16/bf16 at `2 ≤ M < 128` (decode batch); CUTLASS otherwise. **3× speedup at M=32 f16** (55.6µs → 19.0µs, parity with cuBLAS direct). M=1 stays on CUTLASS (cuBLAS RCR→col-major transa=T mapping slower than CUTLASS sm_80 GEMV-tile at K=N≥2048). Capture-mode auto-fallback to CUTLASS (cuBLAS-classic not capture-safe). 9 new smoke tests. | done |
| 31 | Fuel Phase 6c.2 storage.rs unblock (alpha.46): 5 gaps closed — ELU α parameter (breaking; 8 sigs modified), `powf` (8 new), `step` + `gelu_erf` (16 new), cast `u32`/`i16` (36 new × 2 directions), `reduce_sum_to`/`reduce_max_to` broadcast-reverse reductions (8 new). **~76 new/modified FFI symbols + 17 new smoke tests.** Unblocks Fuel's full PTX retirement (AFFINE/UNARY/BINARY/CAST/REDUCE/INDEXING/TERNARY/FILL/SORT modules). | done |
| 32 | Descriptor `#[non_exhaustive]` + builder pattern (alpha.47): 18 descriptors retrofitted with `::new()` builders + chainable setters (`with_stride`/`with_padding`/`with_dilation`/etc.). Conv {1,2,3}d + ConvTranspose {1,2,3}d + Pool {1,2,3}d + AdaptivePool {1,2,3}d + LpPool {1,2}d + FractionalMaxPool {2,3}d + Interpolate + InterpolateBackward. **Breaking change for downstream struct-literal callers** — pre-1.0 hardening. Migration: `Conv2dDescriptor { ... }` → `Conv2dDescriptor::new(input_shape, filter_shape, element).with_stride(...)`. | done |
| 33 | Multi-M MMVQ via Q8_1 staging (alpha.48): closes Phase 27's deferred opportunity. NEW `GgufMmvqMultiMPlan` + `quantize_q8_1` staging kernel + 4 Q8_0 multi-M launchers (M ∈ {1, 2, 4, 8}). **Bench: 7.29-7.96× speedup at M=8** on Llama-2 7B layer shapes (4096²; 11008×4096; 32000×4096). Q8_0 only this phase (clean partial); 9 remaining block formats (Q4_0/Q4_1/Q5_0/Q5_1/Q2_K..Q6_K) are mechanical fanout for a follow-up. 8 new FFI symbols (3 staging + 4 multi-M + 1 workspace). | done |
| 34 | Multi-M MMVQ block format fanout (alpha.49): 9 remaining GGUF formats shipped — Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K. 36 new FFI symbols (9 fmts × 4 M-sizes). **Bench at N=K=4096 M=8**: Q5_0 **17.32×**, Q5_1 15.05×, Q4_0 12.78×, Q4_1 12.15×, Q8_0 8.79× — type-0/1 formats massively exceeded Phase 27's 3-7× target. K-quants (Q2_K..Q6_K) hit 3-7× at M=8 (larger 256-elem super-blocks dilute weight-reuse savings). Q8_K MMVQ correctly rejected at select() — bespoke per Phase 11.4. | done |
| 35 | Test-infra hardening (alpha.50): **first zero-failure regression** in the entire Phase 22-35 sweep (2229/0 across 638 binaries). Five fixes: (a) `mmvq_w_offset_alignment_misaligned_rejected_debug` `#[cfg(debug_assertions)]` gate; (b) cuBLAS handle retry with 5× linear backoff (Phase 30 parallel-init race); (c) cuDNN handle retry on CTC path (1001 NOT_INITIALIZED race); (d) `Stream::capture` panic-safe Drop guard (ThreadLocal capture state leak under cargo's thread reuse → cudaErrorStreamCaptureImplicit on subsequent tests); (e) **`cudaResourceDesc` 48→128 byte expansion + `repr(align(8))`** (Rust struct under-allocated by 16+ bytes AND missing 8-byte alignment that the union's `void*`/`size_t` arms require — caused release-only STATUS_ACCESS_VIOLATION in wave5_smoke). | done |
| 36 | Fuel 6c.4 unblock — Phase 1/3 (alpha.51): RoPE apply with precomputed cos/sin tables (FW+BW × 4 fp dtypes; 16 symbols) + Fill missing dtypes & strided variant (3 new contig + 11 strided; 28 symbols) + Argsort dtype fanout (u8/i8/u32/i16/bf16/f16/fp8e4m3; 14 symbols). 58 new FFI declarations total. | done |
| 37 | Fuel 6c.4 part 2/4 (alpha.52): Reduce family Gap 1 — `reduce_min_to`/`prod_to` broadcast-reverse for 4 fp dtypes (16 symbols) + integer-dtype single-axis sum/min/max/prod + argmin/argmax for U8/I8/U32/I16/I32/I64 (48 symbols, with U64/I64 widened accumulator + store-time narrow on Sum/Prod). 64 new FFI declarations total. Documented bit-exact wrap-on-overflow contract for u8/u32 sum/prod. | done |
| 38 | Fuel 6c.4 part 3/4 (alpha.53): Ternary `where_cond` dtype-matrix fanout — Cond lifted to template parameter; U8 (existing, untouched) + U32 + I64 cond × {f32/f64/f16/bf16, u8/i8/u32/i16/i32/i64, fp8e4m3} value × {contig, strided}. 87 new FFI declarations (58 `_run` + 29 `_can_implement`). Existing `where_<value>_run` family preserved bit-identically (default Cond=uint8_t). | done |
| 39 | Fuel 6c.4 part 4/4 (alpha.53, bundled with Phase 38): Indexing Tier 1 — NEW scatter (pure assign) + index_add for {f32/f64/f16/bf16} × {i32, i64idx} (16 syms) + gather u8idx extras for {f32, f64} (2 syms). 18 new FFI symbols total. Existing per-axis stride arrays meant no separate contig/strided split needed. f16/bf16 index_add uses the Phase 11.3 `atomic::add<T>` atomicCAS helper. Scatter documented + tested with disjoint-target indices (last-writer-wins on collisions, caller-aware non-determinism). | done |
| 40 | Fuel 6c.4 final cleanup (alpha.54): multi-block radix argsort via CUB `DeviceSegmentedRadixSort` for `row_len > 1024` (4 dtypes × 3 entries = 12 syms; bitonic stays for ≤1024) + Indexing Tier 2 integer value-dtype matrix (gather/index_select/scatter for u8/i8/u16/i16/u32/i32/i64 × i32/i64idx = 38 syms; index_add for i32/u32/i64 only = 6 syms). 56 new C symbols total. New `atomic::add<int64_t>` specialization via `unsigned long long*` reinterpret. Tier 3 (fp8e4m3 + sub-32-bit ints for index_add) deferred — no concrete caller. | done |
| 41 | Fuel 6c.5 final unblock (alpha.55): RoPE interleaved-pair (Gap 7) + RoPE THD-layout (Gap 8) variants. 28 new FFI symbols (FW+BW × 4 fp dtypes × 2 variants + `_can_implement` companions). **Closes the entire Fuel 6c.4/6c.5 batch ask** — Fuel can now drop the last `Id::Reduce` PTX module + retire `fuel-cuda-kernels` workspace member + drop the `cudaforge` build dep. Discovery: existing `rope_apply_*` was already using `(2k, 2k+1)` pairing (not `(i, i+d/2)` as the brief stated) → interleaved symbols are name-aliases on the same kernel; THD is genuinely new. | done |
| 42 | Flash Attention v2 vendor + `FlashSdpaPlan` backend (alpha.56): Tri Dao's FA2 v2.8.3 (BSD-3) vendored under `crates/baracuda-kernels-sys/vendor/flash-attention/` — Tier 1 (head_dim=128, fp16+bf16, sm_80, FW only) — wired as `BackendKind::FlashAttentionV2` on `FlashSdpaPlan` behind the `fa2` cargo feature. Heuristic routes long-context (seq_q×seq_k ≥ 1024×1024) shapes to FA2, bespoke otherwise; `PlanPreference::prefer_backend` overrides. PyTorch shim headers (at::PhiloxCudaState + C10_CUDA_CHECK) decouple the vendor from torch deps. Tier 2 (BW, varlen, paged, other head_dims) deferred. | done |
| 43 | mHC.cu vendor + `HyperConnectionPlan` family (alpha.56): DeepSeek-AI's Manifold-Constrained Hyper-Connections residual-mixing op (arXiv:2512.24880) from AndreSlavescu/mHC.cu (MIT) vendored under `crates/baracuda-kernels-sys/vendor/mhc/` — Tier 1 (static-H, bf16 only) — exposed as `HyperConnectionPlan` behind the `mhc` cargo feature. Replaces bare `y = x + sublayer(x)` residual with a learned `n×n` Sinkhorn-Knopp doubly-stochastic mixing matrix. Tier 2 (BW, dynamic-H, fp16/f32) deferred. Requires cuBLAS-Lt (already linked). | done |
| 44 | ozIMMU FP64-via-Int8-TC backend (alpha.56): enp1s0/ozIMMU (MIT) — Ootomo/Ozaki/Yokota's Ozaki-scheme DGEMM that synthesizes FP64 from S² int8 tensor-core matmuls — vendored under `crates/baracuda-ozimmu-sys/vendor/ozimmu/` with `cutf` submodule pinned alongside. NEW `baracuda-ozimmu-sys` + `baracuda-ozimmu` sibling crates. Wired into `GemmPlan` f64 path as opt-in `BackendKind::Ozaki { slices }` (default stays on CUTLASS/cuBLAS DGEMM — Ozaki is NOT bit-equivalent). Two patches: direct-link mode (no LD_PRELOAD), exclude `cublas.cu`/`culip.cu`. | done |
| 46 | FlashInfer cherry-pick — paged-KV decode + sort-free sampling + cascade attention (alpha.57 Checkpoint A, **closed in the alpha.58 consolidation pass**): surgical extraction of three FlashInfer kernel families (Apache-2.0, v0.6.12, commit `eee0d75f`) vendored under `crates/baracuda-kernels-sys/vendor/flashinfer/` (~12 kLOC across 25 headers, no wholesale wrap). NEW plan families: `BatchPagedDecodePlan` + `PagedKvAppendPlan` (vLLM-style paged KV cache decode), `TopKTopPSamplingPlan` (sort-free TopK/TopP/MinP/combined samplers), `CascadeAttentionPlan` (LSE-merge for prefix-cache sharing). NEW `BackendKind::FlashInfer` + `RandomKind::Multinomial` discriminants. NEW `flashinfer` cargo feature on both `baracuda-kernels-sys` and `baracuda-kernels` (default OFF). 7 MSVC-portability patches to vendored headers (see `vendor/flashinfer/VENDOR.md`). **Checkpoint B (alpha.58 consolidation)**: `flashinfer_paged_decode_launcher.cu` now compiles cleanly under MSVC nvcc — root cause was `std::max(unsigned long, size_t)` type mismatch inside `decode.cuh` (the earlier hypothesis about `cudaLaunchKernel_ptsz` was incorrect). Patched via `static_cast<size_t>(...)` on both arguments; launcher TU also carries a defensive `cudaLaunchKernel` shim macro. All 4 launchers now build under the `flashinfer` feature. | done |
| 44b | ozIMMU clean-fork + cutf elimination + Windows port (alpha.57): full internalize of ozIMMU sources (no longer vendored — we own them at `crates/baracuda-ozimmu-sys/cuda/`). `cutf` submodule eliminated entirely (upstream went offline); ~360 LOC of useful FP / cp_async utilities preserved as native `baracuda_fp_bits.cuh` + `baracuda_cp_async.cuh`; ~2,200 LOC of cutf duplicates deleted. Portable `baracuda::Uint128` replaces `__uint128_t` for Windows compile (typedef alias on Linux — bit-for-bit preservation). LD_PRELOAD path removed entirely. Linux + Windows both build clean. | done |
| 44c | ozIMMU RIKEN-RCCS perf-enhancement variants (alpha.57, no version bump): folds in `accelerator_for_ozIMMU` (Uchino/Ozaki/Imamura 2024, arXiv:2409.13313) — three new variants `EF` (group-wise error-free summation; chains int8 cublasGemmEx with `beta_i=1` to delay int32→f64 materialization), `RN` (nearest-rounding `(a+t)-t` split; ~2 extra effective bits per slice), `H` (= EF + RN), plus n-blocking (chunk `n > 12288` into 8192-wide pieces on the int8 GEMM call). Variant selected via `BackendKind::Ozaki { slices }` discriminant's high-3-bits field; `ozaki_slices::{base,ef,rn,h}(s)` helper constructors in `baracuda-kernels-types::sku`. NEW `OzakiVariant` enum + `Handle::dgemm_with_variant` on `baracuda-ozimmu`. Source-compatible with Phase 44b callers (`slices: 8` decodes as Base/S=8). **Discovered + fixed a pre-existing Phase 44b MSVC bug** in `axby` / `axy_complex`: upstream's `(1l << 44)` overflows on Windows (where `long` is 32-bit, LLP64) → silent `inf` output. Fixed by switching to `static_cast<double>(1ull << 44)`. 9 new accuracy/variant/n-blocking smoke tests, all green on RTX 4070; the pre-existing Phase 44b accuracy_smoke tests (4 cases) also unbreak. | done |
| 47 | Fused Linear Cross-Entropy (alpha.56, single-kernel port from LinkedIn's Liger-Kernel BSD-2): NEW `FusedLinearCrossEntropyPlan` family that fuses lm_head GEMM + CE loss in a chunked outer loop, never materializing the `[BT, V]` logits tensor. At BT=16K, V=128K, bf16 (Llama-3-class) saves **5-10 GiB of activation memory**. Bespoke per-chunk fused softmax+CE+gradient kernel (FP32 accumulator across 4 fp dtypes — f16/bf16/f32/f64); GEMMs dispatched via `cublasGemmEx`. Backward produces `grad_input`+`grad_weight` during the FW pass (chunked loop); BW call just scales by `dy_scalar` (no-op when `dy=1.0`, the typical "CE is the last layer" case). 16 new bespoke FFI symbols (per_row + per_row_cast + scalar_finalize + inplace_scale, each × 4 dtypes) + 1 count-non-ignore helper + `cublasGemmEx` binding. NEW `LossKind::FusedLinearCrossEntropy` variant. **Algorithm credit**: LinkedIn Liger-Kernel (BSD-2-Clause, clean-room CUDA reimplementation — no source vendored). | done |
| 45 | SmoothQuant compose + YaRN/LongRoPE Rust helper (alpha.56, no version bump — consolidation phase will bump): **two zero-new-CUDA pure-Rust additions**. (a) `SmoothQuantLinearPlan<TIn, TWQ>` (in `crates/baracuda-kernels/src/quantize/smoothquant.rs`) composes the existing Phase 8.3 `quantized_linear_w8a8` kernel + `fill_<dt>` broadcast for the per-tensor activation scale. Caller supplies pre-smoothed-and-quantized int8 activations + int8 weights (smoothing itself is offline Python per the SmoothQuant paper — mit-han-lab/smoothquant MIT, Xiao et al. ICML 2023; not in scope). (b) `RopeScaledTableBuilder` + `RopeScaling` enum (Linear / YaRN / LongRoPE, in `crates/baracuda-kernels/src/attention/rope_scaling.rs`) — host-side cos/sin table builder feeding the Phase 36 `rope_apply_<dt>_run` kernel. YaRN (jquesnelle/yarn MIT, Peng et al. arXiv:2309.00071) implements §3.2 NTK-by-parts frequency interpolation + §3.3 attention-temperature absorption into cos/sin. LongRoPE (microsoft/LongRoPE MIT, Ding et al. arXiv:2402.13753) multiplies inv-freq by caller-supplied per-dim factors (evolutionary search itself is offline + out of scope). Existing Phase 36 `RopeApply*` types source-compat preserved. | done |
| 51 | Arbitrary-mask `FlashSdpaPlan` + spec-decode composition doc (alpha.57, no version bump — consolidation phase will bump): NEW optional `mask: TensorRef<f32, 4>` field on `FlashSdpaArgs` routing to a bespoke arbmask SDPA kernel that adds an f32 `[B, H, Q, K]` additive bias to `S = Q·K^T·scale` before softmax. Unlocks spec-decode tree masks (EAGLE / Medusa / lookahead), MoE expert masking, prefix-LM, sliding-window with attention sinks — all entirely from caller-side composition. 4 dtypes (f32/f16/bf16/f64) × `_run` + `_can_implement` = 8 new FFI symbols. `is_causal` composes with the mask correctly (`-INF + finite == -INF`). New header `baracuda_attn_arbmask.cuh` reuses Phase 6.6's online-softmax tile pipeline; 1 new .cu instantiation file. FA2 vendor untouched (FA2 v2.8.3's `Mask` template has no arbitrary-mask hook). Runnable example at `crates/baracuda-kernels/examples/speculative_decode_compose.rs`; design doc at [`docs/guides/spec-decode.md`](docs/guides/spec-decode.md). FW only; BW deferred. | done |
| 50 | Mamba-2 SSD chunk-scan + Dao-AILab causal-conv1d (alpha.57, gated behind `mamba` cargo feature): **opens the state-space LLM class (Mamba-2 8B, Codestral-Mamba, Falcon-Mamba, Zamba2 — Mamba-1 selective_scan deferred to Phase 50b).** NEW `SsdChunkScanPlan` + `SsdChunkScanBackwardPlan` (lives under `attention` because of the SSD-as-attention duality) and `CausalConv1dPlan` + `CausalConv1dBackwardPlan` (top-level module — bespoke kernels, no cuDNN dep). Vendor attribution + LICENSE at `crates/baracuda-kernels-sys/vendor/causal-conv1d/` (Tri Dao, BSD-3) and `crates/baracuda-kernels-sys/vendor/mamba/` (state-spaces/mamba, Apache-2.0). Hand-port of the upstream Triton SSD reference + causal-conv1d primitive. **Dtypes**: causal-conv1d f32/f16/bf16/f64 × widths 2/3/4 × {SiLU, identity}; SSD f32/f16/bf16 (no f64 upstream). FW caps state at D,N ≤ 256; BW tighter at 64 (SMEM budget). 30 new FFI symbols (8 causal-conv1d FW + 8 BW + 6 SSD FW + 6 SSD BW + 2 can_implement extras). 5 new smoke tests (causal_conv1d_smoke/bw + ssd_chunk_scan_smoke/bw + mamba2_block_smoke). | done |
| 50b | Mamba-1 `selective_scan` (alpha.57, gated behind the same `mamba` cargo feature as Phase 50): **completes the state-space LLM coverage by adding the original Mamba-1 op family that powers Mamba-7B, Falcon-Mamba, and Codestral-Mamba** — Phase 50's SSD covers Mamba-2 / Codestral-Mamba / Falcon-Mamba / Zamba2, but every Mamba-1-shipping model still uses v1's `selective_scan`, not v2's SSD reformulation. NEW `SelectiveScanPlan` + `SelectiveScanBackwardPlan` (sibling to `SsdChunkScanPlan` under `attention/`). Shape `(B, L, D, N)` with the full Mamba-1 surface: optional `D[d]` skip, optional SiLU-gated `z[t, d]` tail, optional `delta_bias[d]` + optional `softplus(delta)` mapping (all 9 args of upstream `selective_scan_fn` wired). Dtypes f32/f16/bf16 (complex deferred — no shipping LLM uses it). Hand-port of `state-spaces/mamba`'s `csrc/selective_scan/` under Apache-2.0; same `vendor/mamba/` directory as Phase 50 (VENDOR.md updated, no new LICENSE file). FW caps state at `N ≤ 256`; BW uses two-pass record-then-reverse with `B*D*L*N*sizeof(T)` workspace. NEW `AttentionKind::SelectiveScan = 8` variant (`#[non_exhaustive]` so source-compat). 17 new FFI symbols (3 FW + 3 FW-can-impl + 3 BW + 1 workspace-bytes + module-internal launchers). 3 new smoke tests (selective_scan_smoke covering 4 option-combinations + f16/bf16 loose-tol, selective_scan_bw_smoke with FD checks on du/ddelta/dA + topology rejection, mamba1_block_smoke end-to-end). | done |
| 52 | NCCL foundation crate pair (alpha.57, no version bump — consolidation phase will bump): `baracuda-nccl-sys` (raw FFI types + libloading lazy-resolve, NO bindgen / NO link-time dep) + `baracuda-nccl` (safe `Communicator` with full collective surface — `all_reduce` / `reduce` / `reduce_scatter` / `all_gather` / `broadcast` / `send` / `recv` + group API + `NcclMem` + custom `pre_mul_sum` reduction op + `register` / `deregister` for zero-copy). **The distributed-roadmap prerequisite** for Ring Attention, distributed MoE, Megatron-LM tensor parallelism, FSDP-style shard collectives — Phase 52 only ships the substrate; consumer plans land in Phase 53+. Spec-named API: `Communicator::new_single_gpu` / `new_with_id`, cached infallible `rank()` / `world_size()`, `NcclReduceOp` / `NcclUniqueId` / `NcclDataType` aliases, `NcclUniqueId::generate()`. Linux-primary (NCCL ships with the CUDA toolkit there); Windows builds clean and defers the "is NCCL installed?" question to first `nccl()` call (loader fails with `LoaderError::LibraryNotFound`). 20 new smoke tests (10 dtype mapping — runs on every host; 10 `#[ignore]` NCCL-required). No baracuda-kernels integration in this phase. | done |
| 49 | Apex optimizer subset (alpha.57, gated behind `optim` cargo feature): **deliberate scope expansion — training-framework-adjacent.** NEW sibling crate `baracuda-optim` (~600 LOC Rust + ~750 LOC CUDA) vendoring the NVIDIA Apex (BSD-3-Clause) `multi_tensor_apply` idiom + fused Adam / LAMB / SGD functors. Single launch over thousands of parameter tensors (Apex `MAX_TENSORS_PER_LAUNCH = 110` per batch, multi-launch transparent) — eliminates the ~10,000-launch optimizer step overhead on 32B-param models. Plans: `AdamStepPlan<T>` (f32/f16/bf16 + AdamW mode), `LambStepPlan` (f32; two-stage with atomicAdd-fused L2-norm + sqrt + trust-ratio scaling), `SgdStepPlan<T>` (f32/f16/bf16 + momentum + Nesterov + weight-decay). Inference-only consumers (e.g. Fuel) don't pay the FFI surface cost — the vendored sources only build / link when the feature is enabled. Re-exported under `baracuda_kernels::optim` when enabled. **Measured 41× speedup at 1000-tensor multi-tensor Adam vs 1000 individual launches on RTX 4070** (0.173 ms vs 7.096 ms; smoke test in `crates/baracuda-optim/tests/multi_tensor_dispatch_smoke.rs`). 4 smoke tests, 6 GPU tests total, all green. | done |
| 53 | bitsandbytes NF4 dequant + GEMV vendor — QLoRA inference (alpha.57, gated behind `bnb_nf4` cargo feature): **opens the QLoRA-trained Llama / Mistral / Qwen inference class** by vendoring the bitsandbytes (Dettmers et al. arXiv:2305.14314, MIT) NF4 (NormalFloat 4-bit) dequant + GEMV kernels. NF4 is the dominant 4-bit format for QLoRA-trained prebuilts on the HuggingFace Hub — **distinct from GGUF Q4_0** (symmetric int4*scale, llama.cpp, Phase 8) and AWQ int4 (asymmetric int4 + zero-points). NF4 uses a 16-entry **non-uniform quantile codebook** derived from the inverse CDF of `N(0, 1)` — dequant is a 16-entry lookup, not arithmetic; better accuracy than symmetric int4 for normally-distributed weights. NEW plan trio: `Nf4DequantizePlan<T>` (bulk unpack `[N/2, K]` u8 → `[N, K]` T), `Nf4MmvqPlan<T>` (M=1 single-vector decode GEMV), `Nf4MmvqMultiMPlan<T>` (M ∈ {1, 2, 4, 8} batched-decode with weight gmem reuse, Phase 33 pattern applied to NF4). 11 new FFI symbols (3 dequant + 2 M=1 + 6 multi-M). Pack layout matches bitsandbytes upstream `Linear4bit`: pair-packed nibbles in `[N/2, K]` u8 (N must be even) + `[N * (K/block_size)]` f32 per-block absmax (block_size typically 64). Activation/output dtypes f16+bf16 (PyTorch convention); f32 accumulator. Codebook reproduced bit-identical to upstream as device-side switch + host-side `NF4_CODEBOOK: [f32; 16]` const + `nf4_pack_weight` host helper. Vendor metadata at `crates/baracuda-kernels-sys/vendor/bitsandbytes/{LICENSE,AUTHORS,VENDOR.md}`. 3 smoke test files (dequant roundtrip, M=1 GEMV f16+bf16, multi-M f16 vs M=1-looped). Out of scope: 8-bit optimizers (Phase 49 overlap), LLM.int8 (Phase 45 obsoletes), FP4 (different codebook — separate phase if asked), double quantization (Tier 2). | done |
| 54 | xFormers BlockSparseAttention + 2:4 sparse GEMM (alpha.57, no version bump; clean-room hand-port of facebookresearch/xformers BSD-3-Clause algorithmic reference): NEW `SdpaBlockSparsePlan` (`xformers_blocksparse` cargo feature) — block-sparse SDPA FW where the attention mask is a per-block boolean pattern `[B, H, num_blocks_q × num_blocks_k]` (uint8); only the active (q_block, k_block) pairs participate in the QK^T matmul + online-softmax accumulation. Different from Phase 51's arbitrary additive-mask path (which still computes every cell) — block-sparse actually SKIPS compute on masked blocks → real wall-clock speedup on long-context attention with known sparse patterns. NEW `GemmSparse24Plan` (`xformers_sparse24` cargo feature) — 2:4 structured sparsity GEMM consuming pre-compressed `[M, K/2]` weights + `[M, K/8]` u16 metadata. **Tier-1 implementation**: inflate-then-dense reference matmul (correctness first; sparse-tensor-core `mma.sp.sync` / cuSPARSELt backend deferred to Tier 2). 16 new FFI symbols (8 block-sparse SDPA × 4 dtypes × 2 entries; 11 sparse24 × 3 dtypes × 4 entries with workspace_bytes helper). NEW `AttentionKind::BlockSparseAttention = 9` variant (`#[non_exhaustive]` so source-compat). 3 smoke tests (block-sparse all-ones-matches-dense + diagonal-band + empty-pattern; sparse24 matches host reference + K-rejection + throughput timing). Vendor attribution at `crates/baracuda-kernels-sys/vendor/xformers/` (no upstream sources vendored verbatim — algorithmic reference only). NOT vendored from xFormers: memory-efficient attention (overlaps with FA2 vendor); fused biases / RoPE / norm (overlaps with existing baracuda phases); Triton kernel paths (no Triton toolchain). | done (Tier 1) |
| 55 | TransformerEngine FP8 cast + delayed-scaling recipe (alpha.57, gated behind `tensor_engine` cargo feature; clean-room hand-port of NVIDIA TransformerEngine Apache-2.0 algorithm — only the cast + recipe subset). NEW sibling crate pair `baracuda-transformer-engine-sys` + `baracuda-transformer-engine`. The differentiated value of TE is the **per-tensor delayed-scaling recipe with amax history** for stable FP8 training; that's the load-bearing piece this phase ships. Public API: `Fp8Recipe` (RAII handle holding amax_history ring + scale + scale_inv device scalars), `Fp8CastPlan<TIn>` (fused FP8 cast + `max(\|x\|)` amax reduction in one kernel — atomicMax into `amax_history[write_pos]`), `Fp8DequantPlan<TOut>` (symmetric dequant via `scale_inv`). Both formats: E4M3 (max=448) for fwd/weights, E5M2 (max=57344) for grads. Wide dtypes: f32/f16/bf16. 4 new C-ABI symbols (`baracuda_te_fused_cast_amax_run` / `baracuda_te_dequant_run` / `baracuda_te_recipe_update_run` / `baracuda_te_recipe_init_run`) + format/dtype id helpers. **NO cuDNN dep** (cast/recipe paths don't need it — cuDNN is only needed for `fused_attn`, which we skip). **NO pybind11** (raw C ABI, not Python). 10 GPU smoke tests, all green on RTX 4070. **Sm_89 reality check**: FP8 storage + cast intrinsics work natively on Ada, but tensor-core FP8 MMA throughput equals BF16 — so the wins here are bandwidth-saving (KV cache, weight storage, activation memory) not compute. Recipe machinery is forward-compatible with Hopper (sm_90a) / Blackwell (sm_100) where the MMA throughput win also materializes. Deliberately NOT lifted: `normalization` (Phase 5), `fused_rope` (Phase 14/36/41), `fused_attn` (Phase 17/42; cuDNN dep), `fused_softmax` (Phase 5), `activation` (Phase 3/31), `gemm` (Phase 1+24+30), `comm_gemm_overlap` (Hopper TMA), `fused_router` (Phase 8+20), `hadamard`/`newton_schulz`/`swizzle`/`permutation` (niche), `multi_tensor` (Phase 49), `dropout` (composable), Python bindings (out of scope). Algorithmic reference: `transformer_engine/common/{cast,recipe}/*.cu` upstream + FP8 spec Micikevicius et al. 2022 (arXiv:2209.05433). Vendor attribution + full Apache-2.0 text at `crates/baracuda-transformer-engine-sys/ATTRIBUTION.md`. | done |
| 57 | Megatron-LM tensor-parallel primitives (alpha.57, no version bump; gated behind `megatron_tp` cargo feature). **NEW sibling crate `baracuda-megatron`** — pure-composition over `baracuda-cublas` (local GEMM via `cublasSgemm` for f32, `cublasGemmEx` with `Compute32F` accumulator for f16/bf16) + `baracuda-nccl` (cross-rank `all_gather` / `all_reduce` collectives). **NO new CUDA kernels** — foundational TP primitives for Megatron-style models are pure orchestration; the kernel substrate already exists in baracuda. NEW plans: `ColumnParallelLinearPlan<T>` (splits W along output dim; FW `Y_local = X @ W_local^T` + `all_gather`; BW `dX_partial = dY_local @ W_local` + `all_reduce(Sum)`, `dW_local = dY_local^T @ X` local) and `RowParallelLinearPlan<T>` (splits W along input dim; FW `Y_partial = X_local @ W_local^T` + `all_reduce(Sum)`; BW `dX_local = dY @ W_local` local, `dW_local = dY^T @ X_local` local — **no BW collective**, the Megatron pairing only needs one collective per layer-pair). NEW `TensorParallelContext` borrow type holding `&Communicator` + `in_features` / `out_features` / cached `rank` / `world_size`. Dtypes: f32 always; f16 + bf16 behind the crate-level `half-crate` feature (which the kernel-facade `megatron_tp` feature pulls in). Tier 1 scope — bias rejected at call site with a Tier-2 marker error (caller can Affine-add post-FW; matters for RowParallel where the bias must be added **after** the all_reduce so it doesn't get summed N times). 5 smoke tests across 3 files: `column_parallel_smoke` (FW + BW, single-rank, matches CPU `Linear` ref), `row_parallel_smoke` (same), `multi_rank_scaffold` (`#[ignore]`-gated 2-GPU scaffold — exits cleanly on single-GPU dev boxes). Algorithmic reference: Shoeybi et al. arXiv:1909.08053 (NVIDIA Megatron-LM, Apache-2.0); no source vendored. Out of scope: async overlap (Hopper TMA); sequence parallelism (Phase 56's domain); pipeline parallelism (future phase); VocabParallelEmbedding (future polish); distributed gradient accumulation (Phase 58's domain); expert parallelism (separate phase). | done |
| 58 | DistributedAdam — ZeRO-1-style sharded optimizer state (alpha.57, no version bump; gated behind the new `distributed_optim` cargo feature on `baracuda-optim`, pulls `baracuda-nccl` as optional dep). **Pure-Rust composition** over Phase 49 [`AdamStepPlan`] + Phase 52 NCCL collectives — **NO new CUDA kernels**, **NO new `baracuda-kernels-sys` FFI**. NEW `DistributedAdamStepPlan<T>` wrapping the inner Adam plan + a borrowed `&Communicator`; orchestrates the canonical ZeRO-1 protocol: `all_reduce(grads, Sum, in-place)` → local Adam step on this rank's `1/world_size` shard → `all_gather(updated_params, in-place)`. **Single-rank degenerate case** (`world_size == 1`) elides both collectives and reduces to `AdamStepPlan::step` bit-exactly (smoke test verifies this on single-GPU dev hardware). f32 + f16 + bf16 dtypes, AdamW + classic mode, mixed-precision `step_with_f32_state` variant. NEW `shard_range(n, rank, world_size)` helper matching `torch.chunk` semantics. Phase 58 constraint: tensor element counts must be `world_size`-multiples (ring all_gather symmetry); per-tensor broadcast fallback for ragged shards is future work. Out of scope: ZeRO-2 (gradient sharding); ZeRO-3 (parameter sharding during FW/BW); DistributedLamb / DistributedSGD (same pattern, defer until concrete demand); CPU-offload optimizer state; 8-bit distributed optimizer state. 3 smoke test files (4 pure-Rust shard_range tests run unconditionally; 2 single-rank GPU smokes `#[ignore]`-gated for NCCL; 1 multi-rank scaffold `#[ignore]`-gated for 2+ GPU validation). Algorithmic reference: Rajbhandari et al. SC20 "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", Microsoft DeepSpeed (Apache-2.0; no source vendored — pure Rust composition). | done |
| 56 | Ring Attention — sequence-parallel attention (alpha.57, no version bump — consolidation phase will bump; gated behind the new `ring_attention` cargo feature, pulls `baracuda-nccl` + `baracuda-nccl-sys` as optional deps). **First Phase 52 NCCL consumer** — proves the substrate. Hand-port of Liu/Yan/Abbeel 2023 (arXiv:2310.01889; algorithmic reference at `https://github.com/lhao499/RingAttention`, Apache-2.0 — no JAX source vendored, clean-room CUDA implementation). NEW `RingAttentionPlan<T>` + `RingAttentionDescriptor` + `RingAttentionArgs` in `crates/baracuda-kernels/src/attention/ring_attention.rs`; bespoke `kernels/attention/ring_attention_kernel.cu` (~480 LOC kernel + ~390 LOC plan). Per-rank online-softmax fold of resident K/V chunk into persistent `(o_acc, m_acc, l_acc)` f32 state; ring rotation via `comm.send`/`recv` inside `group_start`/`group_end`; finalize kernel emits `y = o_acc / l_acc` (+ optional `lse`). **Tier 1 scope**: f16/bf16 (f32/f64 deferred), `head_dim == 128`, FW only (BW Tier 2), no GQA broadcast, no arbitrary additive mask. Causal masking applied on **global** indices (each step kernel takes `q_global_base` + `k_global_base` so masking is consistent across rotation steps). 12 new FFI symbols (`workspace_bytes` + dtype-independent `init_run` + 5 per-dtype × 2 dtypes: `step_run` / `step_can_implement` / `finalize_run` / `finalize_can_implement`). Unlocks **million-token context length** across N GPUs with O(N/P) memory where N = total seq len, P = ring size. Complementary to Phase 57's tensor-parallelism (sequence-dim sharding vs head-dim sharding compose). 4 smoke tests (3 single-rank degenerate cases validating against `FlashSdpaPlan` ground truth: f16 + bf16 + f16 causal — all pass on RTX 4070; 1 multi-rank scaffold `#[ignore]`-gated for 2+ GPU validation). Single-rank `world_size == 1` reduces to standard FlashAttention math (the validation path on single-GPU hardware). | done |
| 48 | Marlin + AWQ 4-bit GEMM vendor + GPTQ→Marlin repack utility (alpha.57, no version bump — consolidation phase will bump; gated behind the new `marlin` + `awq` cargo features on `baracuda-kernels-sys` and `baracuda-kernels`). **Two complementary 4-bit GEMM vendors** completing the "4-bit hub coverage" started in Phase 53 (NF4). **Marlin** (IST-DASLab, Apache-2.0 + §3 patent grant, vendored at `crates/baracuda-kernels-sys/vendor/marlin/`) — state-of-the-art W4A16 GEMM for the decode-batch regime, ~3.87× speedup over FP16 GEMM at M ∈ [1, 32] on Ampere / Ada per the paper. **Symmetric** int4 (zero-point fused into dequant as `q - 8`); group size 128 or per-channel; sm_80/86/89 only (sm_90 needs WGMMA rewrite — Marlin v2 territory, deferred). NEW `Int4MarlinGemmPlan<f16>`. **AWQ** (mit-han-lab, MIT — no patent grant, vendored at `crates/baracuda-kernels-sys/vendor/awq/`) — natively supports the **most-deployed 4-bit format on the Hugging Face Hub** (Llama / Mistral / Qwen `*-AWQ`). **Asymmetric** int4 with explicit per-group zero-points; group size 64 or 128; loads directly from HF checkpoints without repack. NEW `Int4AwqGemmPlan<f16>`. **GPTQ→Marlin repack utility** — pure-Rust host-side `gptq_to_marlin_repack` bridging GPTQ asymmetric checkpoints into Marlin's symmetric layout via zero-point absorption (trailblazer implementation; act_order=True deferred, the upstream Marlin intra-fragment permutation table is documented but uses identity permutation in the trailblazer). 4 new FFI symbols total (2 Marlin: `_run` + `_can_implement`; 4 AWQ: `_run` + `_workspace_bytes` + `_can_implement` + dequant stub). AWQ vendor source patched to strip the upstream `<torch/extension.h>` host wrapper (`__asm__ __volatile__` → `asm volatile` for MSVC nvcc portability) and re-export only the device-side `__global__` template kernel. Marlin needs `--expt-relaxed-constexpr` (constexpr `ceildiv` called from `__global__`). Both kernels build clean on RTX 4070 with the gated features. 3 smoke test files (marlin_smoke `#[ignore]` GPU + descriptor validation; awq_smoke `#[ignore]` GPU + descriptor validation; gptq_to_marlin_smoke pure-Rust roundtrip + zp-fold verification + clamp-at-extremes). | done |
| 46+ | Phase 46-51 mainstream-techniques roadmap (FlashInfer cherry-pick, Marlin/AWQ); Hopper sm_90a / Blackwell sm_100; 1.0 freeze. | pending (see [`ROADMAP.md`](ROADMAP.md)) |

API stability is **not** promised before beta.0. Breaking changes ship in
each alpha bump and are documented in the workspace `CHANGELOG.md`.

## Quick start

Add the kernel facade and the driver crate:

```toml
[dependencies]
baracuda-kernels = { version = "0.0.1-alpha.57", features = ["sm89", "cudnn"] }
baracuda-driver  = "0.0.1-alpha.57"
```

A representative example — single-axis numerically stable softmax over a
device-resident tensor:

```rust,no_run
use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    PlanPreference, SoftmaxArgs, SoftmaxDescriptor, SoftmaxKind, SoftmaxPlan,
    TensorMut, TensorRef, Workspace,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Standard CUDA bring-up via baracuda-driver.
    let ctx = Context::new(&Device::get(0)?)?;
    let stream = Stream::new(&ctx)?;

    // 2. Allocate device input + output buffers (rank-2: rows × cols).
    let rows = 32i32;
    let cols = 1024i32;
    let n_elems = (rows * cols) as usize;
    let dev_x: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_elems)?;
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_elems)?;

    // 3. Build the descriptor — pure shape + dtype + op-kind, no handles.
    let desc = SoftmaxDescriptor::<2> {
        kind: SoftmaxKind::Softmax,
        input_shape: [rows, cols],
        softmax_axis: 1,
        element: <f32 as baracuda_kernels::KernelDtype>::KIND,
    };

    // 4. Plan selection — picks a kernel SKU (bespoke softmax kernel here).
    let plan = SoftmaxPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())?;

    // 5. Args carry the per-call tensor handles + strides.
    let args = SoftmaxArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: [rows, cols], stride: [cols as i64, 1] },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: [rows, cols], stride: [cols as i64, 1] },
    };

    // 6. Launch. Workspace::None for plans that need no scratch.
    plan.run(&stream, Workspace::None, args)?;
    stream.synchronize()?;
    Ok(())
}
```

The same `select` → `run` shape applies to every op. GEMM, attention,
conv2d, FFT, scatter — the descriptor / args fields differ per family but
the lifecycle is identical. See the [`crates/baracuda-kernels`
README](crates/baracuda-kernels/README.md) for the int8-GEMM variant of
the same example.

## Workspace layout

The user-facing crates a typical caller will reach for:

```text
baracuda-kernels             # the unified Plan-based ML op facade
baracuda-kernels-types       # shared type vocabulary (Element, TensorRef, KernelSku, ...)
baracuda-kernels-sys         # raw FFI to bespoke .cu kernels
baracuda-kernels-bench       # criterion harness for sm_89 perf sweeps (not published)
baracuda-cutlass             # safe wrapper for CUTLASS GEMM (float, int8 RCR, batched, grouped)
baracuda-driver              # safe wrapper for the CUDA Driver API
baracuda-runtime             # safe wrapper for the CUDA Runtime API
```

The per-library wrappers used internally by the facade (you can also use
them stand-alone):

```text
baracuda-cublas{,-sys}       # cuBLAS + cuBLASLt + cuBLASXt
baracuda-cudnn{,-sys}        # cuDNN classic + Graph API
baracuda-cufft{,-sys}        # cuFFT
baracuda-cusolver{,-sys}     # cuSOLVER dense + sparse + Rf + Mg
baracuda-cusparse{,-sys}     # cuSPARSE
baracuda-curand{,-sys}       # cuRAND
baracuda-cutensor{,-sys}     # cuTENSOR
baracuda-npp{,-sys}          # NPP
baracuda-nccl{,-sys}         # NCCL
baracuda-cvcuda{,-sys}       # CV-CUDA
baracuda-nvjpeg{,-sys}       # nvJPEG
baracuda-nvcomp{,-sys}       # nvCOMP
```

And the supporting low-level crates (FFI, build infrastructure, profiling):

```text
baracuda-cuda-sys            # Driver + Runtime FFI
baracuda-nvrtc{,-sys}        # runtime CUDA C++ → PTX
baracuda-nvjitlink{,-sys}    # CUDA 12+ JIT linker
baracuda-cupti{,-sys}        # profiling APIs
baracuda-nvml{,-sys}         # device monitoring
baracuda-cufile{,-sys}       # GPUDirect Storage (Linux-only)
baracuda-tensorrt{,-sys}     # TensorRT inference runtime
baracuda-forge              # build-time .cu → PTX compiler driver
baracuda-build              # build.rs helpers
baracuda-core                # loader + Error plumbing
baracuda-types{,-derive}     # pure-data types: Half, BFloat16, Complex, DeviceRepr
```

The full umbrella crate (`baracuda`) re-exports everything behind cargo
features — convenient when you want everything; overkill when you don't.

## Hardware support

baracuda targets **Ampere and newer** by design. Pre-Ampere GPUs lack the
tensor-core instructions and async-copy primitives the bespoke kernels are
written against (`mma.sync.m16n8k*`, `cp.async`, `ldmatrix`), and we have
no desire to ship a slower SIMT fallback for hardware that's eight years
old.

| Compute capability | NVIDIA marketing names | baracuda support |
| --- | --- | --- |
| sm_80 | Ampere (A100, A40, A30, RTX 30xx) | **default baseline** |
| sm_89 | Ada Lovelace (RTX 40xx, L40, L4) | feature-gated specialized kernels (FP8, larger Flash Attention tiles) |
| sm_90a | Hopper async (H100, H200) | stubs in place; full specialization pending Phase 11 |
| sm_100 | Blackwell | post-Phase-11 |
| ≤ sm_75 (Turing, Volta, Pascal, …) | — | **unsupported** |

The default `sm80` build runs forward-compatibly on Ada and Hopper through
JIT-compiled PTX; turn on `sm89` to pick up the FP8 and Flash-Attention
sibling plans tuned for Ada's larger register file.

## Cargo features

The kernel facade exposes a small feature set:

| Feature | Default | Effect |
| --- | --- | --- |
| `sm80` | yes | Build the Ampere-baseline kernel set. |
| `sm89` | no | Build the Ada Lovelace specializations (FP8 GEMM, `FlashSdpaSm89Plan`). |
| `sm90a` | no | Build the Hopper-specialized kernels (stubs today). |
| `cudnn` | no | Link cuDNN and enable conv / pool / `CtcLossCudnnPlan`. |

`cudnn` is off by default because cuDNN is a separate NVIDIA download not
bundled with the stock CUDA toolkit installer. Enabling it without cuDNN
installed produces a linker error on `cudnn.lib` / `libcudnn.so` — see
the building section for the auto-discovery paths the build script probes.

## Building

Requirements:

- **CUDA Toolkit ≥ 12.0** with `nvcc` on `PATH`. baracuda is tested on
  12.x and 13.x.
- **cuDNN 9.x** (only if you enable the `cudnn` feature) — separate
  NVIDIA download, not bundled with the toolkit.
- **A working Rust toolchain ≥ 1.85** (workspace MSRV pinned in
  `rust-toolchain.toml`).
- **Windows users**: `lld-link.exe` somewhere on `PATH`. The CUDA `nvcc`
  invocation links through it; the install location is typically
  `C:\Program Files\LLVM\bin`. Install the LLVM Windows package and add
  that directory to `PATH` if `cargo build` complains about
  `lld-link.exe` not being found.

A typical full build with all GPU-side features (CUDA toolkit + cuDNN
present):

```bash
cargo build -p baracuda-kernels --features sm89,cudnn --release
```

Or, to verify the public API surface compiles without the full kernel
build (fast — type-check only):

```bash
cargo check -p baracuda-kernels --features sm89,cudnn
```

The `baracuda-kernels-sys` build script auto-discovers cuDNN at the
following paths in order: `CUDNN_PATH` / `CUDNN_ROOT` / `CUDNN_HOME` env
vars, then `C:\Program Files\NVIDIA\CUDNN\v<X.Y>\` on Windows, then the
CUDA toolkit's own `lib/` directory (pre-cuDNN-9 layout), then the
standard Linux distro paths under `/usr/lib/`.

## Troubleshooting

### Windows: Git-for-Windows fake `link.exe` shadowing the MSVC linker

Git-for-Windows ships a GNU coreutils binary named `link.exe` at
`C:\Program Files\Git\usr\bin\link.exe` — its job is to create a hard
link, **not** to link object files. If that directory appears on `PATH`
ahead of the MSVC linker (or LLVM's `lld-link.exe`), `cargo build`
invokes the coreutils binary instead of the real linker and fails with a
cryptic error (it doesn't understand `/OUT:` and friends).

baracuda's `baracuda-kernels-sys` and `baracuda-cutlass-sys` build
scripts probe `PATH` on Windows and emit a `cargo:warning` if they
detect this shadowing. **Fix:** re-order `PATH` so the MSVC linker
(typically reached via the Visual Studio "x64 Native Tools Command
Prompt") or LLVM's `lld-link.exe` (`C:\Program Files\LLVM\bin\`) appears
before `C:\Program Files\Git\usr\bin\`. Building from the VS x64 Native
Tools prompt is the most reliable option; alternatively, install LLVM
and put its `bin` directory ahead of Git's on the user/system `PATH`.

## Testing

baracuda's GPU integration tests are gated behind `#[ignore]` so a
host-only `cargo test` doesn't try to launch a kernel on a machine
without an NVIDIA driver. To run them you need a working GPU plus the
`--ignored` flag:

```bash
# Host-only tests (compile + reference logic; no GPU access):
cargo test -p baracuda-kernels --lib

# Full GPU integration sweep — RTX 30xx / 40xx / 50xx required:
cargo test -p baracuda-kernels --release -- --ignored

# Verify the workspace-level API surface compiles (no GPU needed):
cargo check -p baracuda-kernels --features sm89,cudnn
```

The full regression on an RTX 4070 covers 324 binary targets at
~1630 tests passing. Individual op-family suites take 30–90 seconds;
the full sweep is 25–40 minutes.

## Benchmarks

The `baracuda-kernels-bench` crate is a criterion-based harness with
CUDA-event-timed throughput sweeps across GEMM, Flash Attention, and
Conv2d at LLM-typical and ResNet-typical shapes. It is **not** published
to crates.io (it depends on a working GPU).

```bash
cargo bench -p baracuda-kernels-bench --features sm89,cudnn
```

The full sweep takes ~30 minutes on an RTX 4070. Scope to a single family
with `--bench gemm` / `--bench flash_attention` / `--bench conv2d`. See
[`crates/baracuda-kernels-bench/BENCH-sm89.md`](crates/baracuda-kernels-bench/BENCH-sm89.md)
for the baseline table format and methodology.

## Project documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — layered design, Plan-Descriptor-Args
  pattern, `KernelSku` taxonomy, dispatcher design, workspace contract,
  sibling-plan pattern, vendoring convention, phase roadmap.
- `OP-MATRIX.md` — full op × dtype × backend coverage matrix (planned).
- `LESSONS.md` — postmortems, ABI footguns, performance traps (planned).
- Per-crate `README.md` files under `crates/<name>/`.

## License

Dual-licensed under [MIT](LICENSE-MIT) **or** [Apache-2.0](LICENSE-APACHE).
Pick whichever fits your project. Contributions accepted under the same
terms.

NVIDIA's CUDA libraries (`libcuda`, `libcudart`, `libcublas`, `libcudnn`,
…) are **not** redistributed by this project. You obtain them from NVIDIA
separately — either through the CUDA Toolkit installer or through each
library's dedicated download page. baracuda's loader opens whatever the
host driver / toolkit has installed.

## Vendor attribution

A small number of bespoke kernels in `baracuda-kernels-sys` are vendored
from upstream open-source projects (huggingface/candle's CUDA kernel set
via `fuel-cuda-kernels`; llama.cpp's `ggml-cuda` GGUF block-format
quantization + MMVQ; `guoqingbao/attention.rs`'s fused MoE expert
kernels). Each adapted source carries an `SPDX-FileCopyrightText:` +
`SPDX-License-Identifier:` header; the consolidated provenance is in
[`crates/baracuda-kernels-sys/LICENSE-thirdparty.md`](crates/baracuda-kernels-sys/LICENSE-thirdparty.md).

[**FlashAttention v2**](https://github.com/Dao-AILab/flash-attention)
(Tri Dao, BSD-3-Clause, pinned at `v2.8.3` /
`060c9188beec3a8b62b33a3bfa6d5d2d44975fab`) is vendored at
[`crates/baracuda-kernels-sys/vendor/flash-attention/`](crates/baracuda-kernels-sys/vendor/flash-attention/)
with verbatim `LICENSE` + `AUTHORS` files and full vendor / scope notes
in [`VENDOR.md`](crates/baracuda-kernels-sys/vendor/flash-attention/VENDOR.md).
Gated behind the `fa2` cargo feature on `baracuda-kernels-sys` and
`baracuda-kernels`; exposed through a backend-choice path on
`FlashSdpaPlan` (Phase 42).

[**mHC.cu**](https://github.com/AndreSlavescu/mHC.cu) (Andre Slavescu,
MIT, pinned at `a426939c2dbc11c443db041bcff12b65d1b6482a`) — unofficial
CUDA implementation of DeepSeek-AI's
[*Manifold-Constrained Hyper-Connections*](https://arxiv.org/abs/2512.24880)
paper — is vendored at
[`crates/baracuda-kernels-sys/vendor/mhc/`](crates/baracuda-kernels-sys/vendor/mhc/)
with the verbatim upstream `LICENSE`, an `AUTHORS` file, and full
vendor / scope notes in
[`VENDOR.md`](crates/baracuda-kernels-sys/vendor/mhc/VENDOR.md). Gated
behind the `mhc` cargo feature on `baracuda-kernels-sys` and
`baracuda-kernels`; exposed through the new `HyperConnectionPlan`
(Phase 43, Tier 1: static-H FW, bf16 weights / f32 activations).

[**FlashInfer**](https://github.com/flashinfer-ai/flashinfer) (NVIDIA
+ FlashInfer community, Apache-2.0 with full patent grant, pinned at
`v0.6.12` / `eee0d75f91f64c520bfaed07e39a850ea4ddde23`) — a curated
~12 kLOC subset of the FlashInfer header tree is vendored at
[`crates/baracuda-kernels-sys/vendor/flashinfer/`](crates/baracuda-kernels-sys/vendor/flashinfer/)
with verbatim upstream `LICENSE` + `NOTICE` and full vendor / scope /
patch notes in
[`VENDOR.md`](crates/baracuda-kernels-sys/vendor/flashinfer/VENDOR.md).
Gated behind the `flashinfer` cargo feature on `baracuda-kernels-sys`
and `baracuda-kernels`; exposes three NEW plan families — paged-KV
decode + append (`BatchPagedDecodePlan` + `PagedKvAppendPlan` for
vLLM-style serving), sort-free sampling (`TopKTopPSamplingPlan` —
combined TopK/TopP/MinP via a single-kernel rejection sampler), and
cascade attention LSE merge (`CascadeAttentionPlan` for prefix-cache
sharing). Surgical cherry-pick (not a wholesale wrap) — Hopper /
Blackwell / NVSHMEM / Mamba / MLA / POD paths intentionally skipped
to keep the build cost contained. Phase 46.

The [`baracuda-forge`](crates/baracuda-forge) build-time kernel-compiler
crate is a vendored fork of [`cudaforge`](https://github.com/guoqingbao/cudaforge)
by **Guoqing Bao** — see [`crates/baracuda-forge/NOTICE`](crates/baracuda-forge/NOTICE)
for the upstream commit hash.

The [`baracuda-cutlass`](crates/baracuda-cutlass) safe wrapper for NVIDIA
CUTLASS — plan-based GEMM and grouped-GEMM with caller-supplied
workspace, MoE-friendly variable-M-per-group dispatch — was specified
by the **Fuel ML library team**. See
[`crates/baracuda-cutlass/NOTICE`](crates/baracuda-cutlass/NOTICE) for
the design lineage.

[`baracuda-kernels`]: crates/baracuda-kernels
[`baracuda-kernels-sys`]: crates/baracuda-kernels-sys
