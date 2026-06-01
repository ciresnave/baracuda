# Session prompt — Add `baracuda-flashinfer` wrapper for FlashInfer

Working on baracuda at `c:\Users\cires\OneDrive\Documents\projects\baracuda`.
This is the single highest-ROI CUDA-X library addition per the Phase 65
audit. Other library-addition sessions may be running in parallel.

## Context

[FlashInfer](https://github.com/flashinfer-ai/flashinfer) is NVIDIA's
inference-focused attention/MoE/sampling kernel library (Apache-2.0).
Key features:

- **Paged attention** with arbitrary block sizes — the canonical
  inference-serving abstraction for KV-cache management
- **Continuous batching** kernels (prefill + decode with arbitrary
  per-sequence lengths)
- **Speculative decoding** kernels (verify multiple draft tokens in
  parallel against the target model)
- **Sampling kernels** — top-k, top-p, top-p+top-k composed, multinomial, repetition penalty, frequency penalty
- **Cascade attention** for serving multiple LoRA adapters

baracuda already has:
- Bespoke Flash SDPA (FW + BW)
- Phase 42 Tri-Dao FlashAttention v2 vendor (Tier-1)
- Phase 46 partial FlashInfer cherry-pick — only the sampling kernels +
  paged-KV append + cascade-attention launchers. Paged decode was
  staged but had build issues; needs revisit.

baracuda does NOT yet have:
- A complete `baracuda-flashinfer` safe wrapper crate over the FlashInfer surface
- The paged attention forward/backward as a usable `BatchPagedDecodePlan` / `BatchPrefillPlan`

This session's job: complete the FlashInfer integration that Phase 46
started.

## Scope

**Crates to create or extend:**

1. `crates/baracuda-flashinfer-sys/` — if it doesn't exist, create it.
   `extern "C"` FFI declarations for FlashInfer's C-ABI surface. If the
   surface is C++-only (template-heavy), write thin C-ABI launcher
   shims in `kernels/` and expose those.

2. `crates/baracuda-flashinfer/` — if it doesn't exist, create it.
   Safe Rust wrapper modeled on `baracuda-cublas` / `baracuda-cudnn`
   patterns (typed Plans, error handling, workspace management).

3. The vendored FlashInfer source already lives at
   `crates/baracuda-kernels-sys/vendor/flashinfer/`. Reuse it; don't
   re-vendor.

## Phase 46 status — read this carefully

Phase 46 cherry-picked specific FlashInfer kernels into baracuda's
existing FA family. Look at `crates/baracuda-kernels/src/attention/`
for the `BatchPagedDecodePlan` / `CascadeAttentionPlan` / etc. —
these are partial. Goal of this session: complete them into a
usable production surface.

The Phase 46 carry-forward in Consolidation C (alpha.58) noted that
the paged-decode launcher had a Win64 LLP64 type mismatch
(`std::max(unsigned long, size_t)`) that was patched. Verify that
patch is still in place; the underlying paged_decode kernel should
now build.

## Tier 1 deliverables (this session)

1. `BatchPagedDecodePlan<T>` — fully wire up paged-attention decode
   (single token per sequence, batched across many sequences with KV
   cache stored in fixed-size pages). f16 + bf16. ~head_dim 128 priority,
   add 64 + 256 if cheap.
2. `BatchPagedPrefillPlan<T>` — prefill path (multiple tokens per
   sequence, attention over the KV cache + the new tokens).
3. Sampling kernel safe wrappers: `TopKTopPSamplingPlan`, `MinPSamplingPlan`, repetition-penalty / frequency-penalty kernel.
4. Smoke tests for each plan exercising small fixtures. Mark `#[ignore]` per baracuda convention.

## Tier 2 deferrable (next session)

- Speculative-decode kernels (multi-token verification)
- Multi-LoRA cascade attention safe wrapper (Phase 46 has the FFI; needs the Plan layer)
- bf16 quantization paths (FP8 KV cache)
- Backward kernels (FlashInfer is primarily FW-only for inference; BW lives in training-side FA2)

## Reference patterns

Look at:

- `crates/baracuda-cublas/` for the typed-Plan + workspace pattern
- `crates/baracuda-kernels/src/attention/flash_sdpa.rs` for an existing attention Plan with multiple backends — your new Plans can integrate as a `BackendKind::FlashInfer` arm on a unified `PagedAttentionPlan` if desired, or stand alone.
- Phase 46 commit + memory file `project_phase46_complete.md` for context on what was already done

## Cargo feature gating

Use a `flashinfer` cargo feature (already exists per Phase 46). New code should compile only when that feature is enabled. Off by default.

## Linking

FlashInfer is header-only / template-heavy. No `find_library` needed; the existing build.rs path for the vendored sources should work. Verify.

## Out of scope

- Don't try to integrate FlashInfer's Triton or PyTorch frontends.
  baracuda's surface is C-ABI; ignore the Python bindings.
- Don't change the existing Phase 46 cherry-picks — extend, don't
  rewrite.
- Don't version bump or publish. Accumulating for next release.

## Coordination

- Working directory: `c:\Users\cires\OneDrive\Documents\projects\baracuda`
- Branch: `phase66-flashinfer`
- No version bump, no publish.
- Commit on branch + push + stop. Eric will review parallel branches and merge in order.

## Stop conditions

- If the paged-decode build issue from Phase 46 isn't resolved (still
  `std::max` type mismatch or similar) and the fix is non-trivial:
  stop, report, ask Eric.
- If FlashInfer's API has changed since baracuda's vendored version
  and a major upstream sync is needed: stop, report, don't do the sync
  in this session (too much scope).
- If you find the entire `baracuda-flashinfer{,-sys}` crate pair
  already shipped (someone got here first): stop, report.

## Memory + memory file

After completion, write a `project_phase66_complete.md` memory file
summarizing what was added + any deferrals to Tier 2 (per baracuda's
memory convention).
