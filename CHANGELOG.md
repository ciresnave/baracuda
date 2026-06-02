# Changelog

All notable changes to baracuda are documented here. The project follows
[Semantic Versioning](https://semver.org/), though the `0.0.1-alpha.X`
series uses the alpha increment instead of patch/minor — every published
alpha represents one or more completed phases.

The phase numbering is Fuel-driven (Fuel is baracuda's primary downstream
consumer); see `ROADMAP.md` for the active phase board.

## Unreleased (queued for next alpha)

### Added

- **Phase 64 — In-place aliasing contract docs (extended)**: documented
  `x_ptr == y_ptr` safety for Cast / Where / Triu / Tril / Fill /
  Activation BW; explicitly marked Flip / Roll / Permute / RoPE as **not**
  in-place safe (write-before-read by construction).
- **Phase 65a — Reusable SMEM kernel helpers**: new
  `baracuda_smem_row_stager.cuh` (`smem_stage_row`, strided variants,
  `smem_budget_for_arch`) and `baracuda_smem_reduce.cuh`
  (`warp_reduce_{sum,max,min}_f32`, `block_reduce_{sum,max,min}_f32`,
  cross-warp aggregation via `BARACUDA_MAX_WARPS = 32`). Lifts the SMEM
  staging pattern out of `baracuda_moe.cuh` into a reusable kernel-helper
  library at `crates/baracuda-kernels-sys/kernels/include/`.
- **Phase 65b — RMSNorm SMEM staging + in-place**: SMEM-staged
  `rms_norm_smem_kernel<T>` enables `y_ptr == x_ptr` aliasing for
  f32/f16/bf16 (f64 stays on legacy multi-pass-global). 3 in-place
  proof tests bit-exact vs non-aliased reference.
- **Phase 65c — LayerNorm / Softmax / LogSoftmax SMEM staging +
  in-place**: same pattern, three new `*_smem_kernel<T>` + dispatcher
  pairs. 9 in-place proof tests. f64 stays on legacy (no
  `block_reduce_sum_f64` yet — ~1 day of work if asked).
- **Phase 65d — BN / GN / IN in-place contract** (no kernel changes):
  documented the existing two-stage kernel (stage-1 stats-only,
  stage-2 single-read-then-write per cell) as in-place safe by
  construction. **f64 IS in-place safe here**, unlike Phase 65b/c
  normalizers. 12 in-place proof tests across f32/f16/bf16/f64.

### Documentation

- Greatly expanded README cargo-features table (4 features → 18
  documented, organized by category).
- Brought `ARCHITECTURE.md` and `OP-MATRIX.md` current; removed
  false-deferred claims for ops shipped in Phases 16/17/18/25/26.
- Added per-crate README pointers for the 8 crates that previously
  shipped to crates.io with blank README tabs.

### Fixed

- Closed 95 `clippy -D warnings` failures (CI gate); the warnings were
  stylistic (unnecessary parens, unused imports, mutable not needed) plus
  10 GGUF-K-block test names allowed via `#[allow(non_snake_case)]`.

---

## 0.0.1-alpha.63 — 2026-05-30 (Phase 63)

### Added

- **FA2 saved-tensor wiring for downstream autograd**: new FFI symbol
  `baracuda_kernels_fa2_sdpa_lse_size` (dense sibling of the varlen
  helper from Phase 59b) + `# LSE saved-tensor contract` sections on
  FW + BW trailblazers. NEW docs guide
  `docs/guides/fa2-saved-tensor-contract.md`.
- 12 new tests (3 host `_lse_size` sanity + 5 FW→BW roundtrip + 4 BW
  feature backfill).

### Notes

- LSE has been a documented FW output since alpha.56 (Phase 42); Phase
  63 just added the size helper and wiring clarity that downstream
  autograd integrators were asking for.

## 0.0.1-alpha.62 — 2026-05-30 (Phase 62)

### Added

- **Strided in-place op support**: lifts the in-place aliasing contract
  from contig-only (Phase 61) to strided. 11 new `affine_inplace` FFI
  symbols (4 contig int backfill + 7 strided fp/int). Same-pointer
  aliasing on strided trailblazers documented as a stable public
  contract under `stride_x == stride_y`.
- NEW `baracuda_kernels_types::strides_equal` host helper.
- 39 new tests (14 host-only + 17 GPU smoke + 8 aliasing-contract proof);
  2229+/0 regression on RTX 4070.

## 0.0.1-alpha.61 — 2026-05-30 (Phase 61)

### Added

- **In-place op family completion (contig)**: 2 new `bf16` / `f16`
  `affine_inplace` FFI symbols with f32 scalars. Same-pointer aliasing
  safety documented as a stable public contract on unary / binary /
  ternary trailblazers (covers ~30 unary + 20 binary + parameterized
  unary launchers across all dtypes).
- Unblocks Fuel's planned in-place op fanout (16+ unary + 4 binary +
  Clamp + PowI families) with zero new symbols per family.

## 0.0.1-alpha.60 — 2026-05-29 (Phase 60)

### Added

- **FA2 head_dim {160, 224, 512} expansion via EricLBuehler/candle
  fork**: 2184/0 regression. Closes the FA2 FW head_dim gap relative to
  upstream FlashAttention v2.8.3 + GQA / ALiBi / sliding / softcap.

## 0.0.1-alpha.59 — 2026-05-28 (Phase 59)

### Added

- **Phase 59a**: FA2 FW expansion to full v2.8.3 head_dim set + GQA /
  ALiBi / sliding / softcap.
- **Phase 59b**: FA2 BW + varlen; closes Fuel's FA2-retirement
  requirements.
- **Phase 59c**: Flash SMEM race fix; regression re-validation.

## 0.0.1-alpha.58 → 0.0.1-alpha.45

Phase 30–58 release-by-release detail is preserved in `ROADMAP.md`'s
phase log. Major milestones in this band:

- **Phase 57** — Megatron-LM TP primitives (composition-only).
- **Phase 56** — Ring Attention (Apache-2.0 reference).
- **Phase 55** — TransformerEngine FP8 cast + delayed-scaling recipe.
- **Phase 54** — xFormers block-sparse SDPA + 2:4 structured sparsity.
- **Phase 53** — bitsandbytes NF4 dequant + GEMV.
- **Phase 50–52** — Mamba-2 SSD + causal-conv1d; vendor cleanup.
- **Phase 49** — NVIDIA Apex multi-tensor optimizer kernels.
- **Phase 48** — Marlin + AWQ W4A16 GEMM (both, side by side).
- **Phase 46** — FlashInfer Tier-1 (paged decode, cascade, sampling).
- **Phase 44/44b** — ozIMMU Ozaki DGEMM (with Windows port).
- **Phase 43** — DeepSeek-AI mHC.cu HyperConnection.
- **Phase 42** — Dao-AILab FlashAttention v2 Tier-1.
- **Phase 30** — `GemmPlan` gains cuBLAS-backed dispatch (3× speedup
  at M=32 f16 decode batch).

## Pre-Phase-30

See `ROADMAP.md` "Historical phase log" for the alpha.1 → alpha.45 band
covering the original kernel matrix build-out (Phases 1–29).

---

## Conventions

- Each phase is a self-contained scope completed before publish (no
  partial phases per release). Unfinished phases accumulate in
  "Unreleased" above until the next published alpha rolls them up.
- **Memory entries** live under
  `~/.claude/projects/c--Users-cires-OneDrive-Documents-projects-baracuda/memory/`
  with one `project_phase<N>_complete.md` per phase — they carry
  load-bearing context (gotchas, surprises) that doesn't fit a
  user-facing changelog.
- **Feature gates** are called out per release — see `README.md`'s
  cargo-features table for what each feature enables.
