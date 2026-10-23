# Changelog

All notable changes to baracuda are documented here. The project follows
[Semantic Versioning](https://semver.org/), though the `0.0.1-alpha.X`
series uses the alpha increment instead of patch/minor — every published
alpha represents one or more completed phases.

The phase numbering is Fuel-driven (Fuel is baracuda's primary downstream
consumer); see `ROADMAP.md` for the active phase board.

**Versioning exception (from alpha.81):** a crate whose version answers "which
version of the other project does this work with?" does not share the
workspace version — its MAJOR.MINOR follow the `unpopped` release it builds
against, and its PATCH is baracuda's own counter. `baracuda-cuda-emit` has been
on this line since alpha.81 (`baracuda-cuda-emit 0.11.0` ships with
`0.0.1-alpha.81` and goes with `unpopped 0.11`); `baracuda-cuda-parse` joined
at alpha.82.

## 0.0.1-alpha.151 - 2026-10-23 (cuda-seam: registry fuel-kernel-seam 0.22.4, structural region ids)

Workspace alpha.151 (PM to confirm at gate). `baracuda-cuda-emit` stays 0.20.2. Non-breaking.

- **Change (#117):** `baracuda-cuda-seam` now depends on `fuel-kernel-seam` and `fuel-kernel-seam-types` `0.22.4` from crates.io instead of a git rev, and `region_op_id` hashes the region structure (`OpTag`, `OpAttrs::region_key()`, children, in an exhaustive `PatternNode` walk) instead of its `Debug` rendering. Kernel ids change once; scalar values no longer reach the id, so two regions differing only in a scalar share one kernel id (#115). The crate stays `publish = false`.
- **Test:** `a_scalar_value_changes_the_region_id` is inverted to `a_scalar_value_does_not_change_the_region_id`; the per-field injectivity test drops its `scalars` row.

## 0.0.1-alpha.150 - 2026-10-22 (sys/types: drop false #![no_std])

Workspace alpha.150 (PM-allocated). `baracuda-cuda-emit` stays 0.20.2. Non-breaking.

- **Fix (#143, #144):** removed `#![no_std]` from `baracuda-ozimmu-sys`, `baracuda-transformer-engine-sys`, `baracuda-flashinfer` and `baracuda-types`. Each depends on a std crate (`baracuda-cuda-sys`/libloading, `baracuda-kernels`, `unpopped-vocab`), so the attribute never held on a no_std-only target. No code used anything std-only; `cutlass-sys`, `cutlass-kernels-sys`, `kernels-sys` and `flashinfer-sys` have no std deps and keep it.

## 0.0.1-alpha.149 - 2026-10-22 (docs: device lane is not running)

Workspace alpha.149 (proposed; PM allocates at gate). `baracuda-cuda-emit` stays 0.20.2. Docs only; no crate source changed, non-breaking.

- **Docs:** `docs/DEVICE-TESTS.md` states that no self-hosted runner exists for this public repo, every device job is skipped, and nothing in the device lane has run since the runner went away; local `gpu-run` results are LOCAL-ONLY.

## 0.0.1-alpha.148 - 2026-10-22 (CI: feature-gated kernel device tests)

Workspace alpha.148 (proposed; PM allocates at gate). `baracuda-cuda-emit` stays 0.20.2. CI only; no crate source changed, non-breaking.

- **CI:** new `device-tests-kernel-features.yml` runs the feature-gated `baracuda-kernels` device tests that previously compiled empty (10 feature sets, 55 test binaries, 257 tests by static census). Its first `workflow_dispatch` result is not yet verified.

## 0.0.1-alpha.147 - 2026-10-22 (AWQ padded-layout tests; Marlin repack advisory)

Workspace alpha.147 (proposed; PM allocates at gate). `baracuda-cuda-emit` stays 0.20.2. Tests and docs only; no API change.

- **Advisory (Marlin):** `gptq_to_marlin_repack` produced a buffer that is NOT in Marlin's layout in every release from `0.0.1-alpha.64` (Phase 48, which introduced it) through `0.0.1-alpha.145`; it was fixed in alpha.146 (#218). Any weights repacked by those versions and fed to `Int4MarlinGemmPlan` give wrong results (measured: ~75% of output cells wrong at M=1, N=K=4096). Repack them again with alpha.146 or later. No crash or error signalled it. Whether any shipped consumer called it is not known to us.
- **Tests:** `awq_reference` now also covers group size 64, `IC/group_size % 8 != 0` (the kernel's padded `zeros`/`scales` rows) and `M > 128`, each with the corrupted-scale control. 7 of 7 pass on the RTX 4070 (sm_89).
- **Docs:** `Int4AwqGemmArgs` documents the padded OC-major layout formula.

## 0.0.1-alpha.146 - 2026-10-22 (BREAKING: Marlin packer is real; Marlin/AWQ tested numerically)

Workspace alpha.146 (proposed; PM allocates at gate). `baracuda-cuda-emit` stays 0.20.2.

- **Breaking:** `repack` now returns `Err` for `OC % 64 != 0` (it used to return a wrong layout), and the public const `MARLIN_PERM_LEN` changes from 64 to 1024. Weights repacked by any earlier version are NOT in Marlin's layout and must be repacked.
- **Fix:** `gptq_to_marlin_repack` was a stub. It packed nibbles row-major with an identity permutation and an identity scale permutation, so its output was not Marlin's layout. It now applies upstream's `_perm` / `_scale_perm` reshuffle (`Layer.pack` in `marlin/__init__.py`). Measured on the RTX 4070 (sm_89): with the old packer the new reference test mismatches 3100 of 4096 cells at M=1, N=K=4096; with the new one all three shapes match in every cell. `repack` now also rejects `OC % 64 != 0`. `MARLIN_PERM_LEN` is now 1024 (was a placeholder 64).
- **Tests:** `tests/marlin_reference.rs` and `tests/awq_reference.rs` compare kernel output with a dequantise-then-GEMM reference built from the unpacked weights (M=1/16, N=K=4096 and 1024, group 128; AWQ with random zero-points), plus a corrupted-scale control per shape. The old finite/zero-weight smokes are removed. Before this, `marlin`/`awq` were off in every CI and device-runner build, so these tests compiled to empty binaries.
- **CI:** the self-hosted `device-tests.yml` now runs the Marlin/AWQ tests with `--features marlin,awq,device-tests`; `ci.yml` type-checks them on every PR.
- **Docs:** `Int4AwqGemmArgs` now states the OC-major memory layout the kernel reads for `scales` / `zeros` (the `[IC/group_size, OC]` shape label does not describe it). API unchanged.

## 0.0.1-alpha.145 - 2026-10-21 (cuda-emit: top-1 with indices no longer panics; unpopped 0.20.13)

Workspace alpha.145 (proposed; PM allocates at gate). `baracuda-cuda-emit` 0.20.1 -> 0.20.2 (its own PATCH counter; builds against unpopped 0.20.13).

- **Fix:** `baracuda-cuda-emit` panicked generating a top-1 with indices (`row_sort_indices` / `row_topk`, outputs `[R,1]`) at the RowSort `Both` backstop (`oi.bcast.is_empty()` on out_idx, two sites). `OperandKey.bcast` is the FRAME mask: a `[R,1]` output against the `[R,K]` frame sets its bit while `contig` stays `Contig`. The check now uses `contig == Contig && !flipped` (as the Im2Col output check already did). Red-first test `topk_both_k_out_1_generates_for_the_size_1_out_axis`; the generated top-1 kernel was also run on an RTX 4070 against a host argmax (64 rows, 0 mismatches, tie resolves to the lower index).
- **Deps:** `unpopped`, `unpopped-vocab`, `unpopped-cpu-c`, `unpopped-slang` 0.20.1 -> 0.20.13 (the repro needs it: on 0.20.1 unpopped's own `plan.rs` check rejects the shape first).

## 0.0.1-alpha.144 - 2026-10-20 (ROADMAP: bench run-to-run spread finding; quiet-box suite refresh row)

Workspace alpha.144 (proposed; PM allocates at gate).

- ROADMAP only: (1) the cholesky row now includes four runs of the real `benches/linalg.rs` (cholesky N256 372-516 us, N512 784-1153 us; `lu` N512 1009-1024 us, 1.2x faster than PyTorch): the harness does not reproduce the alpha.141 rollup's 1.65 ms, so that run was a uniformly slow session; cholesky is at parity at N256 and ~1.4-2x slower at N512; (2) new row "Bench run-to-run spread on this box": 1.3-1.4x within a session, 3x vs the rollup run, added CPU load and GPU clocks did not explain it, cause NOT found, idle-vs-loaded comparison not done; (3) new row "Whole-suite benchmark refresh on a QUIET box" with its preconditions and procedure.

## 0.0.1-alpha.143 - 2026-10-20 (Linalg cholesky: first rollup value retracted; potrf_direct micro-bench)

Workspace alpha.143 (proposed; PM allocates at gate).

- **Correction:** the alpha.141 rollup's `cholesky` row (f32 N256 1.65 ms, N512 3.46 ms; 0.26x / 0.17x of PyTorch) and the alpha.142 ROADMAP row built on it ("4-6x slower") were one noisy observation and are retracted: the same plan on the same box measures N256 540 us / N512 1.13 ms (0.78x / 0.51x). New `benches/potrf_direct.rs` (publish=false bench crate, now depends on baracuda-kernels-sys) times `CholeskyPlan::run` against a direct `cusolverDnSpotrf`, per-call events vs pipelined wall-clock: plan vs direct <= 7%, discipline <= 5%, the op is host-launch-bound; the remaining 1.2-2x vs the frozen PyTorch number is unexplained (torch is not installed on the box, so PyTorch was not re-run in session). ROADMAP row rewritten; a correction note added above the generated rollup in BENCHMARKS.md. Run was NOT quiet (Balanced plan, GPU at 1605/3105 MHz, other lanes' cargo processes).

## 0.0.1-alpha.142 - 2026-10-20 (ROADMAP: Linalg cholesky perf row (measured, uninvestigated))

Workspace alpha.142 (proposed; PM allocates at gate).

- ROADMAP: new perf row for `CholeskyPlan` at 0.17-0.26x of PyTorch (alpha.141 rollup, noisy box, single observation per cell), with the evidence, the min-vs-median check, the three untested hypotheses and the next step. Docs only.

## 0.0.1-alpha.141 - 2026-10-20 (bench rollup regenerated at alpha.140; Linalg (cholesky, lu) added)

Workspace alpha.141 (proposed; PM allocates at gate).

- Regenerated the auto-generated `BENCHMARKS.md` rollup from a fresh whole-suite run (20 CSV-emitting benches, `--quick`, `--features sm89,cudnn`, one continuous `gpu-run` hold): 456 cells, 68 op families (was 452 / 66), 0 families dropped, 4 new cells = Linalg `cholesky` and `lu` at f32 N256 / N512 vs the frozen PyTorch baseline (baracuda 0.26x / 0.17x / 0.70x / 1.01x of PyTorch: `PyTorch_ns / baracuda_ns`). `hyper_connection` emits no rows (needs the `mhc` feature). NOT a quiet run: Balanced power plan, other lanes' cargo processes running, laptop GPU at 1605 of 3105 MHz SM clock; run conditions are in the PR. Across the 452 cells common with the previous table the baracuda column moved by a median 1.275x slower, but the cuBLAS reference cells in the same run moved 1.77x and the cuDNN ones 1.23x slower, so box state, not a kernel change, explains the shift; do not read cross-version deltas from this table. ROADMAP's benchmark row updated.

## 0.0.1-alpha.140 - 2026-10-19 (BREAKING: MatrixRef/MatrixMut/VectorRef fields are private; validated constructors)

Workspace alpha.140 (proposed; PM allocates at gate).

- **BREAKING (API):** `MatrixRef`, `MatrixMut` and `VectorRef` (baracuda-kernels-types, re-exported by baracuda-cutlass and baracuda-kernels) no longer have public fields. Build with `MatrixRef::new(data, rows, cols, ld)` / `MatrixMut::new(..)` / `VectorRef::new(data, len, stride)`, which return `Err(ViewError)` for a negative dimension / ld / stride or a slice shorter than the layout-independent floor (`rows * cols` logical elements, counted in slots for packed S4/U4/Bin; `(len-1)*stride + 1` for a vector). The plan that consumes the matrix still checks the exact footprint of its layout (row- or column-major, byte-counted for packed operands). Read back with `.data()` / `.rows()` / `.cols()` / `.ld()` (`VectorRef`: `.data()` / `.len()` / `.stride()`; `MatrixMut` also `.data_mut()`). `unsafe ::new_unchecked` exists for callers that have proved their own footprint. A struct literal no longer compiles. Migration: `MatrixRef { data, rows, cols, ld }` -> `MatrixRef::new(data, rows, cols, ld)?`; `m.ld` -> `m.ld()`. Fuel (the only known downstream) pins baracuda-kernels-types alpha.116 and has 3 literals in fuel-cuda-backend/src/cutlass.rs to migrate when it bumps; coordinated with the fuel lane.
- Mechanical migration: `privatize_views.py --matrix` rewrote 264 literals in 37 files; ~390 field reads fixed from rustc E0616 spans (`fix_reads.py`, now also for the Matrix/Vector types, driven over every (crate, feature) pair of baracuda-kernels-types, -cutlass, -kernels, -kernels-bench and -examples). Stated limits: device-verified only for the 30 test binaries that use these types (16 kernels, 14 cutlass; 261 tests, 0 failed), not the whole 520-binary kernels suite; type-checks only for every other feature.

## 0.0.1-alpha.139 - 2026-10-19 (every declared feature now compiles; CI builds each one)

Workspace alpha.139 (proposed; PM allocates at gate).

- Every cargo feature of every workspace crate now compiles, and a new CI job (`each-feature-check`, 4 shards, `scripts/check-each-feature.py`) builds each (crate, feature) pair on its own with `DOCS_RS=1` (167 pairs, derived from `cargo metadata`, so new features join automatically). Before the fixes the script reported failing pairs (18 by my count, accumulated over the runs made while fixing, not from one clean run of origin/main): `baracuda-kernels` `mhc`, `xformers_blocksparse`, `xformers_sparse24` (missing `c_void` / `map_status` imports), `ring_attention` (missed by the alpha.138 private-field migration), and `awq`, `bnb_nf4`, `mamba`, `ring_attention`, `xformers_blocksparse`, `xformers_sparse24` of `baracuda-kernels-sys` (73 undocumented FFI fns under `missing_docs`); `baracuda-examples` `cutlass-smoke` (missing `GemmArgs::bias`) and `forge-hello` (build.rs ignored `DOCS_RS`); `baracuda-kernels-bench` `cuda_l2` (build.rs ignored `DOCS_RS`; bench used removed driver APIs); `baracuda-tensorrt` / `baracuda-tensorrt-sys` `shim` (`extern "C"` without `unsafe`, edition 2024). All fixed; no behaviour change in any default build.
- `scripts/check-feature-coverage.sh` now also fails if `ci.yml` stops running `scripts/check-each-feature.py` (mutation-checked: a comment-only mention does not satisfy it). Stated limit: this type-checks (`cargo check --all-targets`), it does not build kernels or run anything; a feature that truly needs the toolkit must be listed in `NEEDS_TOOLKIT` (currently empty).

## 0.0.1-alpha.138 - 2026-10-19 (BREAKING: TensorRef/TensorMut fields are private; validated constructors; bias vector bounds)

Workspace alpha.138 (proposed; PM allocates at gate).

- **BREAKING (API):** `TensorRef` / `TensorMut` (baracuda-kernels-types) no longer have public fields. Build a view with `TensorRef::new(data, shape, stride)` / `::contiguous(data, shape)` (same for `TensorMut`), which return `Err(ViewError)` when the shape/stride pair reaches outside the slice (packed S4/U4/Bin counted in slots via the new `elements_per_slot::<T>()`); read them back with `.data()` / `.shape()` / `.stride()` (`TensorMut` also `.data_mut()`). `unsafe TensorRef::new_unchecked` / `TensorMut::new_unchecked` remain for offset views (ContiguizePlan's `source_offset`). A struct literal no longer compiles (`compile_fail,E0451` doctest), so a safe API can no longer hold an out-of-bounds view. Closes the follow-up to alpha.135. Migration: `TensorRef { data, shape, stride }` -> `TensorRef::new(data, shape, stride)?`; `v.shape` -> `v.shape()`. `baracuda-kernels` re-exports `ViewError`.
- Mechanical migration with the scripts in `tools/soundness/` (`privatize_views.py`: 5927 literals in 490 files, 3 more by hand; `fix_reads.py` + `loop_reads.py`: ~2900 field reads fixed from rustc E0616 spans). Fuel is unaffected (it pins baracuda-kernels-types alpha.116 and uses none of these types).
- Also fixed: the fp8 / int4 / bespoke-int GEMM bias `VectorRef` only had its `len` compared with N, never its slice (the epilogue could read past it); now `check_vector`. ViewError's Display text no longer contains a run of spaces.
- Bias over-read, exact scope: the bias epilogue of `Fp8GemmPlan` (`Bias`, `BiasRelu`, `BiasGelu`, `BiasSilu`; E4M3/E5M2, RCR and RRR), `Int4GemmPlan` (S4/U4, RCR and RRR, same four `Bias*` epilogues, f32 or i32 bias) and the bespoke RRR backend of `IntGemmPlan` (S8/U8, same four `Bias*` epilogues, f32 or i32 bias) read `bias[0..N]` from the bias slice without checking that the slice holds N elements (the bespoke int RRR path did not even compare `bias.len` with N, and none of these three `run()`s called `can_implement` before alpha.135). The CUTLASS-backed `IntGemmPlan` RCR path and the float GEMM argument validator in baracuda-cutlass (plan.rs) already checked `bias.data.len() >= N`. Introduced with the sm_89 GEMM dtype matrix (commit c471106d, 2026-05-13, workspace alpha.16 era); fixed in alpha.138.
- Stated limits of alpha.138: the full 520-binary device suite was run once on the migration and NOT repeated end to end after the four fallout fixes and the bias change (the 18 affected binaries were re-run green); the cargo features `mhc`, `xformers_blocksparse` and `xformers_sparse24` do not compile on origin/main (pre-existing) so code behind them was not compile-checked here.

## 0.0.1-alpha.137 - 2026-10-18 (types-derive: real compile_fail doctests; drop unused trybuild)

Workspace alpha.137 (proposed; PM allocates at gate).

- `derive_device_repr` docs claimed compile_fail doctests that did not exist and `trybuild` was declared in the workspace but unused. Added three real `compile_fail` doctests (no repr, enum, non-DeviceRepr field; each verified to fail for the intended reason) and removed the unused `trybuild` workspace dependency.

## 0.0.1-alpha.136 - 2026-10-18 (BREAKING: async host copies and PreparedGroupedGemm::run are unsafe fns)

Workspace alpha.136 (proposed; PM allocates at gate).

- **BREAKING (API):** `DeviceBuffer/DeviceSlice::copy_from_host_async`, `copy_to_host_async` (baracuda-driver, baracuda-runtime), `copy_h_to_d_2d_async` (driver + runtime), driver `memcpy3d` `copy_from_host_async` and `baracuda_cutlass::PreparedGroupedGemm::run` are now `unsafe fn` with a `# Safety` contract. They borrowed the host slice only for the call, but with pinned memory the DMA runs after the call returns, so safe code could free or mutate the host buffer under an in-flight copy. Soundness fix; call sites need `unsafe { .. }` and must keep the host memory alive and untouched until the stream is synchronized. Affected-from: baracuda-driver/runtime alpha.2, baracuda-cutlass alpha.6.
- Proof: `compile_fail,E0133` doctests on the driver async copies (a safe call no longer compiles).

## 0.0.1-alpha.135 - 2026-10-18 (BREAKING (stricter validation): every plan bounds-checks its tensor views; fixes safe-API out-of-bounds read/write)

Workspace alpha.135 (proposed; PM allocates at gate).

- **BREAKING (behaviour):** every kernels plan now rejects a launch whose `TensorRef`/`TensorMut`/`MatrixRef` shape+strides reach past the `DeviceSlice` behind it (`Error::BufferTooSmall` / `InvalidProblem`). Previously the safe `run()` bounded buffers only by element count, so `shape=[N], stride=[2]` over an N-element slice read/wrote N-1 elements past it (reproduced on an RTX 4070: 32/64 elements clobbered or leaked). Soundness fix; affected-from baracuda-kernels alpha.26.
- Scope: 172 `can_implement` fns (script `tools/soundness/insert_check_fits.py`), 48 plan files / 89 `run()` fns that had no `can_implement` (`insert_run_checks.py`, dense `numel` check, stride-aware for 3 loss plans), `bin`/`fp8`/`int4`/bespoke-`int` GEMM (no buffer check at all, `run()` now calls `can_implement`), `ContiguizePlan` bounds `source_offset` + strides, packed S4/U4/Bin views counted in slots. Counts of `view_check::check_*` call sites: 0 on origin/main, 952 across 218 files now.
- baracuda-kernels-types: new `TensorRef/TensorMut::new` / `::contiguous` (validated, return `ViewError`), `required_len()`, `required_len(shape, stride)`. The public struct fields stay (making them private is a separate later change).
- Tests: `soundness_tensor_view_bounds` (5, red on origin/main) and `soundness_family_views` (17, one per op family). `flash_decoding_smoke` validation case now sizes its buffer for the strided reach.

## 0.0.1-alpha.134 - 2026-10-16 (cuda-sys: layout assertions; fix 5 drifted FFI struct layouts)

Workspace alpha.134 (proposed; PM allocates at gate).

- Found by comparing size_of/align_of with CUDA 13.3 headers: cudaPointerAttributes 24->56 (runtime wrote past it), cudaResourceViewDesc 48->112 (safe baracuda-runtime wrappers passed a short struct that C reads), CUdevSmResource +flags, CUlaunchAttributeValue/CUlaunchAttribute align 8. New src/layout.rs asserts 60 structs at compile time.

## 0.0.1-alpha.133 - 2026-10-16 (cufft: device test for strided PlanMany; docs)

Workspace alpha.133 (proposed; PM allocates at gate).

- Added strided_c2c_columns_match_contiguous (advanced data layout), documented that NULL embeds ignore stride/dist, ROADMAP cuFFT row marked already exposed.

## 0.0.1-alpha.132 - 2026-10-16 (api-freeze: restore ROPE_DEFAULT_BASE visibility)

Workspace alpha.132 (proposed; PM allocates at gate).

- Census re-run found ROPE_DEFAULT_BASE hidden by an orphaned doc(hidden) left by #185; removed it.

## 0.0.1-alpha.131 - 2026-10-16 (lint: collapse no-reason allow attributes)

Workspace alpha.131 (proposed; PM allocates at gate).

- Removed 394 dead too_many_arguments allows; FFI naming allows collapsed to one reasoned crate-level allow; the rest carry reasons.

## 0.0.1-alpha.130 - 2026-10-16 (Retire #[ignore] in the 21 remaining library crates behind device-tests)

Workspace alpha.130 (proposed; PM allocates at gate).

- baracuda-{cudf,cudnn,cufft,cufile,curand,cusolver,cusparse,cutensor,cuvs,cvcuda,megatron,nccl,npp,nvcomp,nvimagecodec,nvjitlink,nvjpeg,nvml,nvrtc,nvshmem,tensorrt} device tests (53 `#[ignore]`) move behind a per-crate `device-tests` feature, run by the self-hosted device runner. Missing optional libraries (cuDNN, libcudf, cuVS, nvImageCodec, ...) are now declared skips (`require_optional!`) instead of panics. Ran on RTX 4070 (sm_89): 112 passed, 0 failed, 34 declared skips.

## 0.0.1-alpha.129 - 2026-10-16 (Retire #[ignore] in cublas, transformer-engine, optim behind device-tests)

Workspace alpha.129 (proposed; PM allocates at gate).

- baracuda-cublas (10), baracuda-transformer-engine (10) and baracuda-optim (9) device tests move behind a per-crate `device-tests` feature, run by the self-hosted device runner. Ran on RTX 4070 (sm_89): 18 + 11 + 14 passed, 0 failed; 3 optim NCCL tests declared-skipped (UNVERIFIED).

## 0.0.1-alpha.128 - 2026-10-16 (Retire #[ignore] in kernels-bench, cuda-emit, cuda-seam behind device-tests)

Workspace alpha.128 (proposed; PM allocates at gate).

- baracuda-kernels-bench (16), baracuda-cuda-emit (10) and baracuda-cuda-seam (1) device tests move behind a per-crate `device-tests` feature, run by the self-hosted device runner; CI type-checks both feature states. Ran on RTX 4070 (sm_89): 36 + 424 + 17 passed, 0 failed.

## 0.0.1-alpha.127 - 2026-10-16 (Test: retire #[ignore] in baracuda-ozimmu; ozIMMU handle cache holds its Context)

Workspace alpha.127 (proposed; PM allocates at gate).

- baracuda-ozimmu device tests now behind device-tests (19 run on the sm_89 device runner: 0 failed).
- Fix: baracuda-cutlass's thread-local ozIMMU handle cache now holds a Context clone (same teardown hazard as the cuBLAS cache fixed in alpha.123); new tests/ozaki_backend_smoke.rs drives the Ozaki GemmPlan backend (matches CPU DGEMM; per-thread drop-order loop). The crash was NOT reproducible on the ozimmu cache before the change (0/20), so that edit is defensive.

## 0.0.1-alpha.126 - 2026-10-16 (Test: retire #[ignore] in baracuda-flashinfer; repair the flashinfer feature build)

Workspace alpha.126 (proposed; PM allocates at gate).

- baracuda-flashinfer device tests now behind device-tests (21 run with --features flashinfer,device-tests on the sm_89 device runner: 0 failed).
- Fix: the flashinfer feature did not compile at all (never built by CI): 65 undocumented FFI declarations in baracuda-kernels-sys tripped missing_docs=deny, eight baracuda-kernels attention/random modules lost their c_void/map_status imports to an unused-import sweep run without the feature (now cfg-gated), and the crate-level doctest used stale import paths.

## 0.0.1-alpha.125 - 2026-10-16 (Test: retire #[ignore] in baracuda-runtime)

Workspace alpha.125 (proposed; PM allocates at gate).

- baracuda-runtime device tests now behind the device-tests feature (59 run on the sm_89 device runner: 0 failed; 3 declared skips for exports absent on this Windows box).

## 0.0.1-alpha.124 - 2026-10-16 (CI: --locked for tools/kiss-ref-diff (SUP-001))

Workspace alpha.124 (proposed; PM allocates at gate).

- The kiss-ref-diff check now runs with --locked against a committed tools/kiss-ref-diff/Cargo.lock; each alpha bump must regenerate that lock.

## 0.0.1-alpha.123 - 2026-10-16 (Test: retire #[ignore] in baracuda-cutlass; fix cuBLAS handle cache)

Workspace alpha.123 (proposed; PM allocates at gate).

- baracuda-cutlass device tests now behind the device-tests feature (113 run on the sm_89 device runner: 0 failed).
- Fix: the thread-local cuBLAS handle cache now holds a Context clone, so cublasDestroy at thread exit no longer runs against a destroyed context (intermittent access violation, found by the first real run of these tests).

## 0.0.1-alpha.122 - 2026-10-16 (Docs: UNVERIFIED list for baracuda-kernels feature-gated tests)

Workspace alpha.122 (proposed; PM allocates at gate).

- Adds section D to docs/UNVERIFIED-HARDWARE.md (57 files / 263 tests behind features the device run does not build).

## 0.0.1-alpha.121 - 2026-10-16 (Test: retire #[ignore] in baracuda-kernels (slice 5/5))

Workspace alpha.121 (proposed; PM allocates at gate).

- Final slice behind the device-tests feature; no library code changes.

## 0.0.1-alpha.120 - 2026-10-16 (Test: retire #[ignore] in baracuda-kernels (slice 4/5))

Workspace alpha.120 (proposed; PM allocates at gate).

- Slice 4 of 5 behind the device-tests feature; no library code changes.

## 0.0.1-alpha.119 - 2026-10-16 (Test: retire #[ignore] in baracuda-kernels (slice 3/5))

Workspace alpha.119 (proposed; PM allocates at gate).

- Slice 3 of 5 behind the device-tests feature; no library code changes.

## 0.0.1-alpha.118 - 2026-10-16 (baracuda-kernels: #[ignore] retired, slice 2/5)

Workspace alpha.118 (proposed; PM allocates at gate).

- 104 more GPU test files (affine_inplace_ffi_smoke .. ) lose every `#[ignore]`, gated by the `device-tests` feature (2 files keep host-only tests with their helpers gated).
- Verified on RTX 4070 at the 5-slice head (which contains this slice's exact gating): `cargo gpu-test -p baracuda-kernels --features device-tests` 2055 passed, 0 failed, 0 skips. Files behind other cargo features are not run by that command: UNVERIFIED.

## 0.0.1-alpha.117 - 2026-10-16 (baracuda-kernels: device-tests feature; #[ignore] retired, slice 1/5)

Workspace alpha.117 (proposed; PM allocates at gate).

- New `device-tests` feature on `baracuda-kernels`; 104 GPU test files (adaptive_avg_pool2d_smoke .. fa2_varlen_smoke) lose every `#[ignore]` (file-level cfg, per-test cfg in 3 files that keep host-only tests). Tests that never carried `#[ignore]` but need a device (`select_rejects_*` and kin) are gated too: they failed a no-GPU `cargo test`.
- CI `arch-gated-check` type-checks the kernels test targets with and without `device-tests`; `device-tests.yml` runs the kernels suite on the self-hosted runner (still inert until `DEVICE_RUNNER_ENABLED`).
- Verified on RTX 4070: `cargo gpu-test -p baracuda-kernels --features device-tests` 2055 passed, 0 failed, 0 skips (default sm80+fa2). Files behind other cargo features (cudnn, fa2_backward, mamba, sm89, awq, flashinfer: 20 in this slice) are not run by that command: UNVERIFIED.

## 0.0.1-alpha.116 - 2026-10-16 (unpopped 0.20.1; contraction v1 typed declines)

Workspace alpha.116 (proposed; PM allocates at gate).

- Dependencies: `unpopped`, `unpopped-vocab`, `unpopped-cpu-c`, `unpopped-slang` and `tools/kiss-ref-diff` move to 0.20.1 (`ContractionAxes` -> `ContractionRoles`; the plan gate's `PlanError::UnservableContraction` arrives as `LowerError::InadmissiblePlan` with no baracuda code). `baracuda-cuda-emit` and `baracuda-cuda-parse` 0.20.1 (they follow the `unpopped` they work with).
- `baracuda-cuda-emit` contraction v1: the Tiny-M `assert_eq!` and float-dtype `assert!` are now typed `LowerError`s, and the emitter also declines what the 0.20 key can declare but it does not spell: accumulator other than f32 (f64 for f64), reduced-mantissa math for non-F32, a designated scale operand, and a weight dtype other than the operand dtype. One test per case plus a lowering control.
- The keyer keys every plain-F32 contraction `rm` (vendor TF32 routing policy); this SIMT kernel is a full-precision binary32 FMA chain with no input rounding. That conforms: `rm` only PERMITS reduced-mantissa rounding (KISS-CLASSIFY-6.7-0006; confirmed by the Unpopped lane), so plain F32 accepts `rm` and other dtypes decline it. A cache joining variants on the key alone treats this kernel and a TF32 vendor kernel as interchangeable; a caller needing bit-exact binary32 declares `st`. Pinned by `contraction_control_lowers_and_f32_rm_key_is_accepted`.

## 0.0.1-alpha.115 - 2026-10-16 (baracuda-driver: no more #[ignore]; device-tests feature + self-hosted runner)

Workspace alpha.115 (proposed; PM allocates at gate).

- `baracuda-driver`: 91 `#[ignore]` removed. On-device tests sit behind the new `device-tests` feature, are type-checked in CI on every PR, and are run by `cargo gpu-test` (run record `target/device-test-results.json`).
- New `device-tests.yml` (self-hosted, push main/schedule/dispatch, never fork PRs) and `device-tests-detector.yml` (red if no successful run in 7 days); both inert until `vars.DEVICE_RUNNER_ENABLED == 'true'`. `docs/DEVICE-TESTS.md`, `docs/UNVERIFIED-HARDWARE.md`.
- Verified locally on RTX 4070: `cargo gpu-test -p baracuda-driver --features device-tests` 114 passed, 0 failed, 1 declared skip.

## 0.0.1-alpha.114 - 2026-10-16 (tests: no more #[ignore]d emitter dumps, no allow(dead_code))

Workspace alpha.114 (proposed; PM allocates at gate).

- `baracuda-cuda-emit`: the 17 source-dump tests (16 in `cuda.rs`, `lift.rs::dump_lift_roundtrip`) run in a plain `cargo test`; output goes to `$VAR` if set, else `<crate>/target/baracuda-emit-dump/<test>`.
- All 37 `#[allow(dead_code)]` removed: dead items deleted, keep-alive fields renamed `_x`, cfg-only items gated by the cfg that uses them (cufile, fa2). No public API change.
- Verified: `DOCS_RS=1 cargo check --workspace --all-targets --locked`, kernels in 3 feature configs, emit 321 tests, nccl/megatron/forge/core/npp/curand tests, rustfmt.

## 0.0.1-alpha.113 - 2026-10-16 (`unpopped` family 0.18.0 -> 0.19.0) - Breaking

Workspace alpha.113 (proposed; PM allocates at gate); `baracuda-cuda-emit` and `baracuda-cuda-parse` 0.19.0 (they follow the `unpopped` 0.19.0 they work with).

- Dependencies: `unpopped`, `unpopped-vocab`, `unpopped-cpu-c`, `unpopped-slang` and `tools/kiss-ref-diff` move to 0.19.0 (KISS 0.4.3 `ctr` stage: caller-declared M/N/K/batch axes, accumulator and math fidelity; the keyer now declines malformed role hints).
- No source or golden change: baracuda has no caller of `derive_structure_key*` or `CellFacts`, so no `ContractionAxesNotDeclared` site. Emit 305 lib + integration, parse, all pass unchanged.
- Verified: sk harness (`unpopped-vocab =0.19.0`) vs KISS `026feba` `structure_key_derivation_vectors.json`, 63 vectors: ctr FAIL 15 -> 0; 47 PASS(non-prefix) + 1 PASS; 15 decline vectors give the same reason under a different name string.

## 0.0.1-alpha.112 � 2026-10-16 (sk6 adoption: `unpopped` family 0.17.3 -> 0.18.0) � Breaking

Workspace alpha.112 (proposed; PM allocates at gate); `baracuda-cuda-emit` and `baracuda-cuda-parse` 0.18.0 (they follow the `unpopped` 0.18.0 they work with).

- Dependencies: `unpopped`, `unpopped-vocab`, `unpopped-cpu-c`, `unpopped-slang` and `tools/kiss-ref-diff` move to 0.18.0 (KISS#588, conformance 0.4.0: structure key `sk5` -> `sk6`, optional `q<dtype>` on quantized `ctr` cells, Announce `list_version` 2 + `op_identity`). Breaking: every emitted structure key now starts `sk6|`.
- `baracuda-cuda-emit` goldens `contract_canonical{,_batched}_f32.cu`: line 2 only, `sk5|ctr|` -> `sk6|ctr|`; kernel bodies byte-identical. Test prefixes: `contract.rs`, `jit.rs`, `link.rs` `sk5|` -> `sk6|`.
- Announce speakers and the keyer live in `unpopped`; baracuda has none of its own, so there is nothing to regenerate here beyond the above.
- Verified: emit 305 unit + integration tests, parse, seam, cuda-vocab, kernels-types, types all pass; sk harness (`unpopped-vocab =0.18.0`) vs KISS `structure_key_derivation_vectors.json` at KISS `86196f7`: 36 vectors, 31 PASS(non-prefix), 1 PASS, 4 DECLINE-DIFFERS (3 classifier string artifacts, 1 contraction skip KISS#543), 0 FAIL.

## 0.0.1-alpha.111 � 2026-10-15 (unpopped family 0.17.1 -> 0.17.3)

Workspace alpha.111 (proposed; PM allocates at gate); `baracuda-cuda-emit` and `baracuda-cuda-parse` 0.17.3 (they follow the `unpopped` 0.17.3 they work with).

- Dependencies: `unpopped`, `unpopped-vocab`, `unpopped-cpu-c`, `unpopped-slang` and `tools/kiss-ref-diff` move to 0.17.3.
- No source or golden change: emit/parse/seam tests pass unchanged (emit 305 unit + contract 50 + jit 22 + link 2; parse 13; seam 17).

## 0.0.1-alpha.110 — 2026-10-14 (ci: CodeRipper `version-consistency` allowances)

PM-allocated (alpha.110). Config only; no source or dependency change. Merging is not a publish.

- New repo-local `.coderipper.toml` with 4 `[[allow]]` entries for `version-consistency`: `baracuda-cuda-emit` and
  `baracuda-cuda-parse` (follow the `unpopped` MAJOR.MINOR by the documented versioning exception) and
  `baracuda-examples` / `xtask` (`publish = false`, `0.0.0`). Chose `[[allow]]` over `[[tracks]]` because `tracks`
  needs a laptop-relative manifest path. CodeRipper 0.6.24: 4 un-waived High -> 4 `[waived]`.

## 0.0.1-alpha.109 — 2026-10-14 (ci: `cargo --locked`, SUP-001)

PM-allocated (alpha.109). CI only; no source or dependency change. Merging is not a publish.

- `ci.yml`: `--locked` on 20 `cargo build|check|clippy|test|doc|metadata` invocations, so CI resolves strictly from the
  committed `Cargo.lock`. Left unlocked on purpose (PM-ruled waiver): `cargo check --manifest-path
  tools/kiss-ref-diff/Cargo.toml` — its own resolution root with no committed lock, path-depending on
  `baracuda-cuda-emit`, whose version moves on every bump.

## 0.0.1-alpha.108 — 2026-10-14 (docs: C-ABI stability policy adopted, board 198)

PM-allocated (alpha.108). Documentation only. Merging is not a publish.

- `docs/planning/api-freeze/README.md`: R4 reconciled to CireSnave's 2026-10-14 ruling (quoted verbatim): from 1.0
  the C ABI does not break within a major version. §4.2 now states what that implies for the 5 809 `stable-c-abi`
  items (freeze list, additive-only, new-symbol-never-edit, deprecation through the major, `abi_version`).

## 0.0.1-alpha.107 — 2026-10-14 (sk5 adoption: `unpopped` family 0.16.0 -> 0.17.1)

PM-allocated: workspace alpha.107; `baracuda-cuda-emit` and `baracuda-cuda-parse` 0.17.1 (they follow the `unpopped` 0.17.1 they work with).
Merging is not a publish. `STRUCTURE_KEY_VERSION` 4 -> 5 (Unpopped#58, wire-visible).

- Dependencies: `unpopped`, `unpopped-vocab`, `unpopped-cpu-c`, `unpopped-slang` and `tools/kiss-ref-diff` move to 0.17.1.
  No source change was needed: baracuda only calls `structure_key`, which keeps its `op` argument.
- `baracuda-cuda-emit` goldens `contract_canonical{,_batched}_f32.cu`: line 2 only, `sk4|gem|` -> `sk5|ctr|`; the kernel
  bodies are byte-identical. Test prefixes: `contract.rs` `sk5|`, `jit.rs` `sk5|`, `link.rs` `sk5|ew2|f32|cuda:sm89|` (an add
  cell is now keyed by its derived class `ew2`, not the op family `bin`).
- Not covered: a declared contraction (`<wdt>` reads operand 1 until Unpopped's weight role lands, Unpopped#31).

## 0.0.1-alpha.106 — 2026-10-14 (ci/docs hygiene: CODEOWNERS, least-privilege permissions, glossary)

PM-tasked from the CodeRipper portfolio triage (alpha.106 allocated by the PM). Merging is not a publish. No behaviour change.

- `.github/CODEOWNERS`: `* @ciresnave` (WSP-001).
- Workflows (SUP-008): `ci.yml` gets top-level `permissions: contents: read`; the two liveness workflows move
  `issues: write` from workflow level to their single job. Every action was already pinned to a full SHA.
- `GLOSSARY.md`: the project's domain terms, defined once (DOC-009).

## 0.0.1-alpha.105 — 2026-10-14 (docs: provenance audit method note)

PM-tasked (alpha.105 allocated by the PM). Merging is not a publish. Documentation only.

- `docs/provenance/METHOD.md`: one-page reusable method (index enumeration, origin tracing, blob-id verification,
  pins, verbatim-licence rule, what to record, three-bucket PR shipping with proofs) for the fuel / lightbulb /
  mlmf lanes. `PROVENANCE.md` links it.

## 0.0.1-alpha.104 — 2026-10-14 (docs: provenance — upstream reference commits recorded)

PM-tasked (alpha.104 allocated by the PM). Merging is not a publish. Documentation only.

- `VENDOR.md` / README of bitsandbytes, causal-conv1d, mamba, xformers, apex, transformer-engine, fuel-q8_1
  and flash-attention now record an upstream reference commit (date-bounded where the vendor-time commit was
  never recorded; stated as such), and the fuel-q8_1 Fuel -> candle -> llama.cpp / xinfer chain by blob equality.

## 0.0.1-alpha.103 — 2026-10-14 (docs: provenance — header-comment corrections)

PM-tasked (alpha.103 allocated by the PM). Merging is not a publish. **Comment-only edits to 32 code files:
after stripping comments and whitespace every one is identical to its previous blob
(`docs/provenance/header-comment-changes-verified.tsv`).**

- cast/fill/affine: the unsupported "Eric Holscher" holder replaced by the true origin (candle contributors via
  Fuel; upstream names no holder). attention.rs licence label corrected to MIT. Apex rewrites: invented NVIDIA
  copyright line replaced by an origin note. SPDX/origin added to sort, cutf-derived headers, CUDA-L2 probes.
- `baracuda-forge`: `NOTICE` restated with measured per-file differences (four "verbatim" claims did not hold);
  `// Origin:` line on each module.

## 0.0.1-alpha.102 — 2026-10-14 (docs: provenance — licence texts carried verbatim)

PM-tasked (alpha.102 allocated by the PM). Merging is not a publish. **Licence/attribution files only: no
.cu/.cuh/.h/.rs byte changed.**

- Added verbatim upstream licence texts: candle (MIT + Apache-2.0), llama.cpp, Ootomo cutf, CUDA-L2,
  Maratyszcza/FP16, bitsandbytes `NOTICE.md`; `docs/provenance/licence-files-verified.tsv` proves blob equality.
- Replaced non-verbatim `LICENSE` (bitsandbytes, causal-conv1d, mamba, xformers, apex), `AUTHORS`
  (causal-conv1d, mamba) and TE `ATTRIBUTION.md`'s embedded Apache text with upstream bytes.
- Baracuda-written AUTHORS files now say so on their first line.

## 0.0.1-alpha.101 — 2026-10-14 (docs: third-party provenance audit — `PROVENANCE.md`)

PM-tasked (alpha.101 allocated by the PM). Merging is not a publish. No code change; no vendored file,
licence file or source header touched.

- New `PROVENANCE.md` + `docs/provenance/*.tsv`: every vendored/derived file traced to origin and
  compared against upstream; gap register G-* and the proposed fixes that need a PM decision.
- Corrected baracuda-authored docs only: FlashInfer tag/commit, FA2 modified-file list + candle pin,
  AWQ launcher path, mHC and Apex header claims, `LICENSE-thirdparty.md` (q8_1 family now ships;
  unresolvable llama.cpp commit id).

## 0.0.1-alpha.100 — 2026-10-14 (docs: API-freeze — Fuel `*Descriptor` exposure measured; census-per-release process)

PM-tasked (alpha.100 allocated by the PM; docs are not exempt). Merging is not a publish. No code change.

- `docs/planning/api-freeze/README.md` §7.1: Fuel `e6cbf782` references 0 of the 257 kernels `*Descriptor`
  structs (positive control included); §7.2: census re-run per release. Section order fixed (7, 8).
- ROADMAP: slices (c)/(d) recorded.

## 0.0.1-alpha.99 — 2026-10-14 (docs: API-freeze slice 2 — family denominators + R2 worklist)

PM-tasked (alpha.99 provisional, PM allocates at gate). Merging is not a publish. No code change.

- `docs/planning/api-freeze/`: `kernels-families.tsv` (R6 denominator), `r2-worklist.tsv` (792 pub-field
  structs lacking `#[non_exhaustive]`, split Descriptor / Args / other), README section 8, `tool/slice2.py.txt`.

## 0.0.1-alpha.98 — 2026-10-14 (docs: `Element` vs `DeviceRepr + Copy` layering note)

PM-tasked (R1 of the API-freeze review; alpha.98 provisional, PM allocates at gate). Merging is not a publish.
No behavioural change.

- `ARCHITECTURE.md`: new "Trait layering" section; corrected the sub-trait list (`IntElement` / `FpElement` /
  `BinElement` are `KernelDtype` markers, not `Element` sub-traits).
- `baracuda-kernels-types` crate docs: why the two bounds coexist and that unifying is Unpopped's.

## 0.0.1-alpha.97 — 2026-10-14 (docs: API-freeze inventory, classification, breaking-change policy — slice 1)

PM-tasked. Merging is not a publish. No code change.

- New `docs/planning/api-freeze/`: a `syn`-based census of every `pub` surface from `git ls-files`
  (20 076 items / 74 library crates; supersedes the line-grep first pass), a mechanical
  stable / C-ABI / vendor-mirror / `doc(hidden)` / candidate-for-removal classification, the
  *proposed* (not adopted) breaking-change policy including the C ABI, and the `Element` vs
  `DeviceRepr + Copy` recommendation (keep the split, document it; unifying is Unpopped's).
- Enumerator + classifier source shipped as `tool/*.txt` for reproduction.
- No `0.1.0-beta.0` cut; nothing hidden, narrowed or removed.

## 0.0.1-alpha.96 — 2026-10-14 (CI: third-party Actions pinned to full commit SHAs)

PM-tasked (alpha.96 allocated by the PM). Merging is not a publish. No code change.

- `.github/workflows/{ci,cuda-vocab-pin-liveness,pytorch-baseline-liveness}.yml`: 11 `uses:`
  lines moved from mutable refs (`@v4`, `@v2`, `@stable`, `@1.85.0`) to 40-hex commit SHAs
  with the tag in a trailing comment (checkout v7.0.1, rust-cache v2.9.2, rust-toolchain).

## 0.0.1-alpha.95 — 2026-10-14 (docs: ROADMAP refresh + API-freeze pub-surface inventory)

PM-tasked (alpha.95 allocated by the PM; docs are not exempt from the version rule).
Merging is not a publish. No code change.

- `ROADMAP.md`: header corrected (`main` was stated as alpha.85); the Linalg bench is
  no longer called the "next tranche" (`benches/linalg.rs` exists; the open part is the
  `BENCHMARKS.md` rollup, which needs a quiet-box run); a dated wait for
  `baracuda-cuda-seam` publishability, blocked on fuel's board #106 Step A.
- NEW `docs/planning/api-freeze-pub-surface-audit.md`: public-surface inventory and the
  `Element` vs `DeviceRepr + Copy` split resolved by reading (Unpopped-owned sealed subtrait).

## 0.0.1-alpha.94 — 2026-10-12 (Fuel seam moves out of baracuda-cuda-emit — **Breaking** for `cuda-emit/seam`)

PM-tasked (alpha.94 and cuda-emit 0.16.1 allocated by the PM). Merging is not a
publish.

- **BREAKING — `baracuda-cuda-emit 0.16.1`: the `seam` feature and
  `baracuda_cuda_emit::seam` / `baracuda_cuda_emit::BaracudaSynthesizer` are
  removed.** They moved, unchanged, to the new `baracuda-cuda-seam` crate
  (`baracuda_cuda_seam::BaracudaSynthesizer`; its `nvrtc` feature forwards to
  cuda-emit's). Why: `cargo publish` of cuda-emit 0.16.0 failed with
  "dependency `fuel-kernel-seam` does not specify a version" — crates.io rejects
  any git dependency, optional or not, and board #106 Step A
  (`JitRequest::target: TargetId`, used by `BaracudaSynthesizer`) exists only on
  fuel `main` (c02da348); the registry's latest `fuel-kernel-seam` is 0.11.2 with
  `arch: ArchSku` only. With the two git lines out of its manifest cuda-emit
  publishes again; crates.io has it only at 0.14.3.
- **Consumer: `fuel-dispatch`** (fuel `origin/main`,
  `fuel-dispatch/Cargo.toml:70`, feature `jit-synth`, plus
  `tests/jit_synth_kernel_live.rs`) depends on `baracuda-cuda-emit` with
  `features = ["seam", "nvrtc"]`. It pins cuda-emit `0.11.0`, so nothing breaks
  until fuel follows to `>= 0.16.1`; then it must depend on `baracuda-cuda-seam`
  instead. `baracuda-cuda-seam` is `publish = false` until fuel publishes its
  Step-A seam crates (no date), so fuel can only reach it by path/git until then.
- **New crate `baracuda-cuda-seam`** (`publish = false`, lockstep `alpha.94`):
  `seam.rs` and both `fuel-kernel-seam{,-types}` git pins (rev `c02da348`). CI's
  three seam legs retarget from `-p baracuda-cuda-emit --features seam` to
  `-p baracuda-cuda-seam`; it is also in the driver-free `cargo doc` list.
- cuda-emit: dropped the `seam` feature; `convert` and `nvrtc` unchanged.

## 0.0.1-alpha.93 — 2026-10-12 (baracuda-types `bytemuck` feature now does something)

PM-tasked (alpha.93 allocated by the PM). Merging is not a publish.

- **`baracuda-types` `bytemuck` feature:** the feature (and its README row,
  "`Pod` / `Zeroable` for the scalar types") existed but nothing in the crate used
  the dependency. It now provides `bytemuck::Zeroable` + `bytemuck::Pod` for
  `Half`, `BFloat16`, `Complex32` and `Complex64` (hand-written, so no
  `bytemuck/derive`; layouts are `repr(transparent)` over `u16` / `repr(C)` over
  two equal-width floats, no padding). Enables `bytemuck::cast_slice` between
  these types and `u8` / `f32` / `f64`. Additive; default features unchanged.

## 0.0.1-alpha.92 — 2026-10-12 (transmute_copy size guard + misaligned test cast)

PM-tasked; alpha.92 allocated by the PM. Merging is not a publish.

- **`baracuda-kernels` `AffinePlan::select` / `FillPlan::select`** refuse a `T`
  whose `size_of::<T>()` disagrees with the width of `T::KIND`
  (`Error::Unsupported`). `run` reads `desc.a` / `desc.b` / `desc.value` with
  `transmute_copy::<T, f32|f64|i32|i64|f16|bf16>` on the strength of the `KIND`
  match alone; a width mismatch would read past (or short of) the scalar. Unit
  tests pin the width table (`kind_size_matches`).
- **`tests/inplace_aliasing_extended_smoke.rs`:** the f32→i32 in-place readback
  cast a `Vec<u8>` pointer to `*mut f32` (a misaligned `&mut [f32]`); it now
  downloads into a `Vec<f32>` and takes `to_bits()`. All 7 on-device tests pass
  on the RTX 4070 (`--ignored`, under `gpu-run`).
- **`scripts/check-device-run-invocations.sh`:** reads each markdown file directly
  (`done < "$f"`) instead of through `body=$(cat "$f")` / `<<< "$body"`. The
  here-string hangs on MSYS/Git bash once a file passes ~64 KB (this entry pushed
  `CHANGELOG.md` over it and hung the Windows job); the rules are unchanged.

## 0.0.1-alpha.91 — 2026-10-12 (unpopped 0.16.0 follower bump)

PM-tasked follower (alpha.91 allocated by the PM). Merging is not a publish;
alpha.89 / alpha.90 stay unpublished.

- **`unpopped` family `0.15.0 -> 0.16.0`** (`unpopped`, `unpopped-vocab`,
  `unpopped-cpu-c`, `unpopped-slang`). `baracuda-cuda-emit` / `baracuda-cuda-parse`
  `0.15.0 -> 0.16.0` (MAJOR.MINOR follow unpopped; versioning exception above).
- **CpuC now lowers `f16` / `bf16`** (through an `unsigned short` carrier and
  integer helpers). `backend_declines.rs` asserted the old typed
  `UnsupportedDtype` decline; it is now
  `cpuc_emits_halves_and_f32_and_declines_vectorized_schedule`: the halves emit a
  host loop at a scalar-schedule key, a vector-aligned key is a typed
  `UnsupportedSchedule` decline (v1 serves only the scalar contiguous path), and
  f32 is the positive control.
- **KISS `cuda:` seed re-vendored at `5b28bf51`** (was `becf90fc`). The seed now
  carries two blocks: the 13-row token set (every `cuda:` token the maintainer
  supports, NVRTC-only ones included) and the 4-row dispatch set.
  `kiss-cuda-seed.tsv` holds both; `semantic_agreement_with_the_pinned_kiss_seed`
  compares manifest members to the dispatch set as before and additionally
  asserts every dispatch token is in the token set (KISS's own rule). The
  `cuda-vocab-pin-liveness` workflow now extracts tokens from both blocks (its
  old `,/^```$/p` slice stopped at the first fence and would have read an empty
  main set). **Open, not decided here:** the manifest still lists the 4 dispatch
  members, not the 13-token set; widening it is a `vocabulary_version` decision.
- Absorbed with no source change: `half_load_intrinsic` / `half_store_intrinsic`
  removal (`cfamily_shadow.rs` carries frozen local copies),
  `scalar_ctype(F16|Bf16) = "unsigned short"` (the shadow's own table),
  FP8 `Cmp*` U8 masks, kiss-ref 0.4.0 `non_exhaustive`, and the `__expf` refusal
  pin in `baracuda-cuda-parse/tests/convert.rs` (still green).

## 0.0.1-alpha.90 — 2026-10-11 (unpopped 0.15.0 / kiss-ref 0.4.0 follower bump)

PM-tasked KISS follower. Lockstep bump to `0.0.1-alpha.90`; merging is not a
publish (alpha.89 publish is deferred until this lands).

- **`unpopped` family `0.14.3 -> 0.15.0`** (`unpopped`, `unpopped-vocab`,
  `unpopped-cpu-c`, `unpopped-slang`). Breaking upstream: `__expf` / `__logf` no
  longer lift to the exact `Exp` / `Log`; they are `LiftError::Unrecognized`
  (`expf` / `logf` unchanged). `baracuda-cuda-parse/tests/convert.rs` pinned the old
  mapping: `cuda_lifts_unary_intrinsic` now uses `expf`, and a new
  `cuda_fast_math_intrinsics_refuse_not_lift` asserts both fast-math spellers refuse.
- **`baracuda-cuda-emit` / `baracuda-cuda-parse` `0.14.4 -> 0.15.0`** — MAJOR.MINOR
  follow the `unpopped` release they build against (versioning exception above).
- **`tools/kiss-ref-diff`: `kiss-ref-core` / `kiss-ops-vocab` / `kiss-classify-vocab`
  `0.3.4 -> 0.4.0`**, `unpopped` / `unpopped-vocab` `0.15.0`. The tool (its own
  `[workspace]`, not a member) type-checks against them.

## 0.0.1-alpha.89 — 2026-10-11 (lockstep bump for board #162 bench run/report guide)

`#161` docs(bench): `bench-reports/` location decision (tracked directory, reports
untrusted), `docs/guides/kernel-bench-reports.md` run/report guide and the
noisy-vs-quiet protocol. Docs only. Lockstep bump to `0.0.1-alpha.89` (every
lockstep crate; `baracuda-cuda-emit` / `baracuda-cuda-parse` stay `0.14.4`).
Merging this bump is not a publish.

## 0.0.1-alpha.88 — 2026-10-11 (lockstep bump after board #162 bench slices 1/2a/2b and the docs/emit chores)

`main` carried several merged changes at the `alpha.87` string with no version
change of their own. Per CireSnave's rule that a pushed change changes the
version, the lockstep family moves to `alpha.88`, which retroactively covers:

- `#145` docs(vendor): corrected `fuel-q8_1` provenance (llama.cpp + xinfer, both MIT).
- `#146` docs: llama.cpp/ggml CUDA technique catalogue (Phase A).
- `#148` docs: sm_61-through-latest is in scope (CireSnave ruling) + parse/emit gap list.
- `#154` chore(cuda-emit): shadow 4 CUDA scalar spellers out of `unpopped::cfamily`.
- `#156` docs(design): board #162 kernel perf baseline + cross-machine report design.
- `#157` feat(bench): board #162 slice 1 — timing stats, machine-state, baseline key.
- `#158` feat(bench): board #162 slice 2a — `BenchReport` schema + validation.
- `#159` feat(bench): board #162 slice 2b — `baracuda-bench run --report`.

Lockstep bump to `0.0.1-alpha.88` (every lockstep crate; `baracuda-cuda-emit`
/ `baracuda-cuda-parse` stay on their `unpopped`-matched line, unaffected).
Merging this bump is not a publish.

## 0.0.1-alpha.86 — 2026-10-06 (registry-build fix: baracuda-ozimmu-sys's hardcoded sibling path)

`baracuda-ozimmu-sys/build.rs` resolved `baracuda-kernels-sys`'s CUDA header
directory via `CARGO_MANIFEST_DIR.parent().join("baracuda-kernels-sys/kernels/include")`
— a filesystem-sibling path that only exists in the monorepo. Published
crates' publish-verify build (and any real consumer installing from
crates.io) panicked, since the registry doesn't lay dependencies out as
literal sibling directories. Blocked `baracuda-ozimmu-sys` and everything
depending on it: `baracuda-ozimmu`, `baracuda-cutlass`, `baracuda-kernels`,
`baracuda-flashinfer`, `baracuda-flashinfer-sys`.

Fix: `baracuda-kernels-sys` now exports its header directory through Cargo's
own `links = "baracuda_kernels"` mechanism (`cargo:include=<absolute path>`,
emitted before either of its own early-returns so it's set even under
`DOCS_RS=1` or with no arch feature active). `baracuda-ozimmu-sys` reads the
resulting `DEP_BARACUDA_KERNELS_INCLUDE` env var instead. This required
declaring `baracuda-kernels-sys` as a real `[dependencies]` edge (not
`[build-dependencies]`) on `baracuda-ozimmu-sys` — empirically confirmed a
`[build-dependencies]`-only edge does NOT trigger Cargo's `DEP_*`
propagation to the dependent's own build script, `default-features = false`
to avoid forcing kernels-sys's default `sm80` forge onto every consumer
that only wants the header.

Verified: a positive-controlled `git grep -n '\.parent()'` across every
`crates/*/build.rs` found exactly this one hardcoded-sibling-path instance
workspace-wide (none of the other 5 affected crates have their own
`build.rs` at all — they only failed transitively). Reproduced the failure
directly (not assumed) against the `[build-dependencies]`-only edge three
times in fresh, isolated target directories; confirmed the fix with a real
build of `baracuda-ozimmu-sys` (22m12s, the actual CUDA compile, not a
type-check) and `baracuda-ozimmu` (13m08s). `baracuda-cutlass` /
`baracuda-kernels` / `baracuda-flashinfer` / `baracuda-flashinfer-sys` were
NOT separately isolated-build-verified in this round — their own `ozimmu`
forwarding feature is opt-in (not default), and this session's build
contention (self-inflicted: too many concurrent fresh-target-directory
attempts while diagnosing the root cause) made further multi-crate CUTLASS
forges impractical to run safely; they inherit the same fix mechanically
(no build.rs of their own), but the PM's own full republish-verify pass is
the first REAL confirmation for those four.

Lockstep bump to `0.0.1-alpha.86` (every lockstep crate; `baracuda-cuda-emit`
/ `baracuda-cuda-parse` stay on `0.14.3`, unaffected — this fix doesn't touch
either).

## 0.0.1-alpha.85 — 2026-10-03 (version-ambiguity fix: lockstep bump after #147/#151)

`#149` set the lockstep family to `alpha.84`, but `#147` (the K-quant row
off-by-one + batched MMVQ width-decline fix, `baracuda-kernels-sys` /
`baracuda-kernels-bench`) and `#151` (board #106 Step B,
`NvrtcCompiler::new(TargetId)`, touching `baracuda-kernels-bench`'s test
callers) both merged to `main` AFTER that bump, with no accompanying version
change. The already-published 46-crate `alpha.84` batch on crates.io does
**not** carry either fix, while `main`'s source does — the same version
string would otherwise name two different code states. Per CireSnave's
one-version-per-project rule, the whole lockstep family (70 crates; the
`baracuda-cuda-emit` / `baracuda-cuda-parse` exception line stays at
`0.14.3`, unaffected) moves to `alpha.85` before any further publishing.

Pure version-string bump — no kernel/code changes in this commit. Every
`crates/*/Cargo.toml` `version` field, the root `Cargo.toml`'s matching 70
dependency pins, and `Cargo.lock`'s matching entries move from
`0.0.1-alpha.84` to `0.0.1-alpha.85`, hand-edited (not an unlocked `cargo
update`) and verified self-consistent with `cargo metadata --locked` /
`cargo check --locked` (zero resolver side effects).

## 0.0.1-alpha.84 — 2026-10-03 (unpopped-vocab family unification; sm_61-through-latest groundwork)

Per CireSnave's ruling that baracuda will support sm_61 through the latest
architecture via Unpopped's CUDA parse/emit pipeline (board #106), the
workspace moves off a stale `unpopped-vocab 0.11.0` pin and onto the current
`unpopped`/`unpopped-vocab`/`unpopped-cpu-c`/`unpopped-slang 0.14.3` line
(#149). Bumping `unpopped-vocab` alone would have put two semver-incompatible
copies of it in `baracuda-cuda-emit`'s dependency graph (one direct, one
transitive through `unpopped 0.11.0`); the whole family moves together.

**`baracuda-cuda-emit` 0.11.1 → 0.14.3** (the `unpopped`-tracking exception
line: its MAJOR.MINOR now matches the `unpopped` it builds against).

**`baracuda-kernels-types` carries a visible dependency change even though its
own public surface is purely additive.** It re-exports `unpopped_vocab::*`
wholesale; diffed byte-for-byte against the 0.11.0 and 0.14.3 published
tarballs: 5 new names (`DeriveDecline`, `dtype_numeric_kind`,
`dtype_storage_bits`, `try_structure_key`, `try_structure_key_token`), nothing
removed or renamed. Fuel, which consumes this re-export, sees only additions.

**Fixed four of baracuda's own `cuda.rs` gates that were reading
unpopped-vocab 0.14.0's structure-key semantics change incorrectly** (that
change itself landed cleanly in #148/the Unpopped#36 review — this is
baracuda's own code catching up, not a defect in Unpopped). unpopped-vocab
0.14.0 made the broadcast mask frame-coordinate and right-aligned
(KISS-Classify §6.5-0014 as amended by KISS#519): a lower-rank operand's
"this frame axis is absent" padding bit now shares bit positions with a real
own-axis stride-0 broadcast bit. The reduction general-path and im2col gates
read that mask by raw axis index, conflating the two, so a reduction/im2col
output that legitimately collapses rank (the entire point of those ops)
started reading as spuriously aliased. Switched both to
`out_key.contig == Contiguity::Broadcast`, which is driven by the operand's
own-axis-only `own_bcast` and is blind to frame padding by construction.

Two test fixes alongside the gate fix, both verified against Unpopped's own
re-derivation: a stale test pinning a now-resolved 0.11-era classification
ambiguity (renamed, expectation reversed to the new, more precise
classification), and one real correctness finding — a test that paired a
`RowReduce` op (which writes one value per input element) with a collapsed
output key, a genuine out-of-bounds-write shape that the test never caught
because it only grepped emitted source text and never ran the kernel.
Unpopped's own plan-validation gate correctly started rejecting it under
0.14.0; the fix was the test's key shape, not the gate.

70 crates move to `0.0.1-alpha.84`. `baracuda-cuda-emit` moves independently
to `0.14.3` on the `unpopped`-tracking exception line; `baracuda-cuda-parse`
stayed at `0.11.0` in #149 (flagged, not fixed there — the same question
`baracuda-cuda-emit` just answered).

**Follow-up, same day**: `baracuda-cuda-parse` — the other crate on the
exception line, same `unpopped`/`unpopped-vocab` dependency — moves to
`0.14.3` too, for the identical reason. Also corrected a stale README claim
that this crate was unpublished (`publish = false`); it has carried no such
field since it was published to crates.io on 2026-09-26, and the README
paragraph describing the exception line hadn't been updated since. The
"70 publishable crates" status line is corrected to the real, grepped count:
72 crates total, 71 publishable, 1 non-publishable (`baracuda-kernels-bench`).

**Publishing to crates.io in two phases (2026-10-03).** Phase 1 publishes 8
crates at `alpha.84`: `baracuda-types-derive`, `baracuda-core`,
`baracuda-cuda-sys`, `baracuda-types`, `baracuda-driver`,
`baracuda-kernels-types`, `baracuda-cuda-vocab`, `baracuda-cuda-emit`, plus
`baracuda-cuda-parse` (joins once this PR merges). Phase 2, during a
scheduled quiet window, publishes the remaining ~63 crates (including the 7
CUTLASS-forging ones). Until Phase 2 completes, those crates stay at their
previous published version on crates.io even though the source tree here is
uniformly at `alpha.84` — `0.0.1-alpha.82` remains the last version actually
live on the registry for most of the family until that run happens.

## 0.0.1-alpha.83 — 2026-09-27 (MMVQ launcher-side width decline)

Closes the direct-`-sys`-caller half of #128 (#139): the type-0/1 GGUF MMVQ
launchers (`launch_type01_mmvq`/`launch_type01_mmvq_strided` in
`baracuda-kernels-sys/kernels/gguf/mmvq.cu`) now decline `ncols % 64 != 0`
themselves, mirroring the Rust plan layer's existing `#127` decline
(`GgufMmvqPlan`), so a caller using this crate's raw `extern "C"` launchers
directly can no longer trigger the cross-row over-read that decline exists to
prevent. Verified with a numeric measurement, not just compute-sanitizer
memcheck: the pre-fix contamination (invisible to memcheck since it stays
inside the allocation for every row but the last) was reproduced and measured
bit-exact against a hand-derived prediction before being confirmed fixed.

Two related findings filed as follow-up issues, not fixed here: #140 (a
`row > nrows` off-by-one in the K-quant templates, latent only because
`GGML_CUDA_MMV_Y = 1` masks it) and #141 (the batched MMVQ path has the same
raw-caller exposure).

70 crates move to `0.0.1-alpha.83`. `baracuda-cuda-emit` and
`baracuda-cuda-parse` stay on the `unpopped`-tracking exception line.

**Not yet published to crates.io as of this entry** — the version is
allocated on `main` at gate time; `0.0.1-alpha.82` remains the last version
actually live on the registry until the publish runs.

## 0.0.1-alpha.82 — 2026-09-25 (baracuda-cuda-parse, CUDA parse-side donation)

Adds `baracuda-cuda-parse` (#133): the parse-side (`.cu` -> IR) half of the
CUDA donation to Unpopped's neutral core, mirroring `baracuda-cuda-emit`'s
emit-side placement. Unpublished (`publish = false`). Wired into the
driver-free CI job (build, clippy, clippy `--tests`, test, doc).

70 crates move to `0.0.1-alpha.82`. `baracuda-cuda-emit` and
`baracuda-cuda-parse` stay on the `unpopped`-tracking exception line at
`0.11.0`.

## 0.0.1-alpha.81 — 2026-09-17 (MMVQ width decline, narrow-float bit moves, unpopped 0.11, licence text)

The 107 commits after the published alpha.79 tree (`16621ddb`). 70 crates are
published: 69 at `0.0.1-alpha.81`, plus `baracuda-cuda-emit 0.11.0`.
`baracuda-cuda-vocab` is published for the first time.

`0.0.1-alpha.80` was tagged (`264882e1`) but never published: its
`baracuda-cuda-vocab` lacked the licence files every other crate carries.
alpha.81 is the same release with those files added and a guard against a
repeat.

### Fixed

- **Licence text ships in every published crate.** The workspace licence is
  `MIT OR Apache-2.0`, but alpha.79's published crates carried no licence
  files: the per-crate `LICENSE-MIT` / `LICENSE-APACHE` copies were added
  after it (#18). `baracuda-cuda-vocab` now has them too.
  `scripts/check-crate-licences.py` fails when a publishable crate lacks
  either file or holds a copy that differs from the root. It runs in CI, and
  `scripts/publish.ps1` refuses to publish when it fails.
- **GGUF MMVQ over-read (#127).** The type-0/1 MMVQ kernel (Q4_0, Q4_1, Q5_0,
  Q5_1, Q8_0) reads whole 64-column strides. `GgufMmvqPlan` and
  `GgufMmvqBatchedPlan` only required a multiple of 32, so at
  `ncols % 64 == 32` they read 32 columns past `ncols`. Under compute-sanitizer
  memcheck on an RTX 4070, Q8_0 2×32 gave 41 errors; 2×64 gives 0. Both plans
  now return `InvalidProblem` for those widths, in release builds too.
  `GgufMmvqMultiMPlan` is unaffected. Direct `baracuda-kernels-sys` callers are
  still exposed; see #128.
- **Narrow-float min/max moves the operand's bits** (KISS #355/#416,
  §6.16-0009). This covers reductions, cumulative min/max, pooling, and
  min/max/neg/abs/copysign inside the element or the epilogue. A moved f16/bf16
  NaN is no longer quieted (#67, #70, #72, #73, #75).
- **Flash SDPA and flash sm_89** decline an over-cap shared-memory request with
  a typed error instead of an opaque status 1001, and one bad shape no longer
  hides the rest (#76, #78, #82).
- **sm_89 `int4_gemm`** compile break (#55). The `BiasElementKind` import is
  now gated to sm89 (#59).
- **cuda-emit** owns the `Rsqrt` intrinsic spelling (#46).
- **Tools:** `kiss-ref-diff` compiles again, and CI now type-checks it (#125).
  The benchmark rollup aggregates runs instead of publishing an arbitrary one
  (#92), and it refuses a regen that would drop op families (#79).

### Added

- **`baracuda-cuda-vocab`:** the owner source for the `cuda:` KISS §6.8
  namespace. It includes the §6.8-0008 manifest generator, a freshness gate,
  and a semantic-agreement gate against KISS's pinned seed (#39–#43).
- **cuda-emit:** the §6.16-0009 bit-move route for `RowReduce` (#110) and
  `Select` (#118).
- **cutlass:** the build fails if a tile exceeds sm_89's shared-memory cap
  (#87, #121).
- **§6.8-0004 reachability guard**, plus a fix for the seam-reachable panic it
  found (#34).
- **Benchmarks:** loss ops against PyTorch in f32/f16/bf16 (#63, #69); Linalg
  tranche 1, cholesky and lu (#89); dispersion reported beside every median,
  including the PyTorch column (#91, #94); frozen PyTorch baseline provenance
  (#54, #61, #62); liveness checks on every self-bench (#56, #58).

### Changed

- **Dependencies:** adopted `unpopped` / `unpopped-vocab` / `unpopped-cpu-c` /
  `unpopped-slang` 0.11.0 (from `unpopped` 0.1.0 at alpha.79). This includes the
  `VariantFidelity` rename and the `Result<Spelling>` lowering seams (#18, #24,
  #33, #65, #85, #101, #123). Also newest compatible versions and `thiserror 2`
  (#47).
- **Build:** forge nvcc parallelism is capped at 4 by default through the
  workspace `[env]` (#53).
- **CI:** Rust toolchain pinned to 1.98.0, with the MSRV (1.85) scoped to the
  crates a job actually builds (#29). Also: intra-doc links denied (#28),
  test-crate locality (#27), the CUDA-free feature-gated surface (#31),
  `DOCS_RS` type-checks of the arch-gated library and bench tests (#55, #124),
  and LF line endings enforced (#30).

## 0.0.1-alpha.79 — 2026-08-15 (the Unpopped carve; contraction layouts)

*Backfilled on 2026-09-17 from commit history (`449583e6..16621ddb`, 53
commits). The published crate was built from `16621ddb`, which the retained
`release/v0.0.1-alpha.79` branch preserves.*

### Changed

- **The kernel generator moved out into the standalone, vendor-neutral
  [`unpopped`](https://crates.io/crates/unpopped) / `unpopped-vocab` crates.**
  Baracuda now consumes them from crates.io. `baracuda-kernelgen` was retired,
  and the CUDA emitter, NVRTC JIT and seam `Synthesizer` now live in the new
  `baracuda-cuda-emit`.
- `DeviceRepr` is owned by the kernel vocab and re-exported by
  `baracuda-types`.

### Added

- **Contraction layouts:** `LayoutOrder` plus lhs/rhs order on
  `ContractionKey`. `classify_mat_layout` derives packed transposes and admits
  broadcast (stride-0) axes for GQA KV. Permute views are allowed on
  contraction operands. Stride binding uses extent products, and the oracle
  reads operand layout by physical strides.
- **`require!`:** a declare-and-report primitive for on-device tests.

### Fixed

- **aarch64 Linux:** `cuDeviceGetLuid` takes a `c_char` buffer (#15).
- **forge:** honours `NUM_JOBS` as a cap and surfaces fatal `ptxas` errors.
- **vocab:** declines batch-inner packed contraction layouts.

## 0.0.1-alpha.78 — 2026-07-31 (IR-hub frontends, shape oracle, sk3, attention weights)

*Backfilled on 2026-09-17 from commit history (`38f156c5..449583e6`, 173
commits).*

### Added

- **Flash decoding:** optional per-key attention weights `a[B,H,Sk]` for
  H2O/R-KV eviction, and an optional head-mean output `a_mean[B,Sk]`.
- **FA2:** backward compilation is split behind the opt-in `fa2_backward`
  feature, so the default `fa2` build is forward-only.
- **Reductions:** S8/U8 support (wrapping store, integer predicates, and exact
  counts past 2^24).
- **Shapes:** the KISS-Ops §6.20 shape-expression vocabulary, evaluator and
  canonical byte codec, plus an output-shape oracle for every access pattern.
- **Contraction:** fused matmul with a per-column bias/activation epilogue, a
  batched `[B,M,K]·[B,K,N]` form, and a CPU numeric reference.
- **Contracts:** recipe-carrying contracts for non-elementwise ops (gather,
  scatter, RowReduce, reduction, scan), a KISC-framed bundle emitter, and the
  seam Announce read side.
- **kernelgen** (now `unpopped`): an IR→Slang backend, a portable-C backend,
  tree-sitter CUDA/Slang converter frontends with `lift`, a CPU oracle,
  precision-first (`MorePrecise`) variants, a human-readable IR text form, and
  generated freestanding helpers.
- **Decode:** capture-safe dense m=1 GEMV and `gather_rows`.
- **driver:** cross-framework `from_raw` / `borrow_raw` on `Context`,
  `Stream`, `Event` and `Device`.

### Changed (breaking)

- **`structure_key` sk2 → sk3:** the GEM precision coordinates (KISS sk3 RFC).
- **`SEAM_MAGIC`** is now `0x4D414553` (LE bytes `"SEAM"`), changed in lockstep
  with Fuel.
- **Recipes:** `Rem` → `rem_floor`, `Round` → `round_even`, and
  `reduce_extent` → `reduced_count`.

### Fixed

- NaN-propagating elementwise Max/Min emit `max_prop` / `min_prop`, which keep
  A on numeric ties.
- Zero-stride window/im2col geometry and `i64::MIN / -1` are now declined
  instead of dividing or panicking.
- The structure-key work class is the frame maximum across operands.

## 0.0.1-alpha.77 — 2026-07-09 (post-ramp codegen breadth + the CapturedRun `_doff` unblock)

Ten adversarially-reviewed increments on top of alpha.76: a device-resident
dynamic range-start WriteSlice that unblocks Fuel's `CapturedRun` (CUDA-graph
decode replay), and nine `baracuda-kernelgen` expressibility/performance
increments closing the last named codegen gaps (rope, triu, fused dropout,
fused argsort, top-k, im2col, f64 params) and adding measured cooperative
variants (block-scan, partial-select top-k). Each shipped through the standing
discipline: design-scout → background implementation → adversarial multi-lens
review (find + refute-verify) → on-device validation on an RTX 4070 (sm_89).
`baracuda-kernels-types` is UNTOUCHED (no `STRUCTURE_KEY_VERSION` bump) — every
increment keys additively.

### Added

- **`_doff` WriteSlice** (`baracuda-kernels-sys`, form B): a device-resident
  dynamic range-start variant of the bespoke `write_slice` so a captured KV
  decode write survives CUDA-graph replay (capture bakes launch-arg values; the
  offset now rides a fixed-address device cell updated by H2D memcpy). Unblocks
  Fuel's `CapturedRun`. ABI frozen with the Fuel JIT-seam session.
- **kernelgen — BASE_OFFSET slice**: runtime per-operand base element offsets
  (`long long off{i}` launch args); **rope closed BIT-EXACT** vs the bespoke.
- **kernelgen — WHERE/SELECT**: `ScalarExpr::Select` (the IR's first 3-child
  node); bit-identical triu closes the increment-0d signed-zero gap.
- **kernelgen — HETERO MULTI-OUTPUT**: per-output dtypes
  (`OpDef.extra_out_dtypes`); the first mixed-dtype multi-output kernel — fused
  dropout (value + U8 mask) bit-exact vs the bespoke.
- **kernelgen — FUSED ARGSORT / TOP-K**: `SortOut::{Values,Indices,Both}` +
  `SortLimit::{Full,TopK}` on `Access::RowSort` — one kernel writes sorted
  values + i32 indices; per-row top-k / bottom-k with a runtime `k_out` and
  shrunk output.
- **kernelgen — IM2COL/unfold**: `Access::Im2Col` 2D conv-lowering structured
  gather (the column matrix `[N, C·kh·kw, oH·oW]`), byte-exact raw-bit gather.
- **kernelgen — F64 scalar launch params**: the scalar-param launch channel is
  dtype-aware (`double` params); f64 dropout the vehicle.
- **kernelgen — BLOCK-SCAN variants** (measured, `BitIdentical`/`Reassociated`):
  the cooperative Kogge-Stone scan extended to reverse + Max/Min + integer
  (I32/I64); FP Max/Min and integer scans are bit-identical to the serial base
  (a design-panel-proven NaN/signed-zero associative combiner).
- **kernelgen — PARTIAL-SELECT top-k** (measured, `BitIdentical`): a streaming
  tiled-bitonic top-k, O(k_in·log²k_out), smem bounded on `k_out` not `k_in` —
  the unique fast path for `k_in > 1024` large-vocab top-k (102.6× the rank base
  at top-10 of 50000, where the full-sort bitonic cannot run). Bit-identical to
  the full-sort top-`k_out` prefix (composition-lemma proven).

### Fixed

- **Build**: plain-shell test linking for `baracuda-optim` /
  `baracuda-transformer-engine` / `baracuda-ozimmu`.
- **partial-select top-k**: init-buffer pad-index saturated at `INT_MAX` (an
  in-contract integer overflow near the i32 `out_idx` cap would wrap a pad index
  negative and tie-beat a real — caught by adversarial review, memory-safety).

## 0.0.1-alpha.76 — 2026-07-08 (FKC contracts import through Fuel end-to-end; `baracuda-kernelgen` published — live JIT unblocked)

The release both Fuel sessions are waiting on: `baracuda-kernelgen` is now **on
crates.io** (so Fuel constructs `BaracudaSynthesizer` behind its `jit` feature —
the last gate on live end-to-end JIT), and a raw Baracuda-generated FKC bundle
now **imports through Fuel's real `import_bundle_str` end-to-end** (proven by a
two-repo acceptance harness against Fuel's actual importer, residual-divergence
list empty).

### Added

- **`baracuda-kernelgen` published** (was `publish = false`). The supported
  surface is the `seam` feature (the frozen `Synthesizer` impl); the generator
  internals are alpha-fluid — pin exact versions.
- **Bespoke NaN-propagating relu kernel family**
  (`unary_relu_propagating_fp.cu`, f32/f64/f16/bf16 + 16 launchers) for Fuel to
  rebind `OpKind::ReluElementwise` to, per Fuel's 2026-07-08 decision (relu =
  NaN-propagating, torch parity). On-device: bit-identical to the generated
  kernelgen relu across full 16-bit sweeps and 64M/32M f32/f64 sweeps,
  contiguous and strided, memcheck clean.

### Fixed

- **op_kind spellings** now match Fuel's dispatch table verbatim (29 mapped
  primitives; unmapped roots are honest misses — one unknown `op_kind` fails a
  WHOLE Fuel bundle). All 29 proven through Fuel's live importer. The Relu
  withhold (a NaN-semantics divergence caught by adversarial review) was lifted
  after Fuel pinned the propagating convention.
- **Bundle schema**: backend token casing; `## <kernel>` section headings via a
  shared `bundle()` assembler (Fuel silently drops headingless blocks);
  named accept inputs with truthful five-flag `LayoutSpec` maps;
  `dtype_rule: passthrough(in0)` (the old spelling silently dropped the output
  dtype from Fuel's binding key); `in_place: false` (the old value inverted
  Fuel's §4.6 semantics); the precision block re-emitted with only Fuel schema
  keys; cost as Fuel expression keys; `layout_guarantee` on outputs;
  `shape_rule: same_as(in0)`; `awkward_layout_strategy`.
- **Contracts can no longer lie** (adversarial review, 18 findings, 3 critical):
  cells with baked-broadcast or flipped operands are withheld (honest miss —
  the kernel facts are unspeakable in the current schema); strided cells now
  truthfully advertise `broadcast_stride0: accepted` (projecting Fuel
  `strided_input=true`); fused contracts are withheld from bundles (Fuel's
  `FusedOps` table has no matching names and an unknown name is bundle-fatal)
  while the JIT seam keeps shipping them as bare blocks (Fuel stores unparsed).

### Build

- CI: feature-gated surfaces (cuvs, kernelgen seam/nvrtc) now compiled + tested;
  workspace-wide `clippy -D warnings` green (~180 warnings swept); the fmt check
  pins `--style-edition 2024` explicitly with a one-time strict-2024 sweep.

## 0.0.1-alpha.75 — 2026-07-08 (window pooling + row sort — the IR-expansion ramp is COMPLETE)

The final two increments of the `baracuda-kernelgen` IR-expansion ramp (0a–#8). The
generator IR can now express the functionality of the bespoke kernel surface across
every access class the roadmap chartered: elementwise (+multi-output), reductions,
row-reductions, contraction, layout views, gather, scatter/histogram, prefix scan,
sliding-window pooling, and row sort/argsort. Both increments adversarially reviewed;
each review's confirmed findings fixed and mutation-verified.

### Added

- **Sliding-window pooling (`Access::Window`, ramp #7)** — max / min / sum / avg pool
  along the innermost axis with stride, dilation, and padding (both avg divisor
  policies), output axis downsampled. One-thread-per-output emitter; every input tap
  bounds-checked (a wrong caller `out_len` cannot OOB-read). 17 gates, each
  mutation-checked. On-device: 12 checks pass, four sanitizers clean, `max_pool` at
  **102% of the copy-bandwidth ceiling** (the bespoke fixed-window pools ride cuDNN —
  no math-matched sibling). AOT-only honest miss; keying additive.
- **Stable row sort / argsort (`Access::RowSort`, ramp #8)** — ascending/descending,
  always the stable `(key, original-index)` pair-sort (unique total order ⇒
  deterministic, base ≡ variant byte-identical); NaN-greatest ordering (torch
  convention) with NaN payloads and `-0.0` preserved raw-bit. Values-sort plus an
  I32-output argsort (mutually consistent). Base = any-`k` rank sort; variant =
  bitonic smem pair-sort (k ≤ 1024 launch precondition), **1.59× the bespoke stable
  `msort`** on device — and the audit surfaced that the bespoke stable tie-break is
  not input-order-preserving, while the generated argsort matches
  `torch.sort(stable=True)`. 212 device checks + four sanitizers clean. The
  adversarial review's headline catch (an `int` tie index that was also the load
  address — OOB past 2³¹ on the any-k base) is fixed (`long long` end-to-end) and
  re-validated on device.

### Fixed

- `baracuda-cuvs-sys`: all 74 feature-gated FFI items documented (the `cuvs` feature
  is never compiled in CI, so `missing_docs = deny` had never fired on it).

### Build

- Workspace `rustfmt.toml` pinning `edition`/`style_edition` 2024.

## 0.0.1-alpha.74 — 2026-07-04 (prefix scan + JIT-envelope conform)

Two changes; the seam surface bump unblocks Fuel's JIT adoption path.

### Added

- **Prefix scan (`Access::Scan`)** in `baracuda-kernelgen` — cumsum / cumprod /
  cummax / cummin along the innermost (contiguous) axis, with inclusive/exclusive
  and forward/reverse. A serial-fold **base** (`BitIdentical`, serves every
  dtype/direction incl. integer + reverse) plus a **forward FP Sum/Prod block-scan
  variant** (Kogge-Stone warp scan + cross-warp offset + cross-chunk carry,
  re-emitted headerless inline; `ReassociatedDeterministic`). AOT-only honest miss
  on the Fuel seam; keying additive (`baracuda-kernels-types` untouched). On-device
  (RTX 4070 sm_89): all correctness cases pass, four compute-sanitizer tools clean;
  the block-scan hits **227 GB/s** (copy-bandwidth ceiling, 43× the bespoke naive
  scan), the base 17×. Adversarially reviewed (7 minor findings, all fixed and
  mutation-verified: header-light `±inf` identity, exclusive-only warp shuffle,
  gate-rejection test coverage, reverse block-scan declined to the base pending
  device validation).

### Changed

- **`BaracudaSynthesizer` conforms to the frozen JIT envelope** (`fuel-kernel-seam`
  **0.10.2 → 0.10.3**). Fuel froze the §5.2 two-step handover and Baracuda now builds
  against it: `take_kernel` is a **trait** method (Fuel calls it via `&dyn
  Synthesizer`); `synthesize` returns the light `JitResponse::Synthesized {
  entry_point }`; `take_kernel` hands back the **envelope** `SynthArtifact` (the
  debug-only `.cu` source and the recipe are dropped — the re-fuse `pattern:` rides
  in the contract, `decompose` is the region Fuel holds); `ArtifactKind` is
  loadable-only `{Ptx|Cubin}` and a non-loadable stub returns `Declined` (no
  unloadable placeholder ever crosses the seam); the compile budget is read
  per-request from `JitRequest.budget`. The on-device nvrtc handover
  (`JitRequest` → `Synthesized` → `take_kernel` → real PTX artifact) passes
  end-to-end on sm_89. See
  [`docs/fuel-reply-jit-envelope-reconcile-2026-07-04.md`](docs/fuel-reply-jit-envelope-reconcile-2026-07-04.md).

## 0.0.1-alpha.73 — 2026-07-04 (kernel specialization — generated kernels replace the bespoke surface)

The kernel-specialization release. `baracuda-kernelgen` grows from an
elementwise emitter into a value-numbered-DAG IR + backend that generates
device kernels across the bespoke surface — reductions, contractions, layout
views, gather/scatter — each generated form measured against its bespoke
counterpart before it ships. The IR was then expanded increment-by-increment
(each one adversarially reviewed) until it can express the functionality in the
bespoke kernels, and the mixed-dtype dispatch key was wired to Fuel's Model-A
per-operand dtype tuple. No public API removals; the `seam`/`nvrtc` surfaces
are additive. See the `docs/fuel-*` channel for the Fuel-side agreements.

### Added — the specialization engine (items 01–10)

- **Value-numbered DAG IR** (`ScalarExpr` graph with consumer counts) — a shared
  subtree is emitted once, so multi-use bodies dedup instead of recomputing.
- **Layout `View` IR** as a recognition carrier — the generator recognizes a
  transposed/strided read and fuses `body(transpose(input))` into one pass
  (fused transposed reads beat the bespoke transpose+op by 3.6×–27× on-device).
- **Strided / multi-axis / keepdim reductions** — outer/middle/multi-axis
  geometry classified on the schedule; block-per-row cooperative last-axis
  reduction (4.4× on-device over the naive strided path); exact integer
  (`long long`) accumulation for integer reductions.
- **`Access::Contraction` (ORDER-3 / MatMul)** with a `ContractionKey` on the
  `StructureKey` and a split-K outer-axis reduction variant (go/no-go rematch vs
  cuBLAS: correctness exact, schedule 0.25× → 0.95× of cuBLAS after split-K).
- **Per-arch dispatch-table routing oracle** — the §7 vendor-exclusion mechanism
  (route a structure key to generated vs vendor per arch) with an automated
  variant-gate loop closed end-to-end.
- **Packed `half2`/`bf162` vectorization** for f16/bf16 elementwise, and a
  cross-pass materialization variant (Softmax's shared `exp` cached per row).

### Added — the IR-expansion ramp (increments 0a–#5, each adversarially reviewed)

- **0a** scalar-fn vocabulary broadening (18 unary + 6 binary intrinsics).
- **0b** comparison predicates + u8 mask output.
- **0c** integer compute dtypes + bitwise/logical ops.
- **0d** `Coord(axis)` leaf (the output coordinate as a computable value).
- **0e** reduction `Prod` + fused post-expr + heterogeneous output dtype.
- **#1 MULTI_OUTPUT** — N outputs from one shared body-DAG.
- **#2 RowReduce role generalization** — fused compound backwards (beats the
  bespoke backward by 95×–5976× where the reduction fuses the epilogue).
- **#3 layout/shape node wiring** — fused transposed reads (see View, above).
- **#4 GATHER** — data-dependent index-tensor reads (`ReadIndex::Indexed`).
- **#5 SCATTER + ATOMIC_HISTOGRAM** — data-dependent writes
  (`WriteIndex::ScatterIndexed`), with a `Nondeterministic` variant fidelity for
  atomic-accumulate scatter.

### Added — Model-A mixed-dtype dispatch key

- **`ElementKind::U32`** (additive, end of the enum — no discriminant shift; the
  `structure_key` token stays version 1 and pre-existing tokens are byte-identical).
  U32 is the Fuel-required index dtype (Fuel is U32-index at graph-build).
- **Per-operand dtype in the FKC `accept` block** — the dispatch key is the
  per-operand dtype tuple assembled from `accept.inputs[i].dtype`, emitted as
  Fuel's plural `dtypes: [..]` TensorDesc field (singular `dtype:` is silently
  dropped by Fuel's importer). Gather / index_select / embedding advertise a
  keyed contract with the verified Fuel `op_kind` spelling and an `oob_policy`
  field; scatter stays an honest miss pending operand-arity negotiation.
- U32 is an **index/address dtype only** — the CUDA backend rejects a U32 value
  plan for a non-indexed op (no u32 arithmetic path opens); on-device gather is
  bit-exact vs the i32 kernel with a clean memcheck.

### Fixed

- **`baracuda-forge`**: nvcc / `lib.exe` output is now piped into build errors
  (previously swallowed), and `cl.exe` is auto-discovered via `vswhere` — a
  silent-toolchain-failure fix for the MSVC build path.
- **e-graph soundness**: removed the unsound `x*0` fold and sign-gated the zero
  identities so NaN constants are never folded away; added exact div-by-pow2 and
  abs/relu rewrite rules; NaN-payload sweep over the Tier-A unary intrinsics.
- **Contract flops** are DAG-counted (a shared subtree charged once, not per use).
- **Split-K variant gate** hardened against flipped views, param bodies, and key
  shape; output-store injectivity guard restored on the general reduction path.

### Performance

- Block-per-row cooperative reduction: **4.4×** on-device over the naive
  last-axis path. Split-K contraction: **0.25× → 0.95×** of cuBLAS. Generated
  elementwise/norm/reduction paths sweep the bespoke audit (generated ≥ bespoke
  everywhere measured); three extract-the-delta perf extractions folded back into
  the generator (strength-reduced fold offsets **2.05×** on the general path; an
  int32 innermost counter puts the generated base at 169.6 vs 165.2 GB/s bespoke).

## 0.0.1-alpha.72 — 2026-06-29 (harden alpha.71 stream-ordered free)

Post-ship hardening of the alpha.71 stream-ordered free, from an adversarial
multi-lens review. Two fixes; no API change. Pull alpha.72 rather than alpha.71.
See [`docs/fuel-reply-stream-ordered-free-2026-06-29.md`](docs/fuel-reply-stream-ordered-free-2026-06-29.md)
(updated for alpha.72).

### Fixed

- **`DeviceBuffer::drop` no longer falls back to a *synchronous* free when the
  async free *call* errors** (driver + runtime). alpha.71's `Drop` dropped
  through to `cuMemFree` / `cudaFree` on any non-success from
  `cuMemFreeAsync` / `cudaFreeAsync`, not only on a missing symbol — and a
  synchronous free of a stream-ordered pool allocation that may still be
  referenced by pending stream work is the exact use-after-free the feature
  exists to prevent. Now the synchronous fallback fires **only** when the async
  free *symbol* is unavailable (genuine pre-CUDA-11.2); on an async-free *call*
  error the block is **leaked** (reclaimed at context/device teardown) rather
  than synchronously freed. The happy path is unchanged — the async free
  succeeds and returns — and the error path is unreachable on a healthy stream.
  This is a defensive-path fix: the common single-stream dispatch path never
  reached the old fallback.

### Documentation

- **Documented the single-origin-stream precondition** on `new_async` /
  `zeros_async` (both crates) and in the Fuel reply. The stream-ordered `Drop`
  free is ordered only against work on the buffer's **origin stream**; a buffer
  consumed on a *different* stream needs an explicit event/cross-stream
  dependency (or `free_async` on a suitably-ordered stream) before it drops.
  Replaces the unqualified "safe by construction" wording. (CUDA's standard
  stream-ordered-allocator contract — stated so a future multi-stream scheduler
  doesn't silently reintroduce a UAF.)
- Corrected the `Drop` rustdoc/comments: the synchronous fallback is the
  pre-11.2 + sync-allocated-buffer path, not a catch-all for async-free errors.

## 0.0.1-alpha.71 — 2026-06-29 (Fuel — stream-ordered free for async dispatch)

Stream-ordered *free* to match the existing stream-ordered *alloc*, so Fuel's
Step E A3 async dispatch can drop the per-op `stream.synchronize()` without
risking use-after-free on scratch / output buffers. A `DeviceBuffer` allocated
via a `*_async` constructor now **retains its origin stream** and reclaims via
`cuMemFreeAsync` / `cudaFreeAsync` on [`Drop`] — the free is enqueued *after*
any kernel still using the buffer, so it's safe by construction with no host
sync and no executor-side lifetime guard. Synchronous callers are completely
unchanged. See
[`docs/fuel-reply-stream-ordered-free-2026-06-29.md`](docs/fuel-reply-stream-ordered-free-2026-06-29.md).

### Added

- **`DeviceBuffer::zeros_async(.., len, stream)`** on both `baracuda-driver`
  and `baracuda-runtime` — stream-ordered allocate-and-zero (`*MallocAsync` +
  `*MemsetAsync` on `stream`). The async counterpart of `zeros`, intended for
  **output buffers** so they free stream-ordered like workspaces do. Retains
  the stream; reclaims via the async free on `Drop`.

### Changed

- **`DeviceBuffer` allocated via `new_async` / `zeros_async` now frees
  stream-ordered on `Drop`** (was: synchronous `cuMemFree` / `cudaFree`
  regardless of how it was allocated). The buffer retains its origin stream
  (a cheap `Arc` clone — `Stream` is `Arc`-backed) and `Drop` enqueues
  `cuMemFreeAsync` / `cudaFreeAsync` on it; freed blocks return to the
  device's stream-ordered memory pool for reuse across a realize chain
  (peak VRAM ≈ live working set, no repeated real `cuMemAlloc`). If the async
  free symbol is unavailable (pre-CUDA-11.2), `Drop` falls back to the
  synchronous free.
  - **No behavior change for synchronous callers.** Buffers from `new` /
    `zeros` / `from_slice` retain no stream and free synchronously exactly as
    before.
  - The explicit consuming `free_async(self, stream)` still works for callers
    that want to free at a specific point rather than at scope exit.

### Verified

- New GPU-gated smoke tests on an RTX 4070 (sm_89, driver 610.47, CUDA 11.2+):
  `async_drop_is_stream_ordered` (driver) launches `vector_add`, then drops the
  two `new_async` input buffers **before** any synchronize — the async free is
  ordered after the kernel, so the 16K-element result is bit-exact;
  `zeros_async_is_zeroed` (driver) and `zeros_async_and_implicit_stream_drop`
  (runtime) confirm `zeros_async` zeroing and implicit-`Drop` reclaim. Existing
  `async_alloc_roundtrip` / `async_alloc_round_trip` (explicit `free_async`)
  still green. (compute-sanitizer not run — unavailable on the local box; the
  ordered-free correctness is exercised functionally by the pending-read test.)

## 0.0.1-alpha.70 — 2026-06-28 (Fuel — device-load telemetry aliases)

Convenience aliases for Fuel's `DeviceLoadSelector` (Step E, Phase B2). Both
underlying reads already shipped in alpha.69 (`Stream::is_complete` /
`nvml::Device::utilization`); this adds thin, read-only aliases matching the
shape Fuel's load-balancer wants. No new dependency, no hot-path impact. See
[`docs/fuel-reply-device-load-telemetry-2026-06-28.md`](docs/fuel-reply-device-load-telemetry-2026-06-28.md).

### Added

- **`Stream::is_idle() -> Result<bool>`** on both `baracuda-driver` and
  `baracuda-runtime` — a load-probe alias of `is_complete()` (wraps
  `cuStreamQuery`/`cudaStreamQuery`; `Ok(true)` idle, `Ok(false)` busy).
  Read-only: never synchronizes the stream or perturbs scheduling. Reflects
  only this process's submissions to the stream.
- **`nvml::Device::gpu_utilization_percent() -> Option<u8>`** — GPU core
  busy-percent in the `Option<u8>` shape a load balancer wants, an honest
  `None` (never a fabricated value) when the driver can't answer. Convenience
  wrapper over `utilization()`. Captures *cross-process* device contention
  (the signal Fuel can't measure itself); pairs with the existing
  `compute_processes()`/`graphics_processes()` for an SM proxy.

## 0.0.1-alpha.69 — 2026-06-20 (Fuel — Profile v1 kernel-seam: handshake + structure_key + FKC generator)

Kernel-specialization / Baracuda↔Fuel **Profile v1** release. The cross-project
kernel-seam contract (FDX tensor ABI + FKC kernel contracts + telemetry + a
connect-time version handshake) was ratified by all parties (Fuel, Baracuda,
Vulkane); this ships Baracuda's conforming surface, reconciled against the FKC
fusion-patterns spec rev 4.

### Added

- **`baracuda-seam`** (new crate) — the Profile v1 kernel-seam handshake: the
  frozen 56-byte `#[repr(C)] SeamHello` negotiation envelope plus the out-param
  C-ABI entry point `baracuda_seam_hello()` that Fuel reads to negotiate a seam
  profile and capability set (Kernel-Seam Interop §3.1/§3.5). Capability bits
  reuse the FDX `BackendProbe` tokens.
- **`structure_key` + `structure_key_token`** (`baracuda-kernels-types`) — the
  canonical input/output layout-class join token, computed once by Baracuda from
  a minimal `OperandDesc` projection and called by Fuel for triple duty
  (telemetry tagging, FKC admissibility predicate, runtime dispatch), with a
  lossless string codec. `structure_key_token` is the one-call cross-boundary
  entry point.
- **`baracuda-kernelgen`** (dev tool, `publish = false`) — the AOT
  kernel-specialization generator: an abstract op IR lowered to CUDA, three
  schedules (vectorized `float4`/`double2` / scalar / strided+broadcast),
  f16/bf16/f32/f64 dtype breadth, and full **FKC contract emission** (provider
  front-matter + `accept`/`return`/`op_params`/`caps`/`cost`/`precision`/
  `determinism` + a declarative `pattern:` block with `extract:` for scalar
  params) plus the `link_registry` roster. Go/no-go validated at **2.03×** on
  sm_89; emitters audited conformant against FKC fusion-patterns rev 4 (incl. a
  GELU-flavor fix — exact-erf kernels now emit `GeluErf`, not the tanh-name
  `Gelu`).

### Fixed

- **`StructureKey.rank`** is the raw iteration rank (the widest operand rank),
  renamed from `eff_rank` so the field no longer implies the (deferred)
  contiguous-axis collapse.

## 0.0.1-alpha.68 — 2026-06-15 (Fuel — CUDA 13.3 / CCCL MSVC build fix)

Build-system release driven by the Fuel team's CUDA 13.3 upgrade. No kernel
or public API surface changes; the existing CUTLASS GEMM kernels were
verified end-to-end (102 GPU smoke tests each) against CUTLASS v4.2.0 and
4.5.2 on driver 610.47.

### Fixed — build

- **CUDA 13.3 CCCL traditional-preprocessor hard error.** CUDA 12.5+ / 13.x
  bundle a CCCL whose `<cuda/std/__cccl/preprocessor.h>` raises a hard
  `#error` (MSVC `C1189`) under cl.exe's legacy "traditional" preprocessor,
  breaking every CCCL-using `.cu` (CUTLASS, cub/thrust). `baracuda-forge` now
  passes `-Xcompiler /Zc:preprocessor` on MSVC hosts (in both `build_lib` and
  `build_ptx`), switching cl.exe to the standard-conforming preprocessor —
  NVIDIA's recommended fix. All forge consumers inherit it.
- **`cudart.lib` link-search.** `baracuda-cutlass-kernels-sys` now emits a
  `rustc-link-search` for the detected CUDA toolkit lib dir, so a binary that
  depends on `baracuda-cutlass` (but not `baracuda-kernels-sys`) links
  `cudart` without requiring the CUDA lib dir to already be on `LIB`.
- **`baracuda-build` Windows lib-dir detection.** `pick_lib_dir` returned the
  bare `lib` rather than `lib\x64` on Windows (where the import libraries
  live), contradicting `CudaInstall::lib`'s documented contract and making
  `CudaToolkit::detect()` disagree with `from_nvcc_path()`. It now prefers
  `lib\x64`; Linux ordering is unchanged. Regression-tested.

### Added

- **CUTLASS version guard** in `baracuda-cutlass-sys`: a `CUTLASS_DIR`
  checkout below v4.2.0 is rejected with a hard error, above v4.5.2 warns
  (the curated kernels are verified across the 4.2–4.5.2 range), and an
  in-range version is confirmed.

## 0.0.1-alpha.67 — 2026-06-10 (Phase 74 — Fuel dense FP GEMM + reduce-to closure)

### Added — kernels

- **Phase 74 — Fuel dense-FP-GEMM + reduce-to facade closure** (Fuel
  ask 2026-06-10; full consumer reply in
  `docs/fuel-reply-fp-gemm-reduce-to-2026-06-10.md`):
  - NEW `gemm_dense_cublas_facade` in `baracuda-kernels-sys` — **12
    flat C symbols** `baracuda_kernels_gemm_dense_{f32, f64, f16,
    bf16}_{run, can_implement, workspace_size}`, cuBLAS-backed
    (`cublasGemmEx` / newly-declared `cublasGemmStridedBatchedEx`).
    Row-major dense GEMM with runtime layout tag (RRR / RCR / CRR),
    flexible leading dims, strided-batch folded into the base symbol
    (`stride 0` = broadcast). f16/bf16 accumulate in f32; f32 is true
    IEEE binary32 (default math mode, NOT TF32); f64 = `COMPUTE_64F`.
    Lock-free context-keyed cuBLAS handle pool (hot-path deviation
    from the transient-handle facade convention; documented hazards).
  - NEW `DenseGemmPlan<T>` typed plan over the same symbols, with
    buffer-bounds (`BufferTooSmall`) validation at the safe layer and
    plan-local `DenseGemmLayout` (CRR has no `LayoutSku` variant yet).
  - NEW `ReduceToPlan<T, N>` facade (`{Sum, Max, Min, Prod} × {f32,
    f64, f16, bf16}`) + `ReduceToOp` in `baracuda-kernels-types` over
    the existing Phase 31/37 `reduce_*_to_*` FFI symbols (no new
    kernels — closes the sys-only facade gap that hid the capability
    from Fuel's alpha.66 audit).
  - `UnaryKind::Step` now dispatches through `UnaryPlan` (contig +
    strided × 4 dtypes; the kernels shipped in Phase 31).
  - Gelu flavor disambiguation docs on `unary_gelu_*` /
    `unary_gelu_erf_*` (bit-identical erf-exact twins) /
    `unary_gelu_tanh_*` (tanh approximation) and
    `UnaryKind::{Gelu, GeluTanh}`.
  - Tests: `dense_gemm_smoke` (3 layouts vs f64 CPU ref, padded lds,
    β-accumulate, strided batch + broadcast, all dtypes, direct-FFI
    binding-table shape, rejection matrix), `reduce_to_plan_smoke`,
    `unary_step_smoke` plan-level extension. All green on RTX 4070.

## 0.0.1-alpha.66 — 2026-06-07 (driver VRAM introspection)

- **`Device::vram_info` / `vram_free` / `vram_total`** wrapping
  `cuMemGetInfo_v2` (Fuel ask — powers Fuel's `BackendRuntime` and
  VRAM-pressure backend selection). Current-context caveat
  documented. See `ROADMAP.md` for detail.

## 0.0.1-alpha.65 — 2026-06-05 (Phases 72 + 73)

- **Phase 73 — Cross-impl bench follow-ups**: `FlashDecodingPlan`
  (split-K seq_q=1 decode, 12-16× vs the pre-fix path),
  warp-cooperative QKᵀ (2-2.6× at K≤2048), concat + reduce perf
  closures, `fa2` promoted to a default feature. See `ROADMAP.md`
  for the per-item detail.

### Added — kernels

- **Phase 72 — Strided FFI siblings for normalizer + shape ops**: closes
  the long-standing ROADMAP item. Authored explicit `_strided_run` /
  `_strided_can_implement` siblings for 7 op families:
  `rms_norm`/`layer_norm`/`softmax`/`log_softmax` (FW + BW × 4 dtypes
  each) + `flip`/`roll`/`permute` (FW × 4 dtypes each). **88 new FFI
  symbols** (44 `_strided_run` + 44 `_strided_can_implement`). Each
  sibling routes to the same underlying launcher as the non-strided
  `_run` — the existing exports already accepted stride arrays and the
  C kernels already honored them; the sibling exists so callers
  building explicit dispatch tables can pick the strided path by name
  (matches the Phase 14/18 convention for binary / unary-param ops).
  Test investment: 7 new direct-FFI smoke tests in
  `tests/strided_siblings_ffi_smoke.rs`, all 7/7 pass on RTX 4070 with
  non-contig fixtures (stride-2 over 2x-padded buffer for the
  norm/softmax/roll ops; transposed view for flip + permute).

## 0.0.1-alpha.64 — 2026-06-03 (Phases 64 + 65 + 66-prep + Tier-2 docs sweep)

The biggest single release in the alpha series — closes Phases 64–71
work + the full release-readiness audit (`_can_implement` companion
contract + Tier-2 cross-crate docs + workspace-wide `missing_docs`
sweep).

### Added — kernels

- **Phase 64 — In-place aliasing contract docs (extended)**: documented
  `x_ptr == y_ptr` safety for Cast / Where / Triu / Tril / Fill /
  Activation BW; explicitly marked Flip / Roll / Permute / RoPE as **not**
  in-place safe (write-before-read by construction).
- **Phase 65a — Reusable SMEM kernel helpers**: new
  `baracuda_smem_row_stager.cuh` (`smem_stage_row`, strided variants,
  `smem_budget_for_arch`) and `baracuda_smem_reduce.cuh`
  (`warp_reduce_{sum,max,min}_f32` + `f64`, cross-warp aggregation
  via `BARACUDA_MAX_WARPS = 32`). Lifts the SMEM staging pattern out
  of `baracuda_moe.cuh` into a reusable kernel-helper library at
  `crates/baracuda-kernels-sys/kernels/include/`.
- **Phase 65b — RMSNorm SMEM staging + in-place**: SMEM-staged
  `rms_norm_smem_kernel<T>` enables `y_ptr == x_ptr` aliasing for
  f32/f16/bf16. 3 in-place proof tests bit-exact vs non-aliased reference.
- **Phase 65c — LayerNorm / Softmax / LogSoftmax SMEM staging +
  in-place**: same pattern, three new `*_smem_kernel<T>` + dispatcher
  pairs. 9 in-place proof tests.
- **Phase 65d — BN / GN / IN in-place contract** (no kernel changes):
  documented the existing two-stage kernel (stage-1 stats-only,
  stage-2 single-read-then-write per cell) as in-place safe by
  construction. **f64 IS in-place safe here**. 12 in-place proof tests
  across f32/f16/bf16/f64.
- **Phase 65d-ext — f64 in-place SMEM normalizers**: added
  `block_reduce_{sum,max,min}_f64` to `baracuda_smem_reduce.cuh`;
  RMSNorm / LayerNorm / Softmax / LogSoftmax SMEM kernels gained `<f64>`
  specializations. f64 now uses the in-place fast path under the same
  contig + last-axis preconditions as f32. 4 new f64 in-place proof tests
  (16/16 in-place tests green on RTX 4070).
- **Phase 66-prep — `_can_implement` companion fanout (~2030 symbols)**:
  baracuda's stated FFI convention is one host-side validator per `_run`
  symbol. Audit found ~2026 `_run` symbols missing a `_can_implement`.
  Closed completely across four rounds — 59 C-side macros extended,
  ~2030 Rust extern declarations added. Every `_run` FFI symbol in
  `baracuda-kernels-sys` + `baracuda-transformer-engine-sys` now has
  a `_can_implement` validator.

### Added — new optional sibling crates (Phases 66-71, shipping with this alpha)

- **`baracuda-flashinfer`** + **`baracuda-flashinfer-sys`** — Phase 46+66.
  Safe wrapper for FlashInfer's inference-serving kernels (paged-KV decode
  + prefill + ragged prefill + append, cascade LSE-merge, sort-free top-K
  / top-P / min-P sampling, per-row sampling, FP8 KV decode, spec-decode
  via ChainSpeculativeSampling). Feature-gated behind `flashinfer` on
  `baracuda-kernels`.
- **`baracuda-nvshmem`** + **`baracuda-nvshmem-sys`** — Phase 69.
  Host-side NVSHMEM wrapper: symmetric-heap RDMA (sibling to NCCL's
  collectives). Feature-gated behind `nvshmem`.
- **`baracuda-nvimagecodec`** + **`baracuda-nvimagecodec-sys`** — Phase 70.
  Unified GPU image-codec wrapper (JPEG / JPEG2000 / PNG / TIFF / WebP);
  supersedes the standalone `baracuda-nvjpeg`.
- **`baracuda-cuvs`** + **`baracuda-cuvs-sys`** — Phase 71.
  RAPIDS cuVS vector-search wrapper (IVF-Flat + brute-force k-NN; L2 /
  cosine / inner-product distance). Linux-only.
- **`baracuda-tensorrt` shim feature** — Phase 68.
  Bundled vtable-dispatch C++ shim required to actually call TensorRT
  (NVIDIA ships no flat C ABI). Gated behind the `shim` feature.

### Documentation

- **README** — cargo-features table 4 → 18, workspace-layout +14 missing
  crates, ARCHITECTURE rewritten Phase 14-71, OP-MATRIX swept 11
  false-deferred claims (Pool family, Segment BWs, EmbeddingBag Max,
  BatchedOrmqrWy complex, GgufMmvq f16/bf16, paged FlashAttention, etc.),
  ROADMAP updated.
- **Per-crate READMEs** authored for 8 crates that previously fell back
  to the workspace README on crates.io (flashinfer{,-sys}, megatron,
  optim, ozimmu{,-sys}, transformer-engine{,-sys}).
- **5 stale crate-level `//!` docs** rewritten — kernels, nvjpeg, cudnn,
  cublas, nccl all had Phase-0/Phase-5-era "v0.1 covers X" preambles
  that didn't match the alpha.63 surface; now accurate.
- **13 `-sys` crate "safe-wrapper" pointer** added — every `-sys` crate
  now points downstream readers at its safe-wrapper sibling.
- **~35 executable doctests** added across cublas / cudnn / cufft /
  cusolver / cusparse / cutensor / cuvs / curand / runtime / types.
- **8 examples/ directories** created for the major safe wrappers
  (cublas/cudnn/cufft/cusolver/curand/nccl/runtime/driver).
- **`[workspace.lints]` clause** added to root Cargo.toml + per-crate
  `[lints] workspace = true` propagation (70 crates).
- **Workspace-wide `missing_docs` sweep** (~6900 one-line docs) closed
  every undocumented public item across 13 `-sys` crates. Workspace lint
  promoted from `warn` → `deny` as a regression guard.

### Added — tests

- 12 in-place proof tests for BN/GN/IN (Phase 65d) + 4 new f64 in-place
  proof tests for RMSNorm/LayerNorm/Softmax/LogSoftmax (Phase 65d-ext).
- `bincount_smoke.rs` (4 tests) — direct-FFI coverage for
  `baracuda_kernels_bincount_{i32,i64}_run`.
- `paged_kv_append_smoke.rs` (4 tests) — direct-FFI lifecycle for
  FlashInfer paged-KV append.
- `baracuda-cudf/tests/cudf_smoke.rs` (9 tests) — closed the zero-test
  red flag.
- `baracuda-types-derive` (21 tests, 10 integration + 11 unit) — proc-macro
  positive paths + rejection paths.

### Fixed

- 95 `clippy -D warnings` CI failures (stylistic + unused imports + Rust
  2024 `unsafe_op_in_unsafe_fn` migration via `cargo fix`).
- 10 GGUF-K-block test names allowed via `#[allow(non_snake_case)]`.
- `map_status_pub` + `alibi_dispatch` `#[allow(dead_code)]` for
  feature-gated callers (`xformers_sparse24` / `fa2` respectively).
- `FusedLinearCrossEntropyBackwardPlan::desc` `#[allow(dead_code)]` (BW
  pass is a pure dy-scalar broadcast that doesn't read the descriptor).
- Pre-existing `baracuda-cuvs` crate-level doctest used wrong Result error
  type — would have always failed if anyone ran it; fixed to
  `Box<dyn std::error::Error>`.

---

## Unreleased (queued for next alpha)

(Empty — alpha.64 just shipped.)

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
