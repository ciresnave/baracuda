# Kernel helper library — index + coordination

baracuda's CUDA kernels share a small library of reusable `.cuh` headers
under [`crates/baracuda-kernels-sys/kernels/include/`](../../crates/baracuda-kernels-sys/kernels/include/).
This document is the canonical index of what exists, what's planned, and
the conventions all helpers follow.

**Read this before adding a new helper.** Multiple parallel sessions may
be building helpers simultaneously; consulting this index prevents
duplicate work + diverging naming.

## How the helper library works

A "helper header" is a `.cuh` file containing pure-`__device__`
inline/template functions usable from any CUDA kernel translation unit.
Helpers don't define kernels themselves — they encapsulate reusable
sub-kernel patterns (cooperative loads, block reductions, dtype
promotion, coordinate unraveling, atomic merges, etc.).

Properties of a good helper:

- **Inline-able** — every public function is `__device__ __forceinline__`
  (or a template; templates are implicitly inlinable).
- **Header-only** — no separate `.cu` translation unit per helper. The
  header lives or dies with the `.cu` files that include it.
- **Namespace-isolated** — under `namespace baracuda { ... }` (or a
  sub-namespace like `baracuda::reduce`).
- **Zero side effects on inclusion** — adding a new helper header to
  `kernels/include/` does NOT trigger any kernel rebuild. nvcc only
  recompiles `.cu` files that `#include` the new header.
- **Self-documenting** — the file's top comment explains what pattern
  the helper captures + when to use it. Future kernel authors browse
  the directory listing to find the right helper.

## Naming convention

| Pattern | Example |
|---|---|
| `baracuda_<category>.cuh` | `baracuda_smem_row_stager.cuh` |
| Specific subsystem files keep existing prefix | `baracuda_norm.cuh`, `baracuda_affine.cuh` (these define KERNELS, not just helpers) |

Helper categories so far:
- `smem_*` — cooperative SMEM operations
- `dtype_*` — dtype promotion + conversion
- `coord_*` — index-to-coordinate decomposition
- `block_*` — cross-block coordination patterns (atomics, etc.)
- `vec_*` — vectorized loads

## Existing helpers

### Phase 65a (committed `763bec0`, 2026-06-01)

| File | What it provides |
|---|---|
| [`baracuda_smem_row_stager.cuh`](../../crates/baracuda-kernels-sys/kernels/include/baracuda_smem_row_stager.cuh) | `smem_stage_row<SmemT, GlobalT>(smem, global, n)` + strided variant + `smem_unstage_row` symmetric writeback + `smem_budget_for_arch(cc)` + `smem_stage_max_n(...)` runtime dispatch helper |
| [`baracuda_smem_reduce.cuh`](../../crates/baracuda-kernels-sys/kernels/include/baracuda_smem_reduce.cuh) | `warp_reduce_{sum,max,min}_f32` (warp-shuffle) + `block_reduce_{sum,max,min}_f32(x, warp_buf)` (cross-warp via SMEM scratch) + `BARACUDA_MAX_WARPS = 32` constant |

These two seed the library. Phase 65b will be the first user of them
(retrofitting the normalizer family).

### Phase 67a (committed `7379753`, 2026-06-01)

| File | What it provides |
|---|---|
| [`baracuda_dtype_promote.cuh`](../../crates/baracuda-kernels-sys/kernels/include/baracuda_dtype_promote.cuh) | Three dtype-promotion lanes (all `__device__ __forceinline__`, in `namespace baracuda`): f32 lane `load_as_f32<T>` / `store_from_f32<T>` (lifted from `norm.cuh`'s `load_as_acc`/`store_from_acc`; specializations for `__half`/`__nv_bfloat16`); f64 lane `load_as_f64<T>` / `store_from_f64<T>`; i64 lane `load_as_i64<T>` / `store_from_i64<T>` (integer accumulator, sign/zero-extend on load + two's-complement modular narrow on store, matching `baracuda_reduce_int.cuh`'s `WidePolicy` contract). Pure templates — no standalone test; verified at first retrofit. |

### Phase 67b (branch `phase67b-coord-unravel`, 2026-06-01)

| File | What it provides |
|---|---|
| [`baracuda_coord_unravel.cuh`](../../crates/baracuda-kernels-sys/kernels/include/baracuda_coord_unravel.cuh) | The linear-index → multi-coordinate → byte-offset loop duplicated across ~25 strided kernels (flip, roll, permute, affine_strided, where_strided, ternary_clamp_strided, rms_norm helpers, indexing, …). Four `__device__ __forceinline__` templates under `namespace baracuda::coord`: `unravel_offset_1` (single stride array, returns offset), `unravel_offsets_2` (input+output), `unravel_offsets_3` (binary-op+output), `unravel_offsets_4` (ternary/where+output — 3 call sites justify it). Functions are **templated over `Shape`/`Stride`** so callers pass their own in-scope `DimsI32`/`DimsI64` (deduced at call site) — sidesteps the ODR clash from the 8 existing per-subsystem `Dims*` definitions, which are deliberately left unmodified. Bit-for-bit identical arithmetic to the hand-written loops (including the `s == 0` empty-axis guard); honors the stride-0 broadcast convention transparently. Pure templates — verified by host syntax-check, no standalone test. |

### Phase 67c (committed on `phase67c-block-atomic`, 2026-06-01)

| File | What it provides |
|---|---|
| [`baracuda_atomic.cuh`](../../crates/baracuda-kernels-sys/kernels/include/baracuda_atomic.cuh) | (Pre-existing, Phase 11.3 — now indexed.) `baracuda::atomic::add<T>` — native `atomicAdd` for f32/f64/i32/i64/u32/u64; 32-bit `atomicCAS` loop for `__half` / `__nv_bfloat16`. Already consumed by `baracuda_indexing.cuh` + `baracuda_segment.cuh`. |
| [`baracuda_block_atomic.cuh`](../../crates/baracuda-kernels-sys/kernels/include/baracuda_block_atomic.cuh) | Cross-block atomic-merge family. `#include`s + re-exports `baracuda_atomic.cuh`'s `add`, then adds `baracuda::atomic::max<T>` / `min<T>` (native `atomicMax`/`atomicMin` for int/uint/ll/ull, `atomicCAS`-bit-trick for f32/f64, 16-bit-in-32 CAS for half/bf16) and `mul<T>` (always CAS — no native atomic multiply; f32/f64/half/bf16 + int/uint/ll/ull). Validated add/max/min/mul × all 8 dtypes on RTX 4070 (sm_89); compiles clean on sm_80/sm_89/sm_90a. |

### Phase 67d (branch `phase67d-smem-scan`, 2026-06-01)

| File | What it provides |
|---|---|
| [`baracuda_smem_scan.cuh`](../../crates/baracuda-kernels-sys/kernels/include/baracuda_smem_scan.cuh) | `baracuda::scan::warp_scan_inclusive_{sum,max,min}_f32` + `warp_scan_inclusive_sum_f64` (Kogge-Stone `__shfl_up_sync`, 5 rounds) + `block_scan_inclusive_{sum,max,min}_f32(value, warp_buf)` + `block_scan_exclusive_sum_f32` + `block_scan_inclusive_sum_f64` / `block_scan_exclusive_sum_f64`. Cross-warp aggregation via a 32-slot SMEM scratch (`warp_buf`) — same buffer layout as `block_reduce_*` in `baracuda_smem_reduce.cuh`, so a kernel that both reduces and scans can share one scratch. One-value-per-thread O(log N) prefix scans for the scan-kernel family (cumsum/cumprod/cummax/cummin/logcumsumexp) + online-softmax running stats. **Limitations:** one value per thread, and `blockDim.x` must be a multiple of 32 (the per-warp total is read from lane 31, so a partial final warp leaves its slot unwritten). For multi-`ITEMS_PER_THREAD` block scans use `cub::BlockScan` directly (see sparsemax_fp.cu); for device-wide scans use `cub::DeviceScan`. Pure templates — nvcc sm_89 compile + functional `[1,1,…]→[1,2,3,…]` / inc-max / inc-min checks pass on RTX 4070. |

### Phase 67f (branch `phase67f-hmath`)

| File | What it provides |
|---|---|
| [`baracuda_hmath.cuh`](../../crates/baracuda-kernels-sys/kernels/include/baracuda_hmath.cuh) | Uniform half-precision math facade in `namespace baracuda::hmath`. Scalar `hadd`/`hsub`/`hmul`/`hfma` overloaded for `__half` + `__nv_bfloat16` (native intrinsics on sm_53+/sm_80+, f32-promotion fallback otherwise); transcendentals `hexp`/`hlog`/`htanh`/`hsigmoid`/`hsqrt` (always f32-stable path); packed `hadd2`/`hmul2`/`hfma2` for `__half2` + `__nv_bfloat162` (native `*2` SIMD, component-wise f32 fallback) |

### Phase 67e (branch `phase67e-smem-tile`, 2026-06-01)

| File | What it provides |
|---|---|
| [`baracuda_smem_tile.cuh`](../../crates/baracuda-kernels-sys/kernels/include/baracuda_smem_tile.cuh) | `baracuda::tile::SmemTile2D<T, ROWS, COLS, PAD=1>` — bank-conflict-padded 2D SMEM tile (`data[ROWS][COLS+PAD]`, `operator()` hides padding, `rows()`/`cols()`/`pad()`/`bytes()` constexpr accessors) + cooperative `tile_load_row_major<BLOCK_THREADS>` (syncs) / `tile_store_row_major<BLOCK_THREADS>` (no trailing sync) / `tile_load_col_major<BLOCK_THREADS>` (transposed K-operand load, syncs). f32 fast path issues coalesced `float4` global transfers for row-major load/store when `COLS%4==0` and the stride is 4-aligned; scalar fallback otherwise. **Limitation:** targets SIMT-style scalar-column-stride tiles — does NOT cover the dense `[M_TILE][K_TILE]` warp-MMA tiles in `baracuda_{int8_rrr_sm80,fp8_*_sm89,int4_*_sm89,bin_*_sm89}.cuh` (they read SMEM into `mma.sync` fragments via 4-byte `uint32_t` loads dictated by the hardware fragment layout, not scalar column strides — and a 1-byte `+1` pad would misalign those reads). **Padding vs. swizzle:** padding is the default (any width/dtype, preserves row contiguity); an XOR-swizzled layout wastes zero SMEM but only wins when a profile shows SMEM capacity throttling occupancy *or* same-tile dual-orientation access — add it then as a *layout-policy template param*, not a parallel type (prefer CUTLASS `cute::Swizzle` for warp-MMA tiles). Header docstring carries the full decision criteria. Pure templates — compile-verified (nvcc sm_89), no standalone test; validated at first retrofit. |

### Pre-existing kernel-author helpers (in scope to lift if duplicated elsewhere)

- `load_as_acc<T>` / `store_from_acc<T>` in [`baracuda_norm.cuh`](../../crates/baracuda-kernels-sys/kernels/include/baracuda_norm.cuh) — dtype promotion to f32 for compute. The shared successor (`load_as_f32`/`store_from_f32` in `baracuda_dtype_promote.cuh`, Phase 67a) now exists; norm.cuh's local copy stays until that kernel is the retrofit target. Same for the local `load_as_f32`/`store_from_f32` copies in `baracuda_attention.cuh` / `baracuda_sdpa.cuh` / others — retrofit one kernel at a time.
- `warp_reduce_sum` / `warp_reduce_max` in [`baracuda_moe.cuh`](../../crates/baracuda-kernels-sys/kernels/include/baracuda_moe.cuh) — already lifted into `baracuda_smem_reduce.cuh` as of Phase 65a. The moe.cuh copy can stay until moe.cuh is the next retrofit target.

## Planned helpers (parallel-buildable)

Each row maps to a session prompt under [`docs/sessions/`](../sessions/).
The prompts are self-contained — a new session can pick one up and run.

| Helper | Session prompt | Status |
|---|---|---|
| `baracuda_dtype_promote.cuh` | [`kernel-helper-dtype-promote.md`](../sessions/kernel-helper-dtype-promote.md) | ✅ done (Phase 67a) |
| `baracuda_coord_unravel.cuh` | [`kernel-helper-coord-unravel.md`](../sessions/kernel-helper-coord-unravel.md) | ✅ done (Phase 67b) |
| `baracuda_block_atomic.cuh` | [`kernel-helper-block-atomic.md`](../sessions/kernel-helper-block-atomic.md) | ✅ done (Phase 67c) |
| `baracuda_smem_scan.cuh` | [`kernel-helper-smem-scan.md`](../sessions/kernel-helper-smem-scan.md) | ✅ done (Phase 67d) |
| `baracuda_smem_tile.cuh` | [`kernel-helper-smem-tile.md`](../sessions/kernel-helper-smem-tile.md) | ✅ done (Phase 67e) |
| `baracuda_hmath.cuh` | [`kernel-helper-hmath.md`](../sessions/kernel-helper-hmath.md) | ✅ done (Phase 67f) |

## Adding a new helper — checklist for sessions

1. **Read this index.** Confirm the helper isn't already listed.
2. **Pick a branch name** — `phase67<x>-<short-name>` where `<x>` is a letter (`a`/`b`/`c`/…) so multiple parallel helper sessions don't collide on phase numbers.
3. **Create the file** at `crates/baracuda-kernels-sys/kernels/include/baracuda_<name>.cuh`.
4. **Follow the conventions** above (namespace, `__forceinline__`, header-only, file-top docstring).
5. **Add a host-only sanity test if possible.** Helpers that are pure templates can't be tested standalone (no Rust dispatch). Helpers with host-callable functions (like `smem_budget_for_arch`) CAN be exposed via FFI as a `#[no_mangle] extern "C"` wrapper and unit-tested. Use judgment.
6. **Update this index.** Move the helper from "planned" to "existing" with a one-line description + commit hash.
7. **Commit on your branch.** No version bump (accumulating for next release). No publish.
8. **Push the branch.** Don't merge to main yet — user will review parallel branches and merge in order.

## Coordination protocol for parallel sessions

- Each helper session works on its own git branch.
- Each session updates this index doc as part of its commit (move "planned" → "existing").
- If two parallel sessions both edit this index, the second one to merge has to rebase + manually resolve the index conflict. Trivial.
- If two sessions discover they're building the same helper (one named it differently in scope creep), they coordinate via the user, not via git.

## Cross-references

- [`docs/guides/inplace-op-coverage.md`](../guides/inplace-op-coverage.md) — which kernels are in-place dispatch safe, by family. Useful context when building helpers that enable in-place patterns.
- [`docs/guides/fa2-saved-tensor-contract.md`](../guides/fa2-saved-tensor-contract.md) — the FW→saved→BW contract pattern, an instance of "contract documented at the trailblazer + tests prove it." Same pattern applies to helper docstrings.
- [`ROADMAP.md`](../../ROADMAP.md) — current and recent phase entries.
