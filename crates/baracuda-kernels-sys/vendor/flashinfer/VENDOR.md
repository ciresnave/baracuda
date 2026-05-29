# Vendored: NVIDIA / FlashInfer-AI FlashInfer (Phase 46)

This directory contains a curated subset of the FlashInfer CUDA header
tree, vendored into baracuda as part of Phase 46 to give callers three
specific decode-time fast paths that baracuda lacked:

1. **Batched paged-KV decode** (`include/flashinfer/attention/decode.cuh`
   `BatchDecodeWithPagedKVCacheDispatched`) — the key missing primitive
   for vLLM-style serving stacks. Pairs with baracuda's existing
   contiguous `WriteSlicePlan` (which stays as-is) by adding a paged
   KV-cache write helper from `page.cuh`.
2. **Sort-free top-K / top-P / min-P sampling** (`sampling.cuh`
   `TopKTopPSamplingFromProb`) — a single-kernel rejection-sampling
   approach that is much faster than baracuda's existing
   `argsort + softmax + multinomial` decode pipeline.
3. **Cascade attention** (`attention/cascade.cuh`
   `MergeStates*`) — the LSE-merge building block for prefix-cache
   sharing across requests (system prompts, RAG context reuse).

baracuda's existing public attention plans (`FlashSdpaPlan`,
`SdpaPlan`, `WriteSlicePlan`) are **unchanged**. The paged decode +
sampling + cascade plans are NEW families that sit alongside them
behind the opt-in `flashinfer` cargo feature.

## Provenance

- **Upstream**: <https://github.com/flashinfer-ai/flashinfer>
- **Tag**: `v0.6.12`
- **Commit**: `eee0d75f91f64c520bfaed07e39a850ea4ddde23`
- **License**: Apache-2.0 (see `LICENSE` next to this file; `NOTICE`
  carries the NVIDIA + FlashInfer-community copyright assertions and
  references the third-party `licenses/` directory in the upstream
  tree, none of whose files are vendored here).
- **Vendored**: 2026-05-28.

## License attribution

The verbatim upstream `LICENSE` and `NOTICE` files are checked in
alongside this `VENDOR.md`. **Do not modify them.** Per-file
copyright headers in every `include/flashinfer/**/*.{h,cuh}` file
are also preserved verbatim. baracuda's own license (dual MIT /
Apache-2.0) sits alongside in the workspace root; the vendored
FlashInfer headers retain Apache-2.0 (with the patent grant of
Section 3) independently.

The top-level `README.md` of baracuda's workspace lists FlashInfer
under its third-party attribution section.

## Scope: what we kept

`include/flashinfer/` (top level):

- **`math.cuh`**, **`vec_dtypes.cuh`**, **`cp_async.cuh`**, **`mma.cuh`** —
  hardware-abstraction layer (PTX intrinsics, vector load/store,
  fp16/bf16/fp8 packed math, `cp.async` pipeline helpers, `mma.sync`
  fragment shapes). Pulled transitively by every kernel below.
- **`permuted_smem.cuh`**, **`frag_layout_swizzle.cuh`** — SMEM bank
  conflict avoidance for the K/V tile loads.
- **`layout.cuh`** — `QKVLayout` enum + thin tensor descriptor structs.
- **`utils.cuh`** — error / dtype helpers, `FLASHINFER_CUDA_CALL`.
- **`fastdiv.cuh`** — divisor / modulus precomputation for the page
  table lookup hot path.
- **`allocator.h`** — `AlignedAllocator` (linearly bumps through a
  caller-supplied workspace; used by the scheduler).
- **`exception.h`** — `FLASHINFER_CHECK` macro that throws an
  `std::runtime_error`. baracuda's launcher catches at the C-ABI
  boundary and translates to a status int.
- **`topk_common.cuh`** — small helper (NOT the 3380-LOC `topk.cuh`).
- **`page.cuh`** — paged KV-cache descriptors + `AppendPagedKVCacheKernel`
  for write-time KV insertion. Powers the new `PagedKvAppendPlan`.
- **`pos_enc.cuh`** — RoPE / ALiBi position-encoding helpers used by
  `decode.cuh`'s on-the-fly Q rotation.
- **`sampling.cuh`** — `TopK*FromProb`, `TopP*FromProb`, `MinPSamplingFromProb`,
  `TopKTopPSamplingFromProb` sort-free samplers. Powers
  `TopKTopPSamplingPlan`.

`include/flashinfer/attention/`:

- **`decode.cuh`** — the main paged-KV decode dispatcher
  (`BatchDecodeWithPagedKVCacheDispatched`) + single-batch
  (`SingleDecodeWithKVCacheDispatched`). Powers `BatchPagedDecodePlan`.
- **`cascade.cuh`** — `MergeStateInPlace`, `MergeStates`, `MergeStatesLarge`.
  Powers `CascadeAttentionPlan`.
- **`scheduler.cuh`** — host-side workspace-size + tile-plan helpers
  consumed by the decode dispatcher. baracuda's plan layer drives
  this directly from `BatchPagedDecodePlan::select`.
- **`state.cuh`** — `state_t<>` template for the running
  attention-state accumulator.
- **`mask.cuh`** — `MaskMode` enum referenced by the decode params.
- **`heap.h`** — small priority queue used by the scheduler's
  load-balancing pass.
- **`default_decode_params.cuh`** — `BatchDecodeParams<>` /
  `SingleDecodeParams<>` struct definitions wired to the launcher.
- **`variants.cuh`**, **`variant_helper.cuh`** — `StandardAttention`
  variant (the only one we instantiate; FlashInfer's variant template
  hook lets the same kernel host alibi / sliding-window / softcap
  modes at zero source cost).

Total vendored LOC: ~12 kLOC across ~25 headers.

## Scope: what we removed

We skip these subdirectories outright:

- **`hopper/`**, **`blackwell/`**, **`trtllm/`** — sm_90 / sm_100 paths.
  baracuda targets sm_80 / sm_89 / sm_90a; the architectures behind
  these dirs are out of scope for Phase 46.
- **`comm/`**, **`flat/`** — NVSHMEM-backed multi-GPU comm. baracuda is
  single-GPU scope.
- **`mamba/`** — SSM kernels. Slated for a separate Phase 50 vendor
  pass (likely from a dedicated upstream).
- **`gemm/`**, **`norm/`** — overlap with baracuda's existing
  `GemmPlan` / `LayerNormPlan` / `RmsNormPlan`. Wrapping them would
  add a parallel dispatch path with no perf win on the shapes we
  benchmark.

We skip these top-level headers:

- **`topk.cuh`** (3380 LOC) — large, includes parallel-cluster-aware
  top-k that overlaps with baracuda's `TopKPlan`. `sampling.cuh`
  carries a stray `#include "topk.cuh"` but doesn't actually
  reference any symbols from it; the include is patched out (search
  for `// baracuda: removed unused #include "topk.cuh"` in
  `sampling.cuh`).
- **`pod.cuh`**, **`batch_pod.cuh`**, **`prefill.cuh`**, **`persistent.cuh`**,
  **`mla.cuh`**, **`concat_mla.cuh`**, **`cutlass_mla.cuh`** — prefill /
  POD (mixed prefill+decode) / MLA (DeepSeek-V3) / persistent-kernel
  paths. All explicit Phase 46 non-goals.
- **`activation.cuh`**, **`norm.cuh`**, **`pos_enc.cuh`** — overlap with
  baracuda Phase 5 / Phase 14 / Phase 36 / Phase 41 plans.
  (`pos_enc.cuh` IS vendored because `decode.cuh` references it
  for on-the-fly RoPE in the kernel.)
- **`quantization.cuh`**, **`profiler.cuh`**, **`logging.cuh`**,
  **`fast_topk_clusters_exact.cuh`**, **`fp4_layout.cuh`**,
  **`fp16.h`**, **`cubin_loader.h`**, **`cutlass_utils.cuh`**,
  **`arch_condition.h`** — not in the dependency closure of the three
  vendored kernel families.

We also skip the entire `csrc/`, `flashinfer/` (the Python package),
`tests/`, `benchmarks/`, `docker/`, `docs/`, `scripts/`, `ci/`,
`profiler/`, `3rdparty/`, `flashinfer-jit-cache/`, `flashinfer-cubin/`,
`build_backend.py`, `pyproject.toml`. The vendor is **headers only**;
all .cu instantiations live in baracuda's own `kernels/attention/`
+ `kernels/sampling/` C-ABI launcher files (see "Surface" below).

## CUDA / CUTLASS / CCCL version notes

- FlashInfer v0.6.12 is compiled and tested against CUDA 12.3+ with
  CCCL bundled in the toolkit. baracuda's CI tests on CUDA 12.6;
  the FlashInfer headers compile clean against that toolkit.
- The vendored headers depend on `<cuda/cmath>`, `<cuda/functional>`,
  `<cuda/std/...>` (libcudacxx, shipped inside CCCL). These come
  through the CUDA toolkit's standard include path and are NOT
  separately vendored.
- FlashInfer pulls in NVIDIA CUB through `<cub/cub.cuh>` for the
  sampling kernels' block-level scan / reduce. CUB ships with the
  CUDA toolkit since CUDA 11.x and is on the standard include
  path; we do not vendor it.

## PyTorch dependency

FlashInfer ships extensive PyTorch bindings (`csrc/*.cpp` +
`flashinfer/*.py`). **None of those are vendored**. Only the
header-only kernel implementations live here. baracuda's own C-ABI
launchers live at:

- `kernels/attention/flashinfer_paged_decode_launcher.cu`
- `kernels/attention/flashinfer_paged_kv_append_launcher.cu`
- `kernels/attention/flashinfer_cascade_launcher.cu`
- `kernels/sampling/flashinfer_sampling_launcher.cu`

They forward to the FlashInfer dispatcher templates with explicit
dtype / head-dim / page-size instantiations — same pattern as the
Phase 42 FA2 launcher.

## Patch list

Patches applied to vendored headers (record every divergence here):

1. `sampling.cuh` line 36 — `#include "topk.cuh"` commented out;
   sampling does not actually reference any symbol from that header.
2. `vec_dtypes.cuh` line 36 — `FLASHINFER_INLINE` macro changed from
   `inline __attribute__((always_inline)) __device__` to
   `__forceinline__ __device__`. The GCC-specific
   `__attribute__((always_inline))` does not parse under MSVC nvcc;
   `__forceinline__` is the CUDA-portable equivalent (`__forceinline`
   on MSVC, `inline __attribute__((always_inline))` on GCC).
3. `math.cuh` lines 74 + 149 — `ushort` (Linux/BSD typedef from
   `<sys/types.h>`) replaced with `unsigned short`. `ushort` is not
   available on MSVC and the PTX asm constraint `"=h"` (16-bit reg)
   doesn't care about the C-side type as long as it's 2 bytes.
4. `utils.cuh` `DISPATCH_HEAD_DIM` macro — dropped the `case 512:`
   arm. The Phase 46 launcher surface rejects `head_dim != {64, 128,
   256}` up-front; the 512 arm triggers a `static_assert(num_bits ==
   128 || 256)` failure inside `pred_load` because `sizeof(f32) *
   vec_size * 8 == 512` doesn't fit either of the two `cp.async`
   widths the helper supports. Re-add the 512 arm if a future tier
   needs MLA-style head_dim=512.
5. `fastdiv.cuh` — replaced `cuda::fast_mod_div<uint32_t>` (from
   CCCL 2.4+'s `<cuda/cmath>`) with a pure-C++ libdivide-style
   magic-number divmod. The CCCL version isn't shipped with every
   CUDA toolkit release; the bit-equivalent pure-C++ form sidesteps
   the dependency. Same `divmod()` / `operator unsigned int()` /
   `operator/` / `operator%` surface.
6. `attention/decode.cuh` line 941 — `extern __attribute__((shared))
   uint8_t smem[]` replaced with `extern __shared__ uint8_t smem[]`.
   The `__attribute__((shared))` spelling doesn't parse under MSVC
   nvcc; `__shared__` is CUDA-portable. Line is inside the MLA
   kernel which baracuda's launcher doesn't call, but the template
   still gets instantiated transitively.
7. `attention/decode.cuh` lines 669, 752, 763, 1130 — `std::max(...)`
   calls wrapped in `static_cast<size_t>(...)` on both arguments so the
   common type deduction succeeds under Windows MSVC nvcc. The upstream
   expressions mix `unsigned long` (= 32-bit on MSVC) with `size_t`
   (= 64-bit on Win64) from `sizeof()` operands; gcc / clang silently
   widen, MSVC rejects with "no instance of overloaded function
   std::max matches the argument list". Unblocks Phase 46 paged decode
   launcher compile on Windows. Verified at consolidation pass
   2026-05-28.

## Future scope

- **Tier 2**: prefill path (`prefill.cuh`). baracuda's bespoke
  `SdpaPlan` / `FlashSdpaPlan` already cover this; would only vendor
  if benchmark data shows a 1.5×+ win on prefill shapes.
- **MLA**: DeepSeek-V3 specific; depends on customer demand.
- **POD (mixed prefill+decode)**: niche scheduling optimization.
- **NVFP4 GEMM (Blackwell)**, **mamba / SSM**: separate vendor
  efforts (Phase 47-50 on the mainstream-techniques roadmap).

## Re-vendor checklist

To refresh from upstream:

1. `git clone --depth=1 https://github.com/flashinfer-ai/flashinfer.git /tmp/fi`
2. `cd /tmp/fi && git log -1 --format=%H` — capture the commit.
3. `cp` the headers enumerated in "Scope: what we kept" into this
   directory tree.
4. Re-apply the patches in the "Patch list" section above.
5. `cargo build -p baracuda-kernels-sys --features flashinfer` —
   verify clean compile against the new sources.
6. Update the `Tag`, `Commit`, and `Vendored` lines at the top of
   this `VENDOR.md`.
7. Re-run the smoke tests under `crates/baracuda-kernels/tests/`:
   - `paged_decode_smoke.rs`
   - `topk_sampling_smoke.rs`
   - `cascade_attn_smoke.rs`

Honest note: FlashInfer's hardware-abstraction headers (`mma.cuh`,
`cp_async.cuh`, `vec_dtypes.cuh`, `frag_layout_swizzle.cuh`) churn
across upstream releases as the team adds new arch intrinsics. Plan
a quarterly re-vendor pass when CUDA / CCCL bumps require it.
