# Vendor: NVIDIA TransformerEngine — FP8 cast/recipe subset

**Upstream**: https://github.com/NVIDIA/TransformerEngine

**Reference commit** *(recorded 2026-10-14)*: `9e5a847c90c436c9c9c5f3e756977cf568967ccc` (2026-05-28) — the last upstream commit on or before the
  2026-05-28 vendor date. The vendor-time commit was **not recorded and is not recoverable**; this pin is
  date-bounded. `ATTRIBUTION.md`'s licence text is verbatim `LICENSE` at it. No TE source is copied; upstream has no NOTICE.

**License**: Apache-2.0 (with §3 patent grant — see `../../ATTRIBUTION.md`)
**baracuda phase**: 55

## What's actually here

This directory is intentionally minimal. Phase 55 implements the
delayed-scaling recipe algorithm + FP8 cast directly in the C-ABI
shim at `../../csrc/baracuda_te_shim.cu`, rather than vendoring TE's
C++ template surface verbatim.

Reasons for the shim-only approach (matches Phase 49 with Apex's
`multi_tensor_apply<T>` host launcher):

1. **No pybind11 transitive dep**. TE's vendored C++ surface is
   intertwined with `transformer_engine/common/util/pybind_helper.h`
   and friends — pulling those in would force a Python build dep
   that Rust callers don't want.
2. **No cuDNN 9.3+ dep**. TE's vendored CMake requires cuDNN 9.3+
   for `fused_attn`, which we skip in Phase 55 (baracuda Phase 17
   + Phase 42 already cover the attention surface). The cast/recipe
   paths themselves don't need cuDNN — they only need the in-toolkit
   `<cuda_fp8.h>` cast intrinsics.
3. **Algorithm is small**. The TE delayed-scaling recipe is
   `scale = max_representable / max_amax_in_history` over a
   sliding-window ring of amax samples. Re-implementing this against
   the published reference is cleaner than dragging in TE's full
   `Tensor` / `NVTEDType` / `Quantizer` machinery.
4. **Forward-compatible**. On Hopper / Blackwell where the FP8
   tensor-core MMA throughput actually matters, the same recipe
   state drives whatever MMA-aware GEMM kernel the caller wires up
   (we don't lock the recipe to TE's MMA pipeline).

## What we lifted (algorithmic, not source)

| From upstream                                          | Where it lands in baracuda                          |
|--------------------------------------------------------|-----------------------------------------------------|
| `common/recipe/delayed_scaling.cu` (recipe update)     | `recipe_update_kernel` in `baracuda_te_shim.cu`     |
| `common/cast/cast.cu` (fused amax + cast)              | `fused_cast_amax_kernel` in `baracuda_te_shim.cu`   |
| `common/include/transformer_engine/recipe.h` (constants) | `fp8_max_representable` helper                    |
| `<cuda_fp8.h>` `__nv_cvt_*_to_fp8` (cast intrinsics)   | `cvt_f32_to_e4m3` / `cvt_f32_to_e5m2` helpers       |

## What we deliberately left out

See `../../ATTRIBUTION.md` for the full table. Highlights:

- `normalization` (baracuda Phase 5)
- `fused_attn` (baracuda Phase 17 + 42; the cuDNN 9.3 dep)
- `gemm` (baracuda Phase 1 + 24 + 30)
- All Python (`pytorch/`, `jax/`)
- All Hopper-only (`comm_gemm_overlap`, `nvshmem_api`)

## License compliance

The Apache-2.0 license is preserved at the crate root via
`../../ATTRIBUTION.md`. The shim TU carries a `SPDX-FileCopyrightText`
header attributing baracuda's contributions. The recipe algorithm
itself is documented as NVIDIA's work in both the shim header
comment and the safe wrapper's rustdoc.
