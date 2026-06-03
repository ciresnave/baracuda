# baracuda-transformer-engine-sys

Build + raw FFI bindings to baracuda's port of NVIDIA
**TransformerEngine**'s FP8 cast/transpose + delayed-scaling recipe
primitives.

## What it wraps

[`NVIDIA/TransformerEngine`](https://github.com/NVIDIA/TransformerEngine)
(Apache-2.0). See `ATTRIBUTION.md` at the crate root for full provenance.

## When to use this crate

Almost never directly — the safe wrapper is
[`baracuda-transformer-engine`]. Most baracuda callers go through the
safe wrapper's `Fp8Recipe` / `Fp8CastPlan` / `Fp8DequantPlan` types,
which are re-exported from `baracuda_kernels::transformer_engine` when
the `tensor_engine` cargo feature on `baracuda-kernels` is enabled.

This crate exists so the FFI surface and the vendored sources can be
built and linked independently of the safe wrapper.

## What this crate exposes

A small flat C ABI defined by `csrc/baracuda_te_shim.cu`:

- `baracuda_te_fused_cast_amax_run` — fused FP8 cast + `max(|x|)`
  amax reduction (one kernel launch).
- `baracuda_te_dequant_run` — FP8 → {f32, f16, bf16} dequantize.
- `baracuda_te_recipe_update_run` — TE delayed-scaling recipe update:
  reduces the amax history ring with `fmax`, computes
  `scale = max_repr / max_amax`, publishes `scale` + `scale_inv`.
- `baracuda_te_recipe_init_run` — set recipe defaults
  (`scale=1`, `scale_inv=1`, `amax_history=0`).
- `baracuda_te_fp8_format_e4m3` / `_e5m2` — format id helpers.

The shim implements the published TE algorithm directly — same
`max_representable / max_amax_in_history` formula as TE's
`transformer_engine/common/recipe/delayed_scaling.cu`, `fmax` reduction
(the TE default), wrap-around index ring.

Bypasses pybind11 — Rust talks raw C ABI. Bypasses cuDNN — the
cast/recipe paths don't need it (cuDNN is needed only for
`fused_attn`, which baracuda skips because it overlaps Phase 17 / 42).

## Quick example

```rust,no_run
use baracuda_transformer_engine_sys as te;

# unsafe fn demo(stream: *mut core::ffi::c_void) {
let e4m3 = te::baracuda_te_fp8_format_e4m3();
debug_assert_eq!(e4m3, te::FP8_FORMAT_E4M3);
# }
```

Real call sites should go through the safe wrapper.

## Sm_89 reality check (RTX 4070)

Ada Lovelace (sm_89) has FP8 storage + cast intrinsics, but the
tensor-core FP8 MMA throughput is roughly equivalent to BF16. So on
this hardware:

- The cast intrinsics work — `__nv_cvt_float_to_fp8` etc. compile +
  execute via `<cuda_fp8.h>`.
- You get a **2× bandwidth saving** vs BF16 on the cast endpoints (KV
  cache, weight storage, gradient passes).
- You do **not** get an MMA throughput win on the GEMM that consumes
  the FP8 tensor — same speed as a BF16 GEMM.

On Hopper (sm_90a) and Blackwell (sm_100), the MMA throughput win
actually materializes. The recipe machinery is forward-compatible.

## Cargo features

| Feature | Default | Effect |
| --- | --- | --- |
| `build-vendor` | no | Build the vendored TE cast + recipe sources via nvcc. Off-by-default — the FFI extern blocks always compile, but link only resolves when a downstream crate (typically `baracuda-kernels`'s `tensor_engine` feature) opts in. |
| `sm80` | no | Ampere fallback (FP8 storage emulated through fp16 — slow but correct). |
| `sm89` | no | RTX 4070 / 4090 test target. Bandwidth wins only. |
| `sm90a` | no | Hopper (where the FP8 tensor-core MMA throughput win actually materializes). |

Pick exactly one arch feature. The safe wrapper's `default` feature
pulls in `sm89` to match the rest of the baracuda CI matrix. docs.rs
has no nvcc — build.rs short-circuits on `DOCS_RS=1`.

## Status / scope

Phase 55 — initial vendor. Out-of-scope TE families (normalization,
`fused_rope`, `fused_attn`, `fused_softmax`, activation, gemm,
`comm_gemm_overlap`, `multi_tensor`, Python bindings) are documented
in the sibling [`baracuda-transformer-engine`] crate's scope section.

## Related crates

- [`baracuda-transformer-engine`] — safe, typed API; the documented
  entry point.
- [`baracuda-kernels`] — re-exports the safe API behind the
  `tensor_engine` cargo feature.
- [`baracuda-cuda-sys`] — the shim TU references `cudaStream_t`,
  `cudaMalloc`, `cudaMemsetAsync`.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

[`baracuda-transformer-engine`]: https://docs.rs/baracuda-transformer-engine
[`baracuda-kernels`]: https://docs.rs/baracuda-kernels
[`baracuda-cuda-sys`]: https://docs.rs/baracuda-cuda-sys
