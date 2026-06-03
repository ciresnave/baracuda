# baracuda-transformer-engine

Safe Rust wrapper for baracuda's port of NVIDIA **TransformerEngine**'s
FP8 cast/transpose + delayed-scaling recipe primitives.

## What it wraps

[`NVIDIA/TransformerEngine`](https://github.com/NVIDIA/TransformerEngine)
(Apache-2.0). Phase 55 lifts only the cast / recipe subset — see scope
below.

Provides:

- `Fp8Recipe` — RAII handle for a per-tensor delayed-scaling recipe
  (amax history ring buffer + current scale + reciprocal scale).
- `Fp8CastPlan` — fused cast `{f32, f16, bf16}` → FP8 (E4M3 or E5M2)
  with per-launch `max(|x|)` amax reduction into the recipe.
- `Fp8DequantPlan` — FP8 → `{f32, f16, bf16}` dequant using the
  recipe's `scale_inv` scalar.

## Sm_89 reality check

On Ada Lovelace (RTX 4070, RTX 4090, L40, L4 — sm_89), the FP8 storage
and cast intrinsics work natively, but the tensor-core FP8 MMA
throughput is roughly equivalent to BF16. So on this hardware, the
FP8 wins are **bandwidth-saving only**:

- **Real win**: 2× memory savings on KV cache, weight storage,
  activation memory.
- **No win**: FP8 GEMM consumed downstream runs at BF16-equivalent
  throughput on the tensor cores.

On Hopper (sm_90a) and Blackwell (sm_100), the MMA throughput win also
materializes. The recipe machinery in this crate is forward-compatible
— the same `Fp8Recipe` drives whatever MMA-aware GEMM kernel you wire
up.

## Algorithm

Mirrors TE's published
`transformer_engine/common/recipe/delayed_scaling.cu`:

1. **During each forward pass**: `Fp8CastPlan` fuses cast + `max(|x|)`
   reduction in one kernel. The reduced amax is `atomicMax`-published
   into `amax_history[write_pos]`.
2. **After the forward pass**: `Fp8Recipe::update_after_pass` reduces
   the amax history ring with `fmax`, computes
   `new_scale = max_representable / max_amax`, publishes `scale` +
   `scale_inv`, and resets the just-written slot.
3. The ring write pointer advances; the next forward pass writes
   into the new slot.

## Quick example

```rust,no_run
use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_transformer_engine::{Fp8Format, Fp8Recipe, Fp8CastPlan};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
baracuda_driver::init()?;
let ctx = Context::new(&Device::get(0)?)?;
let stream = Stream::new(&ctx)?;

// 1. Build a recipe (amax history len 1024 is TE's typical default).
let mut recipe = Fp8Recipe::new(&ctx, &stream, Fp8Format::E4M3, 1024)?;

// 2. Build a cast plan for f16 -> E4M3.
let plan: Fp8CastPlan<half::f16> = Fp8CastPlan::select()?;

// 3. Per forward pass: cast inputs through the plan.
let x: DeviceBuffer<half::f16> = DeviceBuffer::zeros(&ctx, 4096)?;
let mut y: DeviceBuffer<u8>    = DeviceBuffer::zeros(&ctx, 4096)?;
plan.run(&x, &mut y, &mut recipe, &stream)?;

// 4. Periodically (e.g. once per training step) advance the recipe.
recipe.update_after_pass(&stream)?;
# Ok(()) }
```

## Cargo features

| Feature | Default | Effect |
| --- | --- | --- |
| `build-vendor` | no | Forward the vendored-source build to `baracuda-transformer-engine-sys`. Auto-enabled by any arch feature. |
| `sm80` | no | Ampere fallback (FP8 storage works via the cast intrinsics emulated through fp16 — slow but correct). |
| `sm89` | no | Ada Lovelace target (RTX 4070 / 4090). Bandwidth wins only; no MMA throughput win. |
| `sm90a` | no | Hopper. MMA throughput win materializes here. |

Pick exactly one arch feature; the build.rs picks
`sm90a > sm89 > sm80` when multiple are on. Each arch feature implies
`build-vendor` so callers don't need to remember to enable both.

## Status / scope

Phase 55 lifts only the cast/recipe subset of TE upstream. Everything
else deliberately skipped (overlaps existing baracuda phases, would
need cuDNN, or targets Hopper-only hardware):

- `normalization` (baracuda Phase 5 RMSNorm / LayerNorm)
- `fused_rope` (baracuda Phase 14 / 36 / 41)
- `fused_attn` (baracuda Phase 17 / 42 — also the one piece of TE
  that *would* need cuDNN 9.3+, hence "no cuDNN" is achievable by
  skipping it)
- `fused_softmax` (baracuda Phase 5)
- `activation` (baracuda Phase 3 / 31)
- `gemm` (baracuda Phase 1 + 24 + 30)
- `comm_gemm_overlap` / `nvshmem_api` (Hopper-only)
- `multi_tensor` (baracuda Phase 49 Apex optimizer)
- All Python bindings (`transformer_engine/pytorch/`,
  `transformer_engine/jax/`) — raw C ABI, not pybind11.

**No cuDNN dep, no pybind11.**

## Related crates

- [`baracuda-transformer-engine-sys`] — raw FFI surface.
- [`baracuda-kernels`] — re-exports `Fp8Recipe` / `Fp8CastPlan` /
  `Fp8DequantPlan` from `baracuda_kernels::transformer_engine` when
  the `tensor_engine` cargo feature is enabled.
- [`baracuda-driver`] — `Context` / `Stream` / `DeviceBuffer`.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

[`baracuda-transformer-engine-sys`]: https://docs.rs/baracuda-transformer-engine-sys
[`baracuda-kernels`]: https://docs.rs/baracuda-kernels
[`baracuda-driver`]: https://docs.rs/baracuda-driver
