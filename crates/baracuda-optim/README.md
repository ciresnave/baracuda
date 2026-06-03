# baracuda-optim

Fused multi-tensor optimizer kernels (Adam / LAMB / SGD) for the
baracuda CUDA stack.

## What it wraps

The `multi_tensor_apply` idiom vendored from
[NVIDIA Apex](https://github.com/NVIDIA/apex) (BSD-3-Clause). Apex
sources live under `vendor/apex/`; PyTorch-specific glue was replaced by
a PyTorch-free C-ABI shim in `csrc/baracuda_optim_shim.cu`.

## Why this crate

baracuda's main facade (`baracuda-kernels`) ships **zero** optimizers —
it's a kernel substrate, not a training framework. Without
multi-tensor apply, the optimizer step on a 32B-parameter model would
launch ~10,000 kernels per step at ~5µs of launch overhead each =
**50ms of pure launch latency per step**. Multi-tensor apply collapses
that to a single launch (or a small handful when the per-launch
tensor cap of 110 is exceeded) by packing all per-tensor pointers +
sizes into one parameter struct. The crate boundary is deliberate —
inference-only consumers don't pay the FFI surface cost.

## What's exposed

| Type | Update rule | Param dtype |
| --- | --- | --- |
| `AdamStepPlan` | classic Adam or AdamW (mode flag) | f32 / f16 / bf16 |
| `LambStepPlan` | LAMB (Adam + per-layer trust ratio) | f32 (Phase 49) |
| `SgdStepPlan` | SGD + momentum + Nesterov + weight decay | f32 / f16 / bf16 |
| `DistributedAdamStepPlan` | ZeRO-1-style sharded Adam (behind `distributed_optim`) | f32 / f16 / bf16 |

Mixed-precision wiring: when param/grad is f16/bf16, the moments
(`exp_avg` / `exp_avg_sq` / momentum_buf) **must** stay in f32 —
half-precision moments lose precision catastrophically.

## Quick example

```rust,no_run
# fn demo(stream: &baracuda_driver::Stream) -> Result<(), Box<dyn std::error::Error>> {
use baracuda_driver::DeviceBuffer;
use baracuda_optim::AdamStepPlan;

let plan = AdamStepPlan::<f32>::new()
    .with_lr(1e-3)
    .with_betas(0.9, 0.999)
    .with_eps(1e-8)
    .with_weight_decay(0.01)
    .with_adamw_mode(true);

// One `step` call updates ALL parameter tensors in a single launch
// (up to the 110-tensor per-launch cap).
plan.step(stream, &mut params, &grads, &mut exp_avg, &mut exp_avg_sq, /* step = */ 1)?;
# Ok(()) }
```

## Cargo features

| Feature | Default | Effect |
| --- | --- | --- |
| `sm80` | no | Build the Ampere-baseline kernel set. Forward-compatible to Ada / Hopper. |
| `sm89` | no | Build the Ada Lovelace specializations. |
| `sm90a` | no | Build the Hopper-specialized kernels. |
| `distributed_optim` | no | Enable `DistributedAdamStepPlan` (ZeRO-1-style sharded optimizer state on top of NCCL collectives). Pulls in `baracuda-nccl` (which dynamically loads `libnccl`; first `nccl()` call returns `LoaderError::LibraryNotFound` on hosts without it). Default OFF — inference-only and single-GPU training consumers don't pay the dep cost. |

Pick exactly one of `sm80` / `sm89` / `sm90a`; the build script picks
`sm90a > sm89 > sm80` when multiple are on.

## Status / scope

- Phase 49 — initial Adam / SGD / LAMB fanout.
- Phase 58 — `DistributedAdamStepPlan` (ZeRO-1). Single-rank
  degenerate case reduces to `AdamStepPlan` bit-exactly.

## Related crates

- [`baracuda-kernels`] — re-exports these plans behind the
  `optim` cargo feature when callers want the unified facade.
- [`baracuda-driver`] — `Stream` / `DeviceBuffer`.
- [`baracuda-nccl`] — optional dep behind `distributed_optim`.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

[`baracuda-kernels`]: https://docs.rs/baracuda-kernels
[`baracuda-driver`]: https://docs.rs/baracuda-driver
[`baracuda-nccl`]: https://docs.rs/baracuda-nccl
