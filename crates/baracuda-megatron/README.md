# baracuda-megatron

Megatron-LM-style tensor-parallel (TP) primitives for the baracuda CUDA
stack. Pure-composition crate over [`baracuda-cublas`] (local GEMM) +
[`baracuda-nccl`] (cross-rank collectives) — **no new CUDA kernels**.

## What it wraps

Algorithmic reference: Shoeybi et al., *"Megatron-LM: Training
Multi-Billion Parameter Language Models Using Model Parallelism"*,
[arXiv:1909.08053](https://arxiv.org/abs/1909.08053) (2019). The upstream
[NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) repository
is Apache-2.0; **no source is vendored** — the kernel primitives are
reused from the rest of the baracuda stack and composed here.

## When to use this crate

You're splitting a `Linear` layer's weight matrix across multiple GPUs
for distributed training or inference. Two foundational TP plans:

| Type | Splits along | Local input | Local GEMM | Collective |
| --- | --- | --- | --- | --- |
| `ColumnParallelLinearPlan` | output (`out_features`) | full `X` | `Y_local = X · W_local^T` | `all_gather` → `Y` |
| `RowParallelLinearPlan` | input (`in_features`) | sharded `X_local` | `Y_partial = X_local · W_local^T` | `all_reduce(Sum)` → `Y` |

Both backward passes mirror the forward pattern (the collective on one
side becomes a no-op on the other side, mediated by the
sharded-vs-replicated input/output convention).

For non-distributed / single-GPU consumers, depend on
[`baracuda-cublas`] directly — this crate is dead weight without an
NCCL world.

## Quick example

```rust,no_run
# fn demo(comm: &baracuda_nccl::Communicator, handle: &baracuda_cublas::Handle, stream: &baracuda_driver::Stream)
#     -> Result<(), Box<dyn std::error::Error>> {
use baracuda_driver::DeviceBuffer;
use baracuda_megatron::{ColumnParallelLinearPlan, TensorParallelContext};

let in_features = 4096;
let out_features = 16384;  // splits across N ranks
let batch = 32;

let ctx = TensorParallelContext::new(comm, in_features, out_features);
let plan = ColumnParallelLinearPlan::<f32>::new(&ctx, handle)?;

let x: DeviceBuffer<f32> = DeviceBuffer::zeros(handle.context(), (batch * in_features) as usize)?;
let w_local: DeviceBuffer<f32> = DeviceBuffer::zeros(
    handle.context(),
    (ctx.partitioned_out_features() * in_features) as usize,
)?;
let mut y: DeviceBuffer<f32> = DeviceBuffer::zeros(handle.context(), (batch * out_features) as usize)?;
plan.forward(stream, batch, &x, &w_local, &mut y)?;
# Ok(()) }
```

The `TensorParallelContext` borrows the NCCL communicator and caches
per-rank shard sizes derived from `world_size`. One context can host
both `ColumnParallel` and `RowParallel` plans as long as the relevant
dim divides cleanly by `world_size`.

## Cargo features

| Feature | Default | Effect |
| --- | --- | --- |
| `half-crate` | no | Opt-in f16 / bf16 dtype impls. Transitively enables the matching `half-crate` features on `baracuda-types` and `baracuda-nccl`. Off by default so f32-only consumers don't pay the dep surface. |

## Status / scope

- **Tier 1 (Phase 57)**: f32 forward + backward; f16/bf16 behind
  `half-crate`. `world_size == 1` short-circuits collectives to a
  stream-ordered D2D copy and behaves bit-equivalently to a standard
  `Linear` layer (used by in-process smoke tests on single-GPU dev
  hardware).
- **Tier 2 (deferred)**: bias via `Affine` composition (Phase 57
  rejects bias args with `InvalidArgument`; callers can do bias-add
  themselves between calls — on `RowParallel` the bias must be added
  **after** `all_reduce` so it doesn't get summed `N` times).
- **Out of scope**: async overlap (Hopper TMA / `comm_gemm_overlap`
  territory), sequence parallelism (separate Ring Attention phase),
  pipeline parallelism, `VocabParallelEmbedding`, distributed gradient
  accumulation, expert parallelism (MoE).

## Related crates

- [`baracuda-cublas`] — the local GEMM backend (forward + backward).
- [`baracuda-nccl`] — the cross-rank `all_reduce` / `all_gather` collectives.
- [`baracuda-kernels`] — re-exports these plans behind the
  `megatron_tp` cargo feature when the caller wants the unified facade.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

[`baracuda-cublas`]: https://docs.rs/baracuda-cublas
[`baracuda-nccl`]: https://docs.rs/baracuda-nccl
[`baracuda-kernels`]: https://docs.rs/baracuda-kernels
