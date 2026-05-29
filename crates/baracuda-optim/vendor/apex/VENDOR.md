# Vendored: NVIDIA Apex — multi-tensor optimizer subset (Phase 49)

This directory contains a curated subset of the NVIDIA Apex CUDA
sources, vendored into baracuda as the Phase 49 foundation for the
`baracuda-optim` crate. Apex's `multi_tensor_apply` idiom is the
load-bearing piece — a single kernel launch over thousands of
parameter tensors instead of one launch per tensor.

## Provenance

- **Upstream**: <https://github.com/NVIDIA/apex>
- **License**: BSD-3-Clause (see `LICENSE` next to this file).
- **Vendored**: 2026-05-28.

## License attribution

The verbatim upstream `LICENSE` and `AUTHORS` files are checked in
alongside this README. **Do not modify them.** Per-file copyright
headers in every kept `.cuh` / `.cu` file are also preserved verbatim.
baracuda's own license (dual MIT / Apache-2.0) sits alongside; the
vendored Apex sources retain BSD-3-Clause independently.

The top-level `README.md` of baracuda's workspace lists Apex under its
third-party attribution section (added by Phase 49).

## Scope: what we kept

The kept set is the minimal source needed to launch
`multi_tensor_adam`, `multi_tensor_lamb`, and `multi_tensor_sgd`
through a flat-C-ABI surface:

| File | LOC | Purpose |
|---|---|---|
| `multi_tensor_apply.cuh` | ~200 | Core launch template + `TensorListMetadata` pack |
| `multi_tensor_adam.cu` | ~150 | Adam fused functor (m + v update + adamw decay) |
| `multi_tensor_lamb.cu` | ~250 | LAMB (Adam + layer-wise adaptive learning rate) |
| `multi_tensor_sgd.cu` | ~100 | SGD with momentum + Nesterov + weight decay |

## Scope: what we removed

- **PyTorch ATen frontends** (`amp_C_frontend.cpp` and the per-op
  `*_frontend.cpp` files) — these expect `at::Tensor` arguments and
  unpack them into raw pointer lists. baracuda replaces them with
  the C-ABI shim at `crates/baracuda-optim/csrc/baracuda_optim_shim.cu`
  which takes raw device pointer arrays directly.
- **fused_dense, mlp_cuda, layer_norm, syncbn, welford, scaled_masked_softmax,
  update_scale_hysteresis, l2norm** — these overlap baracuda's existing
  plans (LayerNorm, MLP via GemmPlan + bias-relu epilogue, SoftmaxPlan,
  ReductionPlan) or are out of scope for the Phase 49 optimizer subset.
- **Python bindings** — baracuda exposes optimizers through
  `AdamStepPlan` / `LambStepPlan` / `SgdStepPlan` Rust types, not Python.

## PyTorch dependency stubs

Apex's `multi_tensor_apply.cuh` references `<ATen/ATen.h>` and uses
`at::Tensor` only inside the `multi_tensor_apply` host-side launcher
template. The baracuda shim does **not** include `multi_tensor_apply.cuh`
directly — instead, the shim:

1. Re-declares the `TensorListMetadata<N>` struct verbatim from the
   header (it's a POD with raw pointer arrays).
2. Re-declares the launch geometry constants (`BLOCK_SIZE`,
   `chunk_size`).
3. Launches the device-side functor (`adam_functor`, `lamb_functor`,
   `sgd_functor`) directly via the kernel launch syntax, instead of
   going through `multi_tensor_apply<T>` which would pull in
   `at::Tensor`.

This keeps the vendored `.cuh` file readable as documentation but
moves the launch logic into baracuda's own TU. The functor structs
themselves are vendored unchanged.

## Future scope

- **AdamW** (weight decay decoupled from gradient) — Apex provides
  `multi_tensor_adam.cu` with an `adamw_mode` flag that toggles between
  classic Adam and AdamW; both are exposed in Phase 49's `AdamStepPlan`.
- **AdaFactor / Sophia / Lion** — separate phases. The
  `multi_tensor_apply` foundation makes adding each new optimizer a
  ~150-LOC functor + Rust plan exercise.
- **8-bit optimizer state** (bitsandbytes) — separate phase (Tier 4
  of the mainstream-techniques roadmap).
- **ZeRO-style sharded / DistributedAdam** — needs NCCL; not in
  baracuda's scope yet.
