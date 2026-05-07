# baracuda-cutensor

Safe Rust wrappers for **NVIDIA cuTENSOR** — high-performance tensor
primitives with arbitrary index permutations. Useful when you need
tensor operations beyond what cuBLAS / cuDNN expose (e.g. arbitrary
contractions, reductions over non-contiguous axes).

## Coverage

Comprehensive:

- **Handle**: cuTENSOR context with stream binding.
- **Descriptors**:
  - `TensorDescriptor` — extents, strides, dtype.
  - `OperationDescriptor` — what op to perform (Contraction, Reduction,
    Elementwise, ...).
  - `ComputeDescriptor` — accumulator dtype.
  - `PlanPreference` — heuristic / search policy.
  - `Plan` — finalized plan ready for execution.
- **Op catalog**:
  - `Contraction` — generalized matmul over arbitrary index sets.
  - `Reduction` — per-axis reductions with arbitrary output layout.
  - `ElementwiseBinary` / `ElementwiseTrinary` — fused elementwise ops.
  - `Permutation` — pure index permutation (transpose generalization).
  - `BlockSparseContraction` / `TrinaryContraction` — specialized variants.
- **Plan-cache I/O**: serialize / deserialize plan caches across runs.

## Stack-size note

cuTENSOR's planner can blow a 1 MiB Windows stack during
`cutensorCreatePlan`. If you hit that, run plan creation on a thread
with a larger stack:

```rust,ignore
let result = std::thread::Builder::new()
    .stack_size(32 * 1024 * 1024)
    .spawn(|| /* plan creation here */)
    .unwrap()
    .join();
```

(This is what the workspace's `cutensor_matmul` example does.)

Pairs with [`baracuda-cutensor-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cutensor-sys`]: https://docs.rs/baracuda-cutensor-sys
