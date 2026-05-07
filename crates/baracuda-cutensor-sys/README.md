# baracuda-cutensor-sys

Raw FFI bindings + dynamic loader for **NVIDIA cuTENSOR** — high-performance
tensor primitives (contraction, reduction, elementwise, permutation)
with arbitrary index permutations.

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libcutensor.so` / `cutensor64_*.dll`.

**Most users want [`baracuda-cutensor`]** — that crate exposes typed
`Handle`, `TensorDescriptor`, `OperationDescriptor`, `ComputeDescriptor`,
`PlanPreference`, and `Plan` builders, plus the full op catalog
(Contraction, Reduction, ElementwiseBinary, ElementwiseTrinary,
Permutation, BlockSparseContraction, TrinaryContraction).

## What's exposed

- All `cutensor*Descriptor_t` types.
- Plan creation + execution.
- Op functions: `cutensorContract`, `cutensorReduce`,
  `cutensorElementwiseBinaryExecute`,
  `cutensorElementwiseTrinaryExecute`, `cutensorPermute`.
- Plan-cache I/O.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cutensor`]: https://docs.rs/baracuda-cutensor
