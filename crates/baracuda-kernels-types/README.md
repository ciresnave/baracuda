# baracuda-kernels-types

Shared type vocabulary for the baracuda ML kernel facade.

This crate has **no GPU code and no behavior of its own**. It ships
pure-data types — traits, enums, structs — that are the contracts the
safe Rust layer ([`baracuda-kernels`]) and the raw FFI layer
([`baracuda-kernels-sys`]) both agree on. The same vocabulary is also
consumed by every per-library wrapper crate
(`baracuda-cublas`, `baracuda-cudnn`, `baracuda-cusolver`, …) so the
whole workspace speaks one dtype + tag language instead of each crate
re-declaring its own.

The types were previously defined in `baracuda-cutlass::types`; they
were lifted out in Phase 0 so the facade could share them without
forcing every consumer to pull in CUTLASS. `baracuda-cutlass` keeps the
old names as re-exports for back-compat.

## What's in here

```text
src/
  element.rs    Element / IntElement / FpElement / BiasElement / BinElement traits
                + ElementKind / MathPrecision / BiasElementKind tag enums
                + scalar wrappers: S8, U8, S4, U4, Bin, F32Strict, Fp8E4M3, Fp8E5M2,
                                   Bool, Complex32, Complex64
                + ScalarType (alpha / beta projection)
  layout.rs     LayoutSku (Rcr, Rrr), ArchSku (Sm80, Sm89, Sm90a),
                EpilogueKind, ActivationKind
  matrix.rs     MatrixRef<T> + MatrixMut<T> + VectorRef<T> — 2-D matrix views for GEMM
  tensor.rs     TensorRef<T, N> + TensorMut<T, N> — rank-N tensor views for
                everything else (const-generic rank, element strides, broadcast convention)
  plan.rs       PlanPreference, PrecisionGuarantee, Workspace
  sku.rs        OpCategory, BackendKind, KernelSku — the structural identity tuple
                returned by every plan's sku() accessor
  ops.rs        Per-category op-discriminant enums: BinaryKind, UnaryKind,
                ReduceKind, ScanKind, SoftmaxKind, NormalizationKind, LossKind,
                AttentionKind, IndexingKind, EmbeddingKind, ShapeLayoutKind,
                SortKind, QuantizeKind, RandomKind, SegmentKind, ImageKind,
                FftKind, LinalgKind, MoeKind, GgufBlockFormat, … (and the
                supporting tag enums: PadMode, FillMode, LossReduction,
                CrossEntropyTargetKind, …)
```

## Why this crate is split out

A few load-bearing reasons:

1. **Zero CUDA dependency at the type level.** Downstream crates that
   only want the dtype vocabulary (e.g. a tensor library that needs
   `Element::KIND` to identify dtypes for printing) don't have to pull
   in CUDA, CUTLASS, or any `*-sys` crate. The runtime dependency
   surface is `baracuda-driver` (for `DeviceSlice`) + `baracuda-types`
   (for `DeviceRepr` / `Half` / `BFloat16`) + `half` + `float8`.
2. **One vocabulary per scalar dtype.** Without this crate, each
   per-library wrapper would re-derive its own `enum Dtype { F32, F16,
   … }` from the underlying NVIDIA library's tag enum (`cudaDataType_t`,
   `cudnnDataType_t`, `cufftType`, …) and the safe facade would spend
   its life translating between them. Centralizing the
   vocabulary here means the translation table lives in one place per
   wrapper.
3. **`KernelSku` is the autotuner cache key.** It needs to be `Copy +
   Eq + Hash` and stable across versions. Defining it here, away from
   any one library wrapper, keeps it neutral.

## Dependencies

```toml
[dependencies]
baracuda-types  = "...features = [\"half-crate\", \"f8-crate\"]"
baracuda-driver = "..."
half            = "2"
float8          = "0.7"
```

`baracuda-driver` is the source of the lifetimed device-slice types
(`DeviceSlice` / `DeviceSliceMut`) that back `TensorRef` / `TensorMut`.
The `half` and `float8` crates supply the precise IEEE half-precision
and 8-bit-float wrappers (`half::f16`, `half::bf16`, `float8::F8E4M3`,
`float8::F8E5M2`) that the `FpElement` impls re-export.

## Usage

You generally don't depend on this crate directly — depend on
[`baracuda-kernels`] (which re-exports the entire surface) and import
from there:

```rust
use baracuda_kernels::{
    Element, ElementKind, KernelSku, PrecisionGuarantee,
    TensorRef, TensorMut, MatrixRef, MatrixMut,
    LayoutSku, EpilogueKind, ArchSku, PlanPreference, Workspace,
};
```

Depending on `baracuda-kernels-types` directly is only useful if you're
writing a sibling wrapper crate that needs the vocabulary but doesn't
want the rest of the facade (`baracuda-cublas` / `baracuda-cudnn` etc.
do exactly this).

## See also

- [`baracuda-kernels`](../baracuda-kernels) — the safe facade that
  re-exports everything in this crate.
- [`baracuda-kernels-sys`](../baracuda-kernels-sys) — the raw FFI
  layer this vocabulary describes the contracts for.
- [`ARCHITECTURE.md`](../../ARCHITECTURE.md) — the layered design,
  the Plan / Descriptor / Args triple, the `KernelSku` taxonomy, the
  workspace contract.

[`baracuda-kernels`]: https://docs.rs/baracuda-kernels
[`baracuda-kernels-sys`]: https://docs.rs/baracuda-kernels-sys
