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
  element.rs    KernelDtype (umbrella marker) + Element / IntElement / FpElement /
                BinElement / BiasElement sibling traits
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

## `Element` vs `KernelDtype` — which to bound on

[`KernelDtype`] is the **umbrella marker** every kernel-usable dtype
implements — including the sub-byte / FP8 / packed-bit newtypes
(`S4`, `U4`, `S8`, `U8`, `Fp8E4M3`, `Fp8E5M2`, `Bin`) that have their
own kernel families. `Element`, `IntElement`, `FpElement`, and
`BinElement` all use `KernelDtype` as a supertrait, so a function
bounded by `<T: KernelDtype>` accepts any kernel-usable type.

The op-shaped sub-traits are what plans actually parameterize on:

| Bound | Accepts | When to use |
| --- | --- | --- |
| `<T: Element>` | `f16, bf16, f32, F32Strict, f64, i32, i64, Bool, Complex32, Complex64` | The elementwise / reduce / scan / norm / loss / shape-layout plans — they consume `BinaryPlan<T, N>` / `UnaryPlan<T, N>` shape with a `type Scalar` projection for α/β. |
| `<T: IntElement>` | `S8, U8, S4, U4` | The int-GEMM plan family. |
| `<T: FpElement>` | `Fp8E4M3, Fp8E5M2` | The FP8 GEMM plan family (sm_89+). |
| `<T: BinElement>` | `Bin` | The 1-bit binary GEMM plan family (XOR + popcount). |
| `<T: KernelDtype>` | union of the four above | Generic utility code that wants to accept *any* dtype — telemetry helpers, dtype-size queries, downstream framework wrappers. |

`KernelDtype::KIND: ElementKind` is the single source of truth for the
runtime dtype tag — pre-Phase-28 code that wrote
`<T as Element>::KIND` should switch to plain `T::KIND` (works under
any of the sub-trait bounds via supertrait inheritance) or
`<T as KernelDtype>::KIND` for the fully-qualified form.

## `#[non_exhaustive]` and forward-compat

Phase 28 marked the op-family discriminant enums plus several tag
enums `#[non_exhaustive]`. Downstream code that `match`es on them
must include a `_ =>` catch-all — adding new variants then no longer
breaks the build. The covered enums:

- **Op-family**: `BinaryKind`, `UnaryKind`, `TernaryKind`,
  `GatedActivationKind`, `PadMode`, `ShapeLayoutKind`, `ArgReduceKind`,
  `ReduceKind`, `SoftmaxKind`, `ScanKind`, `BinaryCmpKind`,
  `NormalizationKind`, `LossKind`, `RandomKind`, `LinalgKind`,
  `FftKind`, `ConvKind`, `PoolKind`, `AttentionKind`, `IndexingKind`,
  `SegmentKind`, `EmbeddingKind`, `QuantizeKind`, `GgufBlockFormat`,
  `MoeKind`, `SortKind`, `ImageKind`.
- **Auxiliary tags**: `OpCategory`, `BackendKind`, `IndexElementKind`,
  `IndexOutputKind`.

Intentionally LEFT exhaustive (deliberate breaking-change events on
new variants):

- `ElementKind` — every kernel dtype is enumerated; a new dtype is a
  workspace-wide event that should surface as a build break across
  every match.
- `LayoutSku`, `ArchSku`, `EpilogueKind`, `ActivationKind`,
  `BiasElementKind` — these are the keys cutlass GEMM and int-GEMM
  dispatchers exhaustively match on to pick per-arch /
  per-fused-epilogue / per-bias-dtype kernel SKUs; adding a variant
  deserves to surface at every match site so each can wire or
  reject.
- `Workspace<'a>` — hot-path-matched by every plan's `run` method;
  the `None` / `Borrowed` split has been stable through every alpha.
- `EmbeddingBagMode`, `FillMode`, `LossReduction`,
  `CrossEntropyTargetKind`, `BatchedOrmqrSide`, `BatchedOrmqrOp` —
  closed mathematical / convention sets (Sum/Mean for the bag,
  Lower/Upper for triangular fill, the LAPACK Left/Right and N/T/C
  ops, the PyTorch reduction modes).

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
