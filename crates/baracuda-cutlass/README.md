# baracuda-cutlass

Safe Rust wrapper for compiled CUTLASS kernels in the baracuda ecosystem.

`baracuda-cutlass` provides a **plan-based GEMM and grouped-GEMM API** with
caller-supplied workspace, typed device-buffer arguments, and capture-safe
launches. It sits above [`baracuda-cutlass-kernels-sys`] (the compiled
kernels) and below framework integration crates like Fuel's `fuel-cublaslt`.

## Scope

- **Op families**: `GemmPlan` (single GEMM), `BatchedGemmPlan`
  (uniform-shape batched GEMM), `GroupedGemmPlan` +
  `PreparedGroupedGemm` (variable-M-per-group, MoE-friendly).
- **Element types**: `half::f16`, `half::bf16`, `f32` (routed through
  TF32 tensor cores at ~10-bit mantissa precision),
  [`F32Strict`](crate::F32Strict) (full IEEE 754 binary32 via SIMT CUDA
  cores — bit-stable, no tensor-core warp-reduction nondeterminism),
  and `f64` (DGEMM via Ampere FP64 tensor cores). See
  [`PrecisionGuarantee::math_precision`] and
  [`ScalarType`](crate::ScalarType) for the per-element math precision
  and alpha/beta scalar mapping.
- **Layouts**: `RCR` (A row-major, B column-major, C/D row-major) and
  `RRR` (all three operands row-major — natural for activation@weight
  matmul without a transpose pass). All shipped element types ship both
  layouts; layout is a per-launch choice on `GemmDescriptor`.
- **Epilogues**: `Identity`, `Bias`, `BiasRelu`, `BiasGelu`, `BiasSilu`.
  The `Bias*` family computes
  `D = activation(α·A·B + β·C + bias_broadcast(N))` in a single fused
  kernel pass via `cutlass::gemm::device::GemmUniversalWithBroadcast` +
  `LinearCombinationBiasElementwise` — the bias add and activation
  both happen inside the epilogue, no extra memory traffic over plain
  `Bias`. The bias vector has length `N` and must be contiguous
  (stride 1). `GemmArgs::bias` is required iff
  `descriptor.epilogue.requires_bias()` is `true`. GELU is the exact
  (erf-based) form, matching PyTorch's default `nn.GELU()`.
- **Architectures**: `sm_80` shipped today (runs on Ampere, Ada, and
  forward-compatibly on Hopper). `sm_90a` selection wiring is in place;
  the Hopper-specialized kernels themselves land when Hopper hardware is
  available for validation.
- **Workspace**: caller-supplied — `Workspace::None` or
  `Workspace::Borrowed(DeviceSliceMut<u8>)`. Plans never own device memory.
  Grouped GEMM additionally packs its per-group metadata into the front of
  the workspace via async H2D, with CUTLASS's internal scratch at the tail.

### Kernel SKU coverage

| API                                                   | Layout × Element                                                              |
|-------------------------------------------------------|-------------------------------------------------------------------------------|
| `GemmPlan` (Identity)                                 | `{Rcr, Rrr} × {F16, Bf16, F32 (TF32), F32Strict (SIMT), F64 (DGEMM)}`         |
| `GemmPlan` (Bias / BiasRelu / BiasGelu / BiasSilu)    | `{Rcr, Rrr} × {F16, Bf16, F32 (TF32), F32Strict (SIMT), F64 (DGEMM)}`         |
| `IntGemmPlan` (Identity)                              | `Rcr × {S8, U8}`                                                              |
| `IntGemmPlan` (Bias / BiasRelu / BiasGelu / BiasSilu) | `Rcr × {S8, U8} × {bias = f32, bias = i32}`                                   |
| `BatchedGemmPlan`                                     | `Rcr × {F16, Bf16}`                                                           |
| `GroupedGemmPlan`                                     | `Rcr × {F16, Bf16}`                                                           |

**Per-element scalar (alpha / beta) types:**

| Element       | `T::Scalar` | Notes                                                                                                        |
|---------------|-------------|--------------------------------------------------------------------------------------------------------------|
| `f16`         | `f32`       | Tensor-core math, F32 accumulator                                                                            |
| `bf16`        | `f32`       | Tensor-core math, F32 accumulator                                                                            |
| `f32`         | `f32`       | TF32 tensor-core math (10-bit mantissa), F32 accumulator                                                     |
| `F32Strict`   | `f32`       | SIMT full-precision math, F32 accumulator, bit-stable                                                        |
| `f64`         | `f64`       | DGEMM tensor-core math, F64 accumulator                                                                      |
| `S8` / `U8`   | `f32`       | Int8 tensor-core math, int32 accumulator, bit-stable. Float alpha/beta let the epilogue act as a dequantize. |

**Int family notes:**

`IntGemmPlan<T: IntElement, BT: BiasElement = f32>` is a sibling type to
`GemmPlan`. The matrix element `T` picks the kernel family
(`S8` / `U8` today; `s4` / `u4` / 1-bit deferred to follow-ups). For
`Bias*` epilogues, `BT` picks the bias broadcast element type — `f32`
(default; matches the float-bias convention used elsewhere) or `i32`
(matches TensorRT's int8 inference convention). Both routes use
`LinearCombinationBiasElementwise` with `ElementCompute = float`, so
the fused activation runs in float space after int32→float dequant and
the final saturating cast back to int8 happens via the
`cvt.rni.sat.{s8,u8}.f32` PTX instruction.

`Rrr` is **not yet shipped** for the int family — CUTLASS 4.2.0 lacks
warp-level iterator specializations for the 8-bit
`TensorOpMultiplicandCongruous` shared-memory layout that
`RowMajor × RowMajor × OpClassTensorOp` would select for int8. Selecting
`LayoutSku::Rrr` on `IntGemmPlan` returns `Error::Unsupported` at plan
selection time. A follow-up release will vendor the missing
specialization.

Remaining int / quantized dtypes (`s4`/`u4`/`b1`) are planned
follow-ups and not yet shipped.

All on `sm_80` (Ampere); `sm_90a` deferred until Hopper validation.

## Why plan-based, not handle-based?

CUTLASS isn't cuBLAS. There is no persistent driver-side state that lives
across kernel launches. Every kernel is a self-contained instantiation of
a template. A `Plan` holds the *selected kernel ID and its host-side
metadata* — not a handle, not a workspace. This makes plans cheap to clone,
trivially `Send + Sync`, and capture-safe by construction (no host
allocations during `run`).

## Quick start

```rust,no_run
use baracuda_cutlass::{
    EpilogueKind, GemmArgs, GemmDescriptor, GemmPlan, LayoutSku,
    MatrixMut, MatrixRef, PlanPreference, Workspace,
};
use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use half::f16;

# fn run() -> Result<(), Box<dyn std::error::Error>> {
let ctx = Context::new(&Device::get(0)?)?;
let stream = Stream::new(&ctx)?;

let m = 128i32; let n = 128i32; let k = 128i32;
let dev_a: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (m * k) as usize)?;
let dev_b: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (k * n) as usize)?;
let mut dev_d: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (m * n) as usize)?;

let desc = GemmDescriptor {
    m, n, k,
    layout: LayoutSku::Rcr,
    epilogue: EpilogueKind::Identity,
};
let plan = GemmPlan::<f16>::select(&stream, &desc, PlanPreference::default())?;

let args = GemmArgs::<f16> {
    a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
    b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: k as i64 },
    c: None,
    d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
    bias: None,
    alpha: 1.0,
    beta: 0.0,
};
plan.can_implement(&args)?;
plan.run(&stream, Workspace::None, args)?;
# Ok(()) }
```

## Grouped GEMM quick start (MoE-friendly)

```rust,no_run
use baracuda_cutlass::{
    EpilogueKind, GroupedGemmPlan, GroupedPlanPreference, GroupedProblem,
    MatrixMut, MatrixRef, Workspace,
};
use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use half::f16;

# fn run() -> Result<(), Box<dyn std::error::Error>> {
let ctx = Context::new(&Device::get(0)?)?;
let stream = Stream::new(&ctx)?;

// One "group" per expert. Variable M (token count), shared K/N.
// Build your GroupedProblem<f16> slice from per-expert device buffers
// (omitted here for brevity — see tests/grouped_gemm_smoke.rs).
let groups: Vec<GroupedProblem<'_, f16>> = todo!("build per-expert problems");

let plan = GroupedGemmPlan::<f16>::select(
    &stream,
    EpilogueKind::Identity,
    GroupedPlanPreference::default(),
)?;
let prepared = plan.prepare(&groups)?;

// Allocate one workspace big enough for the packed metadata + CUTLASS scratch.
let mut workspace: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, prepared.workspace_size())?;
prepared.run(&stream, Workspace::Borrowed(workspace.as_slice_mut()))?;
# Ok(()) }
```

`prepare` validates per-group shapes and v0 invariants (all groups share
α/β; all groups consistently use `c = None` or `c = Some(_)`), packs host
arrays for `problem_sizes`, pointers, and leading dimensions, and queries
CUTLASS for the threadblock count + scratch size. `run` uploads the
metadata to the start of the workspace via async H2D and launches the
grouped kernel using the remainder as CUTLASS internal scratch.

## Integration notes

### Calling from a byte-storage substrate

Frameworks that store all device tensors as `DeviceBuffer<u8>` (e.g.
Fuel's unified-binding-table dispatch path) can construct typed
`MatrixRef` / `MatrixMut` views without copying or transmuting:

```rust,no_run
use baracuda_cutlass::{MatrixMut, MatrixRef};
use baracuda_driver::DeviceBuffer;
use half::bf16;

# fn demo(byte_a: &DeviceBuffer<u8>, byte_d: &mut DeviceBuffer<u8>) {
let m = 128i32; let n = 128i32; let k = 128i32;
let a_view: MatrixRef<bf16> = MatrixRef {
    data: byte_a.view_as::<bf16>(),
    rows: m, cols: k, ld: k as i64,
};
let d_view: MatrixMut<bf16> = MatrixMut {
    data: byte_d.view_as_mut::<bf16>(),
    rows: m, cols: n, ld: n as i64,
};
# let _ = (a_view, d_view); }
```

[`DeviceBuffer<u8>::view_as`](baracuda_driver::DeviceBuffer::view_as)
asserts byte-count divisibility and reuses the buffer's existing
allocation — no copy, no `unsafe` at the consumer site. For non-baracuda
allocations, the lower-level
[`DeviceSlice::from_raw_parts`](baracuda_driver::DeviceSlice::from_raw_parts)
escape hatch is available.

### Sharing a stream across launchers

A consumer that holds an `Arc<Stream>` (e.g. one stream per device,
shared across many kernel launches) can pass it to `plan.run` directly
via `Arc::as_ref` — the `&Stream` borrow shape is the same as for an
owned `Stream`:

```rust,no_run
# use std::sync::Arc;
# use baracuda_driver::Stream;
# fn demo(shared: Arc<Stream>) {
// `shared.run(...)` — Arc<Stream> auto-derefs to &Stream at the
// call site; no extra Stream::new per launcher needed.
# let _ = shared.as_ref(); }
```

### Mapping kernels to precision guarantees

For consumers maintaining a per-decision-point alternatives table
(picking between cuBLAS and CUTLASS at a given precision contract),
[`GemmPlan::precision_guarantee`] (and the grouped equivalent) returns
a `PrecisionGuarantee` value — math-instruction precision, accumulator
type, bit-stability and determinism flags — without re-derivation from
per-kernel docs.

## Acknowledgments

API specification by the Fuel ML library team. Underlying CUTLASS by NVIDIA.
See `NOTICE` for full attribution.

[`baracuda-cutlass-kernels-sys`]: ../baracuda-cutlass-kernels-sys
