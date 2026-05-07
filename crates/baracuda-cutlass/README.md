# baracuda-cutlass

Safe Rust wrapper for compiled CUTLASS kernels in the baracuda ecosystem.

`baracuda-cutlass` provides a **plan-based GEMM and grouped-GEMM API** with
caller-supplied workspace, typed device-buffer arguments, and capture-safe
launches. It sits above [`baracuda-cutlass-kernels-sys`] (the compiled
kernels) and below framework integration crates like Fuel's `fuel-cublaslt`.

## v0 scope

- **Op families**: `GemmPlan` (single GEMM), `GroupedGemmPlan` +
  `PreparedGroupedGemm` (variable-M-per-group, MoE-friendly).
- **Element types**: `half::f16`, `half::bf16`.
- **Layout**: `RCR` only — `A` row-major `[M,K]`, `B` column-major `[K,N]`,
  `C/D` row-major `[M,N]`, `f32` accumulation, `f32` α/β.
- **Epilogues**: `Identity` only. (`Bias` was deferred during the Fuel team
  design review — it'll return when the corresponding kernel instantiation
  ships, alongside the `bias` field on `GemmArgs` / `GroupedProblem`.)
- **Architectures**: `sm_80` shipped today (runs on Ampere, Ada, and
  forward-compatibly on Hopper). `sm_90a` selection wiring is in place;
  the Hopper-specialized kernels themselves land when Hopper hardware is
  available for validation.
- **Workspace**: caller-supplied — `Workspace::None` or
  `Workspace::Borrowed(DeviceSliceMut<u8>)`. Plans never own device memory.
  Grouped GEMM additionally packs its per-group metadata into the front of
  the workspace via async H2D, with CUTLASS's internal scratch at the tail.

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

## Acknowledgments

API specification by the Fuel ML library team. Underlying CUTLASS by NVIDIA.
See `NOTICE` for full attribution.

[`baracuda-cutlass-kernels-sys`]: ../baracuda-cutlass-kernels-sys
