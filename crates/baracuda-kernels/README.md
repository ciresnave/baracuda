# baracuda-kernels

Unified ML-op facade for the baracuda CUDA ecosystem. One import, one
API style; the dispatch decision (NVIDIA-library wrapper vs bespoke
`.cu` kernel) is an internal detail driven by `select`.

## Status — alpha.16

Phase 1 of the comprehensive plan (see
`~/.claude/plans/baracuda-kernels-comprehensive.md`) ships the
**int8 RRR GEMM** family — the SKUs CUTLASS 4.2.0 can't express
because it lacks the 8-bit `TensorOpMultiplicandCongruous` warp
iterator. 18 SKUs, all green on the RTX 4070:

| Element       | Layout   | Epilogue                                            | Bias element       |
|---------------|----------|-----------------------------------------------------|--------------------|
| `S8` / `U8`   | `Rrr`    | `Identity, Bias, BiasRelu, BiasGelu, BiasSilu`      | `f32` / `i32`      |

The rest of the workspace's GEMM coverage (float family across both
layouts, int8 RCR) is re-exported from [`baracuda-cutlass`] so callers
import one crate.

| API                                                   | Source                                | Coverage                                                                       |
|-------------------------------------------------------|---------------------------------------|--------------------------------------------------------------------------------|
| `GemmPlan` (Identity + bias family)                   | `baracuda-cutlass` re-export          | `{Rcr, Rrr} × {F16, Bf16, F32 (TF32), F32Strict (SIMT), F64 (DGEMM)}`          |
| `IntGemmPlan` (Identity + bias family) — `Rcr`        | `baracuda-cutlass` re-export          | `Rcr × {S8, U8} × {bias = f32, bias = i32}`                                    |
| `IntGemmPlan` (Identity + bias family) — `Rrr`        | bespoke (`baracuda-kernels-sys`)      | `Rrr × {S8, U8} × {bias = f32, bias = i32}`                                    |
| `BatchedGemmPlan`, `GroupedGemmPlan`                  | `baracuda-cutlass` re-export          | `Rcr × {F16, Bf16}`                                                            |

## Why this crate

`baracuda-cutlass` covers the CUTLASS-expressible SKUs cleanly. For
the SKUs CUTLASS doesn't cover (today: int8 RRR; later: int4, bin,
FP8 paths CUTLASS handles awkwardly), wrapping CUTLASS would mean
vendoring deeper and deeper into its template chain. Two attempts at
that (alpha.15 Phase 2b, 2b-v2) were reverted in commit `6a1a4dd`
because the vendoring effort exceeded the upstream code it was trying
to reuse. `baracuda-kernels-sys` hosts hand-rolled
`mma.sync.m16n8k32` / `cp.async` / `ldmatrix` kernels for those SKUs;
this crate (`baracuda-kernels`) is the safe Rust facade over them.

The split is **not** a fork: CUTLASS still owns the float family and
the int8 RCR family. Only the SKUs that don't fit CUTLASS's template
chain live in `baracuda-kernels-sys`.

## Quick start

```rust,no_run
use baracuda_kernels::{
    EpilogueKind, IntGemmArgs, IntGemmDescriptor, IntGemmPlan,
    LayoutSku, MatrixMut, MatrixRef, PlanPreference, S8, Workspace,
};
use baracuda_driver::{Context, Device, DeviceBuffer, Stream};

# fn run() -> Result<(), Box<dyn std::error::Error>> {
let ctx = Context::new(&Device::get(0)?)?;
let stream = Stream::new(&ctx)?;

let m = 128i32; let n = 128i32; let k = 128i32;
let dev_a: DeviceBuffer<S8> = DeviceBuffer::zeros(&ctx, (m * k) as usize)?;
let dev_b: DeviceBuffer<S8> = DeviceBuffer::zeros(&ctx, (k * n) as usize)?;
let mut dev_d: DeviceBuffer<S8> = DeviceBuffer::zeros(&ctx, (m * n) as usize)?;

// Note: `Rrr` here dispatches to the bespoke int8 kernels in
// baracuda-kernels-sys. Switching `Rrr` → `Rcr` would dispatch the
// same call through CUTLASS.
let desc = IntGemmDescriptor {
    m, n, k,
    layout: LayoutSku::Rrr,
    epilogue: EpilogueKind::Identity,
};
let plan = IntGemmPlan::<S8>::select(&stream, &desc, PlanPreference::default())?;

let args = IntGemmArgs::<S8, f32> {
    a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
    b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: n as i64 },
    c: None,
    d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
    bias: None,
    alpha: 0.125,
    beta: 0.0,
};
plan.run(&stream, Workspace::None, args)?;
stream.synchronize()?;
# Ok(()) }
```

Switching from `baracuda-cutlass::IntGemmPlan` to
`baracuda-kernels::IntGemmPlan` is a one-line import change — the
plan / descriptor / args / element / epilogue types are all the same
(re-exported from `baracuda-kernels-types`).

## Numerical contract

Per-SKU contract for the int8 RRR kernels (Phase 1):

- **Multiply**: `mma.sync.aligned.m16n8k32.row.col.satfinite.s32.{s8,u8}.{s8,u8}.s32`.
  Bit-stable on the same hardware — integer MMA has no warp-reduction
  nondeterminism.
- **Accumulator**: int32, saturating.
- **Epilogue compute**: f32. `z = α · (acc as f32) + β · C[i,j] + bias[j]`,
  then `activation(z)`, then saturating cast to `{s8, u8}` via
  `cvt.rni.sat.{s8,u8}.f32` (round half to even).
- **Activations**: exact erf-based GELU (matches PyTorch
  `nn.GELU()`), `silu(x) = x / (1 + exp(-x))`, `relu(x) = max(x, 0)`.
- **Bias**: length-`N` device vector, stride 1, broadcast across rows.
  Element type `f32` (default) or `i32` (TensorRT convention) via the
  `BT` generic.

## Workspace requirements

Zero. All Phase 1 SKUs do their work in smem + registers; pass
`Workspace::None`.

## Verification

`crates/baracuda-kernels/tests/int8_rrr_smoke.rs` exercises all 18
SKUs on real hardware against a CPU reference. CPU rounding uses
`f32::round_ties_even()` to match `__float2int_rn` exactly (see the
`sat_cast_s8` docstring for why ordinary `f32::round` is wrong).

```bash
cargo test -p baracuda-kernels --release -- --ignored
```

## Roadmap

- **Phase 2** (alpha.17) — FP8 (E4M3 / E5M2) GEMM RCR + RRR, int4
  (S4 / U4) GEMM, bin (B1) GEMM. Fills out the sm_89 tensor-core
  dtype matrix.
- **Phases 3-9** — elementwise, reductions, normalization, attention,
  convolution, quantization helpers, sort/topk. The PyTorch + JAX
  union op set, each with FW + BW kernels. See the comprehensive
  plan for the full schedule.
- **Phase 10** — sm_89-specific tuning sweep (perf, not correctness).

[`baracuda-cutlass`]: https://docs.rs/baracuda-cutlass
