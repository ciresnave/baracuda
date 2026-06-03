# baracuda-ozimmu

Safe Rust wrapper for baracuda's clean-fork of **ozIMMU** — Hiroyuki
Ootomo's Ozaki-scheme FP64 GEMM library that synthesizes a DGEMM from
`S²` int8 tensor-core matmuls.

## What it wraps

baracuda's clean-fork of [`enp1s0/ozIMMU`](https://github.com/enp1s0/ozIMMU)
(MIT). Reference: Ootomo, Ozaki & Yokota, *"DGEMM on Integer Matrix
Multiplication Unit"*, IJHPCA 2024,
[arXiv:2306.11975](https://arxiv.org/abs/2306.11975). Sources are
internalized under [`baracuda-ozimmu-sys`]'s `cuda/` directory
(Phase 44b retired the `vendor/` subdir + cutf submodule).

## Why this crate

On consumer Ada (sm_89) and Ampere there are **no FP64 tensor cores** —
native DGEMM falls back to the FP64 CUDA pipeline. ozIMMU at `S = 8`
recovers "comparable-to-DGEMM" accuracy on well-conditioned inputs
while running on the int8 tensor cores, which is roughly **10–20×
faster** than native DGEMM on these architectures.

This wrapper provides an RAII [`Handle`] type and a drop-in
[`Handle::dgemm`] entry that lets baracuda's `GemmPlan` route FP64
matmuls through the Ozaki path when the caller opts in via
`PlanPreference::prefer_backend = Some(BackendKind::Ozaki { slices: ... })`.

## Accuracy contract

**NOT bit-equivalent to native DGEMM.** ozIMMU at `S = 8` reaches
"comparable to DGEMM" accuracy on well-conditioned inputs; at `S = 3`
it's intentionally low-precision and at `S = 18` it approaches native
DGEMM accuracy but is slower. Use [`OzakiSlices::S8`] as the default;
raise `S` for ill-conditioned workloads, or use
[`OzakiSlices::Auto`] to let ozIMMU pick dynamically based on each
input's mantissa-loss histogram.

For workloads that need the bit-exact DGEMM contract, do **not** use
this crate — baracuda's default FP64 GEMM path stays on CUTLASS /
cuBLAS DGEMM and gives that guarantee.

## Quick example

```rust,no_run
# fn demo(stream: &baracuda_driver::Stream) -> Result<(), Box<dyn std::error::Error>> {
use baracuda_ozimmu::{Handle, OzakiSlices};

let mut handle = Handle::new()?;
handle.set_stream(stream)?;

let m = 1024i32; let n = 1024i32; let k = 1024i32;
// `a`, `b`, `c` are *mut f64 device pointers in column-major layout.
unsafe {
    handle.dgemm(
        OzakiSlices::S8,
        m, n, k,
        1.0, a, m as i64,
             b, k as i64,
        0.0, c, m as i64,
    )?;
}
# Ok(()) }
```

## Cargo features

| Feature | Default | Effect |
| --- | --- | --- |
| `build-vendor` | yes | Forward the vendored-source build to `baracuda-ozimmu-sys/build-vendor` so the static archive actually compiles. Disable only for docs-only / stub workspace checks. |

## Status / scope

- **Phase 44 (alpha.56)**: initial Linux-only vendor.
- **Phase 44b (alpha.57)**: clean-fork + cutf elimination + Windows
  port. ~360 LOC of upstream FP / cp_async utilities folded into
  baracuda-native `baracuda_fp_bits.cuh` + `baracuda_cp_async.cuh`;
  ~2,200 LOC of cutf wrappers deleted as duplicates of baracuda's
  own `-sys` crates. New portable `baracuda::Uint128` unblocks Windows.
- **Phase 44c**: `OzakiVariant::EF` (group-wise error-free
  summation, Uchino / Ozaki / Imamura 2024,
  [arXiv:2409.13313](https://arxiv.org/abs/2409.13313)) layered on
  top of `OzakiSlices`.

The workspace lifecycle is **not** integrated with baracuda's
stream-ordered allocator in this alpha. A future polish phase may add
that bridge if profiling shows the ozIMMU-internal allocator
contending with the rest of baracuda for VRAM at scale.

## Determinism

Given the same hardware + same `OzakiSlices` setting, ozIMMU is
bit-reproducible across launches (the int8 tensor-core path itself is
deterministic; the upstream library does not use atomics on the
accumulate stage). Switching `OzakiSlices` is a numerical change and
produces different — but still bit-reproducible — output.

## Related crates

- [`baracuda-ozimmu-sys`] — raw FFI + build wiring.
- [`baracuda-kernels`] — `GemmPlan` routes to the Ozaki backend behind
  the `ozimmu` cargo feature.
- [`baracuda-cublas`] — used by tests as the accuracy reference.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

[`baracuda-ozimmu-sys`]: https://docs.rs/baracuda-ozimmu-sys
[`baracuda-kernels`]: https://docs.rs/baracuda-kernels
[`baracuda-cublas`]: https://docs.rs/baracuda-cublas
