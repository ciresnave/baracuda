# baracuda-kernels-sys

Raw `extern "C"` entry points for the bespoke `.cu` kernels behind
[`baracuda-kernels`]. **You almost certainly want `baracuda-kernels`
instead** — it wraps these unsafe calls with typed plans,
lifetime-checked device buffers, and a proper Rust API.

This crate exists at the layer where the workspace's other `*-sys`
crates do: just the C ABI bindings + a `build.rs` that compiles the
`.cu` sources via [`baracuda-forge`].

## What's compiled here

Today (alpha.16) — Phase 1 of the `baracuda-kernels-comprehensive`
plan:

| Kernel family                          | Files                                                                          | SKUs |
|----------------------------------------|--------------------------------------------------------------------------------|-----:|
| int8 GEMM, RRR layout, sm_80           | `kernels/gemm/gemm_{s8,u8}_rrr_sm80{,_bias}.cu`                                |   18 |

The 18 SKUs are `{S8, U8} × Rrr × {Identity, Bias, BiasRelu,
BiasGelu, BiasSilu} × {f32 bias, i32 bias}`. The kernel bodies share
two headers (`kernels/include/baracuda_int8_rrr_sm80.cuh` for the
templated MMA core, `kernels/include/baracuda_epilogue_int8.cuh` for
the activation chain) so adding a new (element, epilogue) instance is
small.

## Why these SKUs aren't in `baracuda-cutlass-kernels-sys`

CUTLASS 4.2.0 doesn't ship a warp-level `MmaTensorOpMultiplicandTileIterator`
specialization for the 8-bit `TensorOpMultiplicandCongruous` shared-memory
layout that `RowMajor × RowMajor × OpClassTensorOp` selects for int8.
Two attempts at vendoring the missing piece into the CUTLASS template
chain (alpha.15 Phase 2b, 2b-v2) were reverted in commit `6a1a4dd`
because the vendoring effort exceeded the upstream code it would
have reused. Hand-rolling the kernel at the PTX level is shorter and
more maintainable. See
`~/.claude/plans/baracuda-kernels-comprehensive.md` §5 for the
post-mortem.

## Cargo features

- `sm80` (default) — Ampere baseline; runs forward-compatibly on Ada /
  Hopper.
- `sm89` — Ada Lovelace specializations (adds FP8 in Phase 2).
- `sm90a` — Hopper specializations (Phase 11).

## Status codes

All `*_run` and `*_can_implement` entry points return an `i32`:

| Code | Meaning                                        |
|-----:|------------------------------------------------|
|  `0` | success                                        |
|  `1` | misaligned operand                             |
|  `2` | invalid problem (M, N, or K non-positive)      |
|  `3` | not supported                                  |
|  `4` | workspace too small or null when required      |
|  `5` | internal kernel error (typically launch fail)  |

## Safety

Functions take raw `void*` pointers, integer dimensions, and a
`cudaStream_t` cast to `*mut c_void`. They are unsafe because:

- pointers are dereferenced without bounds checking;
- they're assumed to be valid device addresses;
- when a workspace pointer is non-null, it's assumed to point at the
  number of writable device bytes the caller asked
  `*_workspace_size` for;
- the stream is assumed to be a live CUDA stream in the calling
  thread's current context.

If any of those are violated, you'll get either an `i32 == 5` (the
launch failed for a CUDA-reported reason) or undefined behavior at
the GPU level. The safe layer in `baracuda-kernels` enforces all of
these for you.

[`baracuda-kernels`]: https://docs.rs/baracuda-kernels
[`baracuda-forge`]: https://docs.rs/baracuda-forge
