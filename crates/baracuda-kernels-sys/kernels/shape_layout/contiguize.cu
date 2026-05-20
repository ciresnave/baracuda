// Phase 13.2 — Contiguize (strided→contiguous copy).
//
// Materializes a contiguous tensor from an arbitrary strided source.
// One kernel body per byte-width covers every byte-aligned dtype
// (f16, bf16, f32, f64, F32Strict, i32, i64, Bool, S8, U8, Fp8E4M3,
// Fp8E5M2, Complex32, Complex64). A separate nibble launcher handles
// S4 / U4 with a documented innermost-stride constraint.
//
// Closes the D2H→CPU contiguize→H2D fallback cliff in Fuel's CUDA
// backend for non-contiguous CUDA inputs.

#include "../include/baracuda_contiguize.cuh"

// Byte-width fanout (one symbol per natural element size).
BARACUDA_KERNELS_CONTIGUIZE_INSTANTIATE(b1,  1)
BARACUDA_KERNELS_CONTIGUIZE_INSTANTIATE(b2,  2)
BARACUDA_KERNELS_CONTIGUIZE_INSTANTIATE(b4,  4)
BARACUDA_KERNELS_CONTIGUIZE_INSTANTIATE(b8,  8)
BARACUDA_KERNELS_CONTIGUIZE_INSTANTIATE(b16, 16)

// Nibble-packed (S4 / U4) — one shared symbol; the launcher returns
// `Unsupported` (status 3) if the source's innermost stride breaks
// nibble alignment.
BARACUDA_KERNELS_CONTIGUIZE_NIBBLE_INSTANTIATE(nibble)
