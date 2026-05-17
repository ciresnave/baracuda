// baracuda-kernels Phase 3 heterogeneous-dtype ternary:
// elementwise `where(cond, a, b)`.
//
// `y = cond ? a : b` where cond is `uint8_t` (PyTorch / NumPy bool
// storage: 0 = false, non-zero = true) and a / b / y share dtype `T`.
// Both contig fast path and strided / broadcast path are wired.
//
// All 4 FP value dtypes wired: {f32, f16, bf16, f64} × {contig,
// strided} = 8 launcher cells. The kernel template is fully generic
// in `T` — each dtype is a single INSTANTIATE invocation.

#include "../include/baracuda_elementwise.cuh"

// No functor here — the op (`cond ? a : b`) is fixed and inlined
// directly in the kernel templates. The INSTANTIATE macros only need
// the value-element type `T` (cond is always `uint8_t`).

BARACUDA_KERNELS_WHERE_INSTANTIATE(where_f32, float)
BARACUDA_KERNELS_WHERE_INSTANTIATE_STRIDED(where_f32, float)

BARACUDA_KERNELS_WHERE_INSTANTIATE(where_f16, __half)
BARACUDA_KERNELS_WHERE_INSTANTIATE_STRIDED(where_f16, __half)

BARACUDA_KERNELS_WHERE_INSTANTIATE(where_bf16, __nv_bfloat16)
BARACUDA_KERNELS_WHERE_INSTANTIATE_STRIDED(where_bf16, __nv_bfloat16)

BARACUDA_KERNELS_WHERE_INSTANTIATE(where_f64, double)
BARACUDA_KERNELS_WHERE_INSTANTIATE_STRIDED(where_f64, double)
