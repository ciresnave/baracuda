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
//
// Phase 38 (Fuel 6c.4 Gap 3) extended the matrix to U32 / I64 cond
// dtypes and all int + Fp8E4M3 value dtypes; those instantiations
// live in `where_dtype_fanout.cu` and use the explicit
// `where_<cond>cond_<value>_run` naming. The u8-cond fp symbols here
// keep their no-prefix names (`where_f32_run`, etc.) for source
// compat — they implicitly mean "u8 cond".

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
