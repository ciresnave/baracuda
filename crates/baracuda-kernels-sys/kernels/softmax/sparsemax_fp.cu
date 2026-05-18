// baracuda-kernels Milestone 5.4 + Phase 11.6: Sparsemax FW for FP types.
//
// `y = ProjSimplex(x)` via the standard sort-then-threshold algorithm.
// Phase 11.6 (Fuel #10) replaced the original per-thread serial
// insertion sort with a block-cooperative `cub::BlockRadixSort` +
// `cub::BlockScan` + `cub::BlockReduce` pipeline. One thread block per
// row; two compiled tile specializations cover extents up to 256
// (`ITEMS_PER_THREAD = 1`) and 1024 (`ITEMS_PER_THREAD = 4`). Lifts
// `BARACUDA_SPARSEMAX_MAX_EXTENT` from 64 to 1024.

#include "../include/baracuda_softmax.cuh"

BARACUDA_KERNELS_SPARSEMAX_INSTANTIATE(sparsemax_f32, float)
BARACUDA_KERNELS_SPARSEMAX_INSTANTIATE(sparsemax_f16, __half)
BARACUDA_KERNELS_SPARSEMAX_INSTANTIATE(sparsemax_bf16, __nv_bfloat16)
BARACUDA_KERNELS_SPARSEMAX_INSTANTIATE(sparsemax_f64, double)
