// baracuda-kernels Phase 4 scan fanout: cummax / cummin backward for
// FP types. Single template, `IS_MAX` bool template parameter selects
// max-vs-min semantics (mirrors the `IsMax` pattern used in the
// pre-existing reduce-max-min backward).
//
// Walks the forward scan tracking first-occurrence argmax/argmin (PyTorch
// tie semantics). One thread per dx cell — each thread's coord-along-
// scan-axis identifies the position it owns; when the running
// argmax/argmin equals that position, the thread accumulates dy[i] into
// its dx slot.

#include "../include/baracuda_elementwise.cuh"

// Cummax — IS_MAX = true.
BARACUDA_KERNELS_SCAN_CUMMAX_MIN_BACKWARD_INSTANTIATE(
    scan_cummax_backward_f32, float, float, true)

BARACUDA_KERNELS_SCAN_CUMMAX_MIN_BACKWARD_INSTANTIATE(
    scan_cummax_backward_f16, __half, float, true)

BARACUDA_KERNELS_SCAN_CUMMAX_MIN_BACKWARD_INSTANTIATE(
    scan_cummax_backward_bf16, __nv_bfloat16, float, true)

BARACUDA_KERNELS_SCAN_CUMMAX_MIN_BACKWARD_INSTANTIATE(
    scan_cummax_backward_f64, double, double, true)

// Cummin — IS_MAX = false.
BARACUDA_KERNELS_SCAN_CUMMAX_MIN_BACKWARD_INSTANTIATE(
    scan_cummin_backward_f32, float, float, false)

BARACUDA_KERNELS_SCAN_CUMMAX_MIN_BACKWARD_INSTANTIATE(
    scan_cummin_backward_f16, __half, float, false)

BARACUDA_KERNELS_SCAN_CUMMAX_MIN_BACKWARD_INSTANTIATE(
    scan_cummin_backward_bf16, __nv_bfloat16, float, false)

BARACUDA_KERNELS_SCAN_CUMMAX_MIN_BACKWARD_INSTANTIATE(
    scan_cummin_backward_f64, double, double, false)
