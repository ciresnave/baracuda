// handle.hpp — internal state of an ozIMMU session.
//
// Holds the cuBLAS handle, the bound stream, the working-memory
// scratch pointer + size, and the mantissa-loss histogram buffer used
// by `auto_mode_select`. The profiler that upstream included here
// (`cutf::debug::time_breakdown::profiler`) is removed in baracuda —
// we have crate-level benchmarks (`baracuda-kernels-bench`) instead
// of an embedded per-handle profiler.

#pragma once

#include <cublas_v2.h>
#include <ozimmu/ozimmu.hpp>

struct mtk::ozimmu::handle {
    // Handles
    cublasHandle_t cublas_handle;
    cudaStream_t cuda_stream;

    // Working memory
    void *working_memory_ptr;
    std::size_t current_working_memory_size;

    // Allocator mode (sync vs async)
    malloc_mode_t malloc_mode;

    // Auto-mode mantissa-loss histogram (covers slice counts 6..13
    // = the `fp64_int8_6` .. `fp64_int8_13` modes; index 0 of the
    // buffer = `fp64_int8_3` per the upstream convention, which
    // doesn't match the enum span — preserved here for bit-for-bit
    // compatibility with the published paper's tables).
    enum { mantissa_loss_counter_length = 13 - 6 + 1 };
    unsigned long long int *d_mantissa_loss_counter_ptr;
    compute_mode_t last_auto_mode = mtk::ozimmu::dgemm;

    double avg_mantissa_loss_threshold = 0;
};

namespace mtk {
namespace ozimmu {

/// Wrapper around `cublasCreate_v2` — the baracuda shim TU implements
/// this in `baracuda_shim.cu` (formerly the LD_PRELOAD indirection
/// hook upstream). Kept as a function so we have one place to add
/// retry-on-failure logic if cuBLAS init becomes flaky again
/// (recurrence of the Phase 35 cuBLAS init race).
cublasStatus_t cublasCreate_org(cublasHandle_t *handle_ptr);

/// Wrapper around `cublasDestroy_v2`. Symmetric to `cublasCreate_org`.
cublasStatus_t cublasDestroy_org(cublasHandle_t handle_ptr);

}  // namespace ozimmu
}  // namespace mtk
