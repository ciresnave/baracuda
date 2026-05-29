// handle.cu — RAII / lifecycle for `mtk::ozimmu::handle`.
//
// Allocates the embedded cuBLAS handle, the working-memory scratch
// buffer, and the mantissa-loss histogram. Lifecycle hooks
// (create/destroy/set_stream/reallocate) are all here.
//
// Phase 44b clean-fork notes:
//   - Removed `<cutf/device.hpp>` include (no device-iteration code
//     in this TU actually used it).
//   - Removed the upstream profiler — `disable_profiling` /
//     `print_profiler_result` / etc. are gone with it (baracuda has
//     crate-level benchmarks instead).
//   - Removed `intercept_threshold_{m,n,k}` env-var reads — those
//     were only meaningful under LD_PRELOAD interception, where ozIMMU
//     decided whether to bypass itself based on shape thresholds.
//     Under direct linking the caller already chose ozIMMU
//     intentionally.

#include "config.hpp"
#include "handle.hpp"
#include "utils.hpp"

int mtk::ozimmu::create(mtk::ozimmu::handle_t *h,
                        mtk::ozimmu::malloc_mode_t mm) {
    ozimmu_log("Initializing ozIMMU handle");
    auto handle = (*h = new mtk::ozimmu::handle);

    const auto cu_status = cublasCreate_org(&(handle->cublas_handle));
    if (cu_status != CUBLAS_STATUS_SUCCESS) {
        delete handle;
        *h = nullptr;
        ozimmu_error("cuBLAS handle creation failed in mtk::ozimmu::create");
        return 1;
    }

    handle->current_working_memory_size = 0;
    handle->working_memory_ptr = nullptr;
    handle->malloc_mode = mm;
    handle->cuda_stream = 0;

    const auto ma_status =
        cudaMalloc(&(handle->d_mantissa_loss_counter_ptr),
                   sizeof(unsigned long long int)
                       * handle->mantissa_loss_counter_length);
    if (ma_status != cudaSuccess) {
        cublasDestroy_org(handle->cublas_handle);
        delete handle;
        *h = nullptr;
        ozimmu_error(std::string("cudaMalloc(mantissa_loss_counter) failed: ")
                     + cudaGetErrorString(ma_status));
        return 2;
    }

    return 0;
}

int mtk::ozimmu::destroy(mtk::ozimmu::handle_t handle) {
    if (handle) {
        ozimmu_log("Destroying ozIMMU handle");
        cublasDestroy_org(handle->cublas_handle);

        if (handle->working_memory_ptr != nullptr) {
            cudaFree(handle->working_memory_ptr);
            handle->working_memory_ptr = nullptr;
        }

        cudaFree(handle->d_mantissa_loss_counter_ptr);
        handle->d_mantissa_loss_counter_ptr = nullptr;

        delete handle;
        handle = nullptr;
    }
    return 0;
}

void mtk::ozimmu::set_cuda_stream(mtk::ozimmu::handle_t handle,
                                  cudaStream_t cuda_stream) {
    cublasSetStream(handle->cublas_handle, cuda_stream);
    handle->cuda_stream = cuda_stream;
}

std::size_t
mtk::ozimmu::reallocate_working_memory(mtk::ozimmu::handle_t handle,
                                       const std::size_t size_in_byte) {
    if (size_in_byte > handle->current_working_memory_size) {
        handle->current_working_memory_size = size_in_byte;

        ozimmu_log("Reallocated memory : " + std::to_string(size_in_byte) + " B");

        if (handle->working_memory_ptr != nullptr) {
            if (handle->malloc_mode == mtk::ozimmu::malloc_sync) {
                cudaFree(handle->working_memory_ptr);
            } else {
                cudaFreeAsync(handle->working_memory_ptr, handle->cuda_stream);
            }
        }

        if (handle->malloc_mode == mtk::ozimmu::malloc_sync) {
            cudaMalloc(&(handle->working_memory_ptr),
                       handle->current_working_memory_size);
        } else {
            cudaMallocAsync(&(handle->working_memory_ptr),
                            handle->current_working_memory_size,
                            handle->cuda_stream);
        }
        return size_in_byte;
    }
    return 0;
}

std::size_t mtk::ozimmu::reallocate_working_memory(
    mtk::ozimmu::handle_t handle, const mtk::ozimmu::gemm_list_t gemm_list) {
    std::size_t max_working_memory_size = 0;

    for (const auto gemm : gemm_list) {
        const auto op_A = std::get<0>(gemm);
        const auto op_B = std::get<1>(gemm);
        const auto m = std::get<2>(gemm);
        const auto n = std::get<3>(gemm);
        const auto k = std::get<4>(gemm);
        const auto element_kind = std::get<5>(gemm);
        const auto mode = std::get<6>(gemm);

        const auto working_memory_A =
            mtk::ozimmu::detail::calculate_working_memory_size(
                op_A, m, k, mode, detail::matrix_A, element_kind);
        const auto working_memory_B =
            mtk::ozimmu::detail::calculate_working_memory_size(
                op_B, k, n, mode, detail::matrix_B, element_kind);
        const auto working_memory_C_fp32 =
            m * n * mtk::ozimmu::get_data_size_in_byte(fp32);
        const auto working_memory_C_fp64 =
            m * n * mtk::ozimmu::get_data_size_in_byte(fp64)
            * (element_kind == mtk::ozimmu::real ? 1 : 2);
        std::size_t etc = 0;
        // For the int8-tensor-core modes, reserve an extra `(m + n) * sizeof(fp64)`
        // for the per-row max-exponent scratch.
        if (mode >= mtk::ozimmu::fp64_int8_3 && mode <= mtk::ozimmu::fp64_int8_18) {
            etc = (m + n) * mtk::ozimmu::get_data_size_in_byte(fp64)
                  * (element_kind == mtk::ozimmu::real ? 1 : 2);
        }

        const auto total = working_memory_A + working_memory_B
                         + working_memory_C_fp32 + working_memory_C_fp64 + etc;
        max_working_memory_size = std::max(max_working_memory_size, total);
    }

    return mtk::ozimmu::reallocate_working_memory(handle, max_working_memory_size);
}

std::string
mtk::ozimmu::get_compute_mode_name_str(const mtk::ozimmu::compute_mode_t mode) {
    switch (mode) {
    case mtk::ozimmu::sgemm:           return "sgemm";
    case mtk::ozimmu::dgemm:           return "dgemm";
    case mtk::ozimmu::fp64_int8_3:     return "fp64_int8_3";
    case mtk::ozimmu::fp64_int8_4:     return "fp64_int8_4";
    case mtk::ozimmu::fp64_int8_5:     return "fp64_int8_5";
    case mtk::ozimmu::fp64_int8_6:     return "fp64_int8_6";
    case mtk::ozimmu::fp64_int8_7:     return "fp64_int8_7";
    case mtk::ozimmu::fp64_int8_8:     return "fp64_int8_8";
    case mtk::ozimmu::fp64_int8_9:     return "fp64_int8_9";
    case mtk::ozimmu::fp64_int8_10:    return "fp64_int8_10";
    case mtk::ozimmu::fp64_int8_11:    return "fp64_int8_11";
    case mtk::ozimmu::fp64_int8_12:    return "fp64_int8_12";
    case mtk::ozimmu::fp64_int8_13:    return "fp64_int8_13";
    case mtk::ozimmu::fp64_int8_14:    return "fp64_int8_14";
    case mtk::ozimmu::fp64_int8_15:    return "fp64_int8_15";
    case mtk::ozimmu::fp64_int8_16:    return "fp64_int8_16";
    case mtk::ozimmu::fp64_int8_17:    return "fp64_int8_17";
    case mtk::ozimmu::fp64_int8_18:    return "fp64_int8_18";
    case mtk::ozimmu::fp64_int8_auto:  return "fp64_int8_auto";
    default:
        break;
    }
    OZIMMU_NOT_IMPLEMENTED;
    return "";
}

mtk::ozimmu::data_t
mtk::ozimmu::get_output_type(const mtk::ozimmu::compute_mode_t compute_mode) {
    switch (compute_mode) {
    case mtk::ozimmu::sgemm:
        return mtk::ozimmu::fp32;

    case mtk::ozimmu::fp64_int8_3:
    case mtk::ozimmu::fp64_int8_4:
    case mtk::ozimmu::fp64_int8_5:
    case mtk::ozimmu::fp64_int8_6:
    case mtk::ozimmu::fp64_int8_7:
    case mtk::ozimmu::fp64_int8_8:
    case mtk::ozimmu::fp64_int8_9:
    case mtk::ozimmu::fp64_int8_10:
    case mtk::ozimmu::fp64_int8_11:
    case mtk::ozimmu::fp64_int8_12:
    case mtk::ozimmu::fp64_int8_13:
    case mtk::ozimmu::fp64_int8_14:
    case mtk::ozimmu::fp64_int8_15:
    case mtk::ozimmu::fp64_int8_16:
    case mtk::ozimmu::fp64_int8_17:
    case mtk::ozimmu::fp64_int8_18:
    case mtk::ozimmu::fp64_int8_auto:
    case mtk::ozimmu::dgemm:
        return mtk::ozimmu::fp64;

    default:
        break;
    }
    OZIMMU_NOT_IMPLEMENTED;
    return mtk::ozimmu::original;
}

std::size_t mtk::ozimmu::get_data_size_in_byte(const mtk::ozimmu::data_t d) {
    switch (d) {
    case mtk::ozimmu::fp64:     return 8;
    case mtk::ozimmu::fp32:     return 4;
    case mtk::ozimmu::fp16:     return 2;
    case mtk::ozimmu::original: return 0;
    case mtk::ozimmu::int8:     return 1;
    default:
        OZIMMU_NOT_IMPLEMENTED;
        break;
    }
    return 0;
}

void mtk::ozimmu::set_auto_mantissa_loss_threashold(
    mtk::ozimmu::handle_t handle, const double threshold) {
    handle->avg_mantissa_loss_threshold = threshold;
}

double get_auto_mantissa_loss_threashold(mtk::ozimmu::handle_t handle) {
    return handle->avg_mantissa_loss_threshold;
}
