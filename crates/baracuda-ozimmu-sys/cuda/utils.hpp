// utils.hpp — shared internal helpers for the baracuda-owned ozIMMU
// translation units.
//
// What this file provides (after the Phase 44b clean-fork):
//
//   - `OZIMMU_NOT_IMPLEMENTED` — diagnostic macro used by switch
//     defaults that should be unreachable. Throws a std::runtime_error
//     with file / line / function — the throw bubbles out to the Rust
//     FFI as a non-zero status (the C wrapper in `baracuda_shim.cu`
//     converts it).
//   - `mtk::ozimmu::detail::real_type<T>::type` — type trait that
//     maps `cuDoubleComplex` to `double`, identity for everything
//     else. Used by the complex path's per-component reductions.
//   - `padded_ld<T>` / `get_slice_ld<T>` / `get_slice_num_elements<T>`
//     — leading-dimension helpers for the int8 slice buffers (each
//     slice is uint32-aligned per upstream policy; for sub-uint32
//     element types they're padded up).
//   - `ozimmu_log` / `ozimmu_error` — env-gated log helpers. The
//     env vars match the upstream names (`OZIMMU_INFO`,
//     `OZIMMU_ERROR`) so users debugging with the upstream
//     reference can use the same toggles here.
//   - `check_gemm_shape` / `check_address_alignment` — host-side
//     argument validators called by `mtk::ozimmu::gemm`.
//
// **What we removed during Phase 44b:**
//
//   - The LD_PRELOAD path (`<dlfcn.h>` / `<unistd.h>` /
//     `ozIMMU_get_function_pointer`). baracuda statically links
//     ozIMMU + cuBLAS, so the LD_PRELOAD interception that the
//     upstream library exists to provide simply isn't relevant. The
//     `OZIMMU_BARACUDA_DIRECT_LINK` preprocessor switch is gone too
//     — there's no conditional path left to gate.
//   - The `OZIMMU_BARACUDA_DIRECT_LINK` direct-call shim that the
//     Phase 44 patch installed; the call sites that used to go
//     through `ozIMMU_get_function_pointer("cublasGemmEx")` now call
//     `::cublasGemmEx` directly.

#pragma once

#include <cuComplex.h>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <ozimmu/ozimmu.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>

#define OZIMMU_NOT_IMPLEMENTED                                                 \
    mtk::ozimmu::detail::print_not_implemented(__FILE__, __LINE__, __func__)

namespace mtk {
namespace ozimmu {
namespace detail {

/// Throw a `std::runtime_error` flagging the unimplemented switch
/// branch with the source location. Used by `OZIMMU_NOT_IMPLEMENTED`;
/// inlined so the throw site shows up at the macro expansion point
/// in the backtrace.
inline void print_not_implemented(const std::string file,
                                  const std::size_t line,
                                  const std::string func) {
    throw std::runtime_error("Not implemented (" + func + " in " + file
                             + ", l." + std::to_string(line) + ")");
}

/// Map `cuDoubleComplex` to `double`; identity for everything else.
/// Used by the complex GEMM path to declare per-component scratch
/// buffers in a generic way.
template <class T>
struct real_type {
    using type = T;
};
template <>
struct real_type<cuDoubleComplex> {
    using type = double;
};

}  // namespace detail

/// Round `n` up to the next multiple of `sizeof(uint32_t) / sizeof(T)`
/// elements. The Ozaki int8 slice buffers are kept uint32-aligned so
/// the GPU's vector-load path can consume them at full width.
template <class slice_element_t>
inline std::uint32_t padded_ld(const std::uint32_t n) {
    using align_t = std::uint32_t;
    if (sizeof(align_t) >= sizeof(slice_element_t)) {
        const auto f = sizeof(align_t) / sizeof(slice_element_t);
        return ((n + f - 1) / f) * f;
    } else {
        return n;
    }
}

/// Leading dimension of an int8 slice for the (m, n, op) shape. The
/// `data_t d` overload supports the run-time-dispatched fallback used
/// by the working-memory sizer.
template <class slice_element_t>
inline std::uint32_t
get_slice_ld(const std::uint32_t m, const std::uint32_t n,
             const mtk::ozimmu::operation_t op,
             const mtk::ozimmu::data_t d = mtk::ozimmu::data_t::none) {
    if constexpr (!std::is_same_v<slice_element_t, void>) {
        return padded_ld<slice_element_t>(op == mtk::ozimmu::op_n ? m : n);
    } else {
        if (mtk::ozimmu::get_data_size_in_byte(d) == 0) {
            return 0;
        } else if (mtk::ozimmu::get_data_size_in_byte(d) == 1) {
            return get_slice_ld<std::uint8_t>(m, n, op);
        } else if (mtk::ozimmu::get_data_size_in_byte(d) == 2) {
            return get_slice_ld<std::uint16_t>(m, n, op);
        } else if (mtk::ozimmu::get_data_size_in_byte(d) == 4) {
            return get_slice_ld<std::uint32_t>(m, n, op);
        }
        OZIMMU_NOT_IMPLEMENTED;
    }

    OZIMMU_NOT_IMPLEMENTED;
    return 0;
}

/// Number of slice elements for the (m, n, op) shape — `ld * (n if N else m)`.
template <class slice_element_t>
inline std::uint32_t get_slice_num_elements(
    const std::uint32_t m, const std::uint32_t n,
    const mtk::ozimmu::operation_t op,
    const mtk::ozimmu::data_t dtype = mtk::ozimmu::data_t::none) {
    return get_slice_ld<slice_element_t>(m, n, op, dtype)
         * (op == mtk::ozimmu::op_n ? n : m);
}

}  // namespace ozimmu
}  // namespace mtk

/// Look up an env-var; return its value or `default_v` if unset.
inline std::string
ozimmu_load_env_if_defined(const std::string env_str,
                           const std::string default_v = "") {
    const auto env = std::getenv(env_str.c_str());
    if (env != nullptr) {
        return env;
    }
    return default_v;
}

/// Run `func` if the env var is set to anything other than "0"; if
/// unset, fall back to `default_v` (treat as the boolean default).
inline void ozimmu_run_if_env_defined(const std::string env_str,
                                      const std::function<void(void)> func,
                                      const bool default_v = 0) {
    const auto env = std::getenv(env_str.c_str());
    if ((env != nullptr && std::string(env) != "0") ||
        (env == nullptr && default_v)) {
        func();
    }
}

/// Print to stdout if `OZIMMU_INFO` is set (logs the slice-count
/// auto-selection path + working-memory growth events).
inline void ozimmu_log(const std::string str) {
    const std::string info_env_name = "OZIMMU_INFO";
    ozimmu_run_if_env_defined(info_env_name, [&]() {
        std::fprintf(stdout, "[ozIMMU LOG] %s\n", str.c_str());
        std::fflush(stdout);
    });
}

/// Print to stdout if `OZIMMU_ERROR` is set (defaults to ON — i.e.
/// errors are visible by default; quiet with `OZIMMU_ERROR=0`).
inline void ozimmu_error(const std::string str) {
    const std::string error_env_name = "OZIMMU_ERROR";
    ozimmu_run_if_env_defined(
        error_env_name,
        [&]() {
            std::fprintf(stdout, "[ozIMMU ERROR] %s\n", str.c_str());
            std::fflush(stdout);
        },
        1);
}

/// Verify the leading dimension `ld` of a (m, n)-shaped matrix is at
/// least as large as the contiguous-dim extent under operation `op`.
/// Returns 1 on violation and emits a log entry; 0 on success.
inline int check_gemm_shape(const mtk::ozimmu::operation_t op,
                            const std::size_t m, const std::size_t n,
                            const std::size_t ld, const std::string mat_name) {
    if ((op == mtk::ozimmu::op_n ? m : n) > ld) {
        const std::string message =
            "The leading dimension of " + mat_name + " (" + std::to_string(ld)
            + ") must be larger or equal to the number of "
            + (op == mtk::ozimmu::op_n ? "rows" : "cols") + " ("
            + std::to_string(op == mtk::ozimmu::op_n ? m : n) + ")";
        ozimmu_error(message);
        return 1;
    }
    return 0;
}

/// Verify `ptr` is aligned to `sizeof(T)`. Returns 1 on violation,
/// 0 on success. The Ozaki int8 splitter does width-promoted vector
/// loads that require this; misalignment otherwise surfaces as a
/// segfault inside a kernel.
template <class T>
inline int check_address_alignment(const T *const ptr,
                                   const std::string mat_name) {
    if (reinterpret_cast<std::uint64_t>(ptr) % sizeof(T)) {
        const std::string message = "Invalid address alignment for matrix " + mat_name;
        ozimmu_error(message);
        return 1;
    }
    return 0;
}
