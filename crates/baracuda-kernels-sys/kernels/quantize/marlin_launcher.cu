// SPDX-FileCopyrightText: 2026 baracuda project contributors  (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// marlin_launcher.cu — Phase 48 C-ABI launcher for the vendored
// IST-DASLab/marlin W4A16 GEMM kernel
// (`vendor/marlin/src/marlin_cuda_kernel.cu`, Apache-2.0).
//
// Upstream Marlin exposes a single host entry point with C++ linkage,
// `int marlin_cuda(...)`. We forward-declare it here (the kernel TU
// is compiled as part of the same static archive, so the link
// resolves) and wrap it in an `extern "C"` symbol with a
// baracuda-namespaced name + an `int32_t`-typed signature matching
// the rest of the baracuda-kernels-sys FFI surface.
//
// Status code mapping mirrors the rest of the surface:
//   0 = success
//   2 = invalid problem (ERR_PROB_SHAPE from upstream — N or K
//       misalignment, or the (M-blocks, N-blocks, K-blocks, G) tuple
//       didn't match any compiled kernel configuration)
//   5 = launch failure (cudaPeekAtLastError() != cudaSuccess)
//
// Marlin's `groupsize` argument is `-1` for per-channel quant or
// `128` for the conventional group-128 quant. Other group sizes are
// rejected at the Rust plan layer; the kernel itself only contains
// instantiations for those two cases.

#include <cstdint>
#include <cuda_runtime.h>

// Forward-declare the upstream Marlin host entry point. Linkage is
// C++ (Marlin declares it in the global namespace, no `extern "C"`).
// Defaulted arguments from the upstream declaration are NOT replicated
// here — the launcher always passes every argument explicitly.
int marlin_cuda(
    const void * A,
    const void * B,
          void * C,
          void * s,
    int prob_m,
    int prob_n,
    int prob_k,
    void * workspace,
    int groupsize,
    int dev,
    cudaStream_t stream,
    int thread_k,
    int thread_n,
    int sms,
    int max_par);

namespace {

inline int32_t map_marlin_status(int upstream_status) {
    // Upstream marlin.cu defines:
    //   const int ERR_PROB_SHAPE = 1;
    //   const int ERR_KERN_SHAPE = 2;
    // We collapse both to baracuda-kernels-sys's "invalid problem"
    // status (2). Successful return is 0 from both sides; we then
    // also gate on cudaPeekAtLastError() for launch-time errors
    // (e.g. SMEM-too-small, async copy failure).
    if (upstream_status != 0) {
        return 2;
    }
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        return 5;
    }
    return 0;
}

} // anonymous namespace

// =============================================================================
// FFI surface — Marlin W4A16 GEMM (fp16 activation + output)
// =============================================================================
//
// Contract (matches Rust `Int4MarlinGemmPlan`):
//   * A `[M, K]` row-major `__half`. Element count.
//   * B `[K/16, N*16/8]` `int32` pre-shuffled int4 weights. Pack-time
//     permutation is the responsibility of the caller — use the
//     `gptq_to_marlin_repack` utility from `baracuda-kernels` for the
//     conversion from GPTQ-format weights, or the upstream Marlin
//     packer for direct-quantized weights.
//   * C `[M, N]` row-major `__half` output. Caller must allocate.
//   * s `[K/groupsize, N]` `__half` per-group scales (or `[1, N]`
//     when `groupsize == -1`), pre-permuted by the packer.
//   * workspace: `int32` buffer with `>= (N / 128) * max_par` entries,
//     zero-initialised. Caller-allocated; Marlin uses it as the
//     per-tile lock array.
//   * `groupsize`: -1 (per-channel) or 128. Other values rejected at
//     the Rust plan layer.
//   * `max_par`: parallel-tile upper bound (upstream default = 16);
//     `> M/64` is wasted work but not incorrect.
//
// All non-stream pointers must be device-allocated. `stream_ptr` is
// the caller's CUDA stream (`*mut c_void`).

extern "C" int32_t baracuda_kernels_int4_marlin_gemm_f16_run(
    int32_t M, int32_t N, int32_t K,
    const void * A,
    const void * B,
    void * C,
    const void * scales,
    void * workspace,
    int32_t groupsize,
    int32_t max_par,
    void * stream_ptr)
{
    if (M < 0 || N <= 0 || K <= 0) { return 2; }
    if (M == 0) { return 0; }
    if (!A || !B || !C || !scales || !workspace) { return 2; }
    if (groupsize != -1 && groupsize != 128) { return 2; }
    if (max_par <= 0) { return 2; }

    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    // The upstream kernel mutates its `s` argument's effective type to
    // `int4 *` via a reinterpret cast; the void* parameter accepts both
    // const and non-const data. We forward as-is — the kernel itself
    // does not write through this pointer.
    void * s_ptr = const_cast<void *>(scales);

    int status = marlin_cuda(
        A, B, C, s_ptr,
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        workspace,
        static_cast<int>(groupsize),
        /* dev      = */ 0,    // 0 → upstream queries cudaDeviceGetAttribute internally
        stream,
        /* thread_k = */ -1,   // -1 → upstream auto-selects per-M heuristic
        /* thread_n = */ -1,
        /* sms      = */ -1,   // -1 → upstream queries cudaDevAttrMultiProcessorCount
        static_cast<int>(max_par));

    return map_marlin_status(status);
}

// can_implement — pure shape/alignment validation (no kernel launch).
//
// Returns 0 if the shape is in the supported range, 2 otherwise.
// Mirrors the upstream `Layer.__init__` validation in the Python
// reference: K must be divisible by 128 (the kernel's K_BLOCK lower
// bound) and N must be divisible by 256 (the kernel's
// `thread_n * parallel-tile` lower bound).

extern "C" int32_t baracuda_kernels_int4_marlin_gemm_f16_can_implement(
    int32_t M, int32_t N, int32_t K,
    int32_t groupsize)
{
    if (M < 0 || N <= 0 || K <= 0)             { return 2; }
    if ((K % 128) != 0)                        { return 2; }
    if ((N % 256) != 0)                        { return 2; }
    if (groupsize != -1 && groupsize != 128)   { return 2; }
    if (groupsize == 128 && (K % 128) != 0)    { return 2; }
    return 0;
}
