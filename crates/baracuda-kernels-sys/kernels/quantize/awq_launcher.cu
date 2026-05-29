// SPDX-FileCopyrightText: 2026 baracuda project contributors  (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// awq_launcher.cu — Phase 48 C-ABI launcher for the vendored
// mit-han-lab/llm-awq W4A16 GEMM kernel
// (`vendor/awq/src/gemm_cuda_gen.cu`, MIT).
//
// Upstream AWQ's `gemm_forward_4bit_cuda_m128n64k32<G>` is a
// `__global__` template (parameterised on the per-group element count
// `G ∈ {64, 128}`). The launcher inline-includes the vendored .cu
// to bring the template definition into the same TU, then exposes a
// flat C-ABI surface keyed on `(G, op)`.
//
// The vendored .cu has been patched to strip the upstream PyTorch
// host wrapper (`gemm_forward_cuda`) and the torch / c10 #includes;
// only the device kernel + the two device helpers
// (`__pack_half2`, `make_divisible`) remain. This launcher provides
// the replacement host-side wrapper:
//
//   1. Allocate a `[split_k_iters, M, OC]` staging buffer for the
//      partial sums from the caller's workspace.
//   2. Launch the kernel.
//   3. Run an axis-0 reduce-sum (split_k_iters → 1) into the final
//      `[M, OC]` output tensor.
//
// AWQ is M-agnostic (the kernel handles any M; it's just optimised
// for M < 16). We expose `_run` for the general case + a `_dequant`
// stub that AWQ doesn't ship natively (deferred). The launcher's
// `workspace_bytes` query returns the required staging-buffer size
// in bytes.
//
// Status codes (matches the rest of baracuda-kernels-sys):
//   0 = success
//   2 = invalid problem (bad alignment, group_size != {64, 128}, ...)
//   4 = workspace too small
//   5 = launch failure
//
// All shape / alignment invariants are caller-validated up in the
// Rust plan layer; this layer re-checks the pointer-non-null and the
// kernel's own divisibility invariants.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Inline-include the vendored .cu (now header-shaped after the
// Phase 48 patch). This is the same pattern the bnb_nf4 launcher
// uses for its kernel headers — co-compile the template definition
// in the launcher TU. The vendored .cu is therefore NOT listed in
// `build.rs`'s standalone source list; only this launcher is.
#include "../../vendor/awq/src/gemm_cuda_gen.cu"

namespace {

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

inline int32_t status_from_launch(cudaError_t err) {
    return (err != cudaSuccess) ? 5 : 0;
}

// Sum-reduce axis 0 of `[split_k, M, OC]` into `[M, OC]` (fp16).
// One thread per (m, oc) cell; simple loop over split_k.
__global__ void awq_split_k_reduce_kernel(
    const __half * __restrict__ staging,   // [split_k, M, OC]
    __half * __restrict__ out,             // [M, OC]
    int split_k,
    int M,
    int OC)
{
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    int m  = blockIdx.y * blockDim.y + threadIdx.y;
    if (oc >= OC || m >= M) return;
    float acc = 0.0f;
    long long row_stride = static_cast<long long>(M) * OC;
    for (int s = 0; s < split_k; ++s) {
        long long off = static_cast<long long>(s) * row_stride
                      + static_cast<long long>(m) * OC + oc;
        acc += __half2float(staging[off]);
    }
    out[static_cast<long long>(m) * OC + oc] = __float2half(acc);
}

} // anonymous namespace

// =============================================================================
// Workspace-size query — caller allocates `[split_k_iters, M, OC]` fp16 bytes.
// =============================================================================
extern "C" size_t baracuda_kernels_int4_awq_gemm_f16_workspace_bytes(
    int32_t M, int32_t OC, int32_t split_k_iters)
{
    if (M <= 0 || OC <= 0 || split_k_iters <= 0) return 0;
    // Pad M to 128 to match the kernel's block-M shape (kernel writes
    // per-128-row blocks; partial tiles still allocate the full block
    // of staging space).
    int padded_M = ((M + 127) / 128) * 128;
    size_t bytes = static_cast<size_t>(split_k_iters) *
                   static_cast<size_t>(padded_M) *
                   static_cast<size_t>(OC) *
                   sizeof(__half);
    return bytes;
}

// =============================================================================
// can_implement — pure shape / alignment validation.
// =============================================================================
extern "C" int32_t baracuda_kernels_int4_awq_gemm_f16_can_implement(
    int32_t M, int32_t IC, int32_t OC,
    int32_t group_size, int32_t split_k_iters)
{
    if (M < 0 || IC <= 0 || OC <= 0)                      { return 2; }
    if (group_size != 64 && group_size != 128)            { return 2; }
    if ((OC % 64) != 0)                                   { return 2; }   // cta_N
    if ((OC % 8)  != 0)                                   { return 2; }   // pack_num
    if ((IC % group_size) != 0)                           { return 2; }
    if (split_k_iters <= 0)                               { return 2; }
    if ((IC % (32 * split_k_iters)) != 0) {
        // The kernel iterates K in chunks of 32 per split_k step;
        // require even division so every block writes its full share.
        // Upstream's launcher silently truncates with `make_divisible`;
        // we reject here for clarity.
        return 2;
    }
    return 0;
}

// =============================================================================
// FFI surface — AWQ W4A16 GEMM (fp16 activation + output, fp32 accumulator)
// =============================================================================
//
// Contract (matches Rust `Int4AwqGemmPlan`):
//   * in_feats  `[M, IC]` row-major `__half`.
//   * kernel    `[OC, IC/8]` `int32` packed int4 (8 nibbles per
//                int32 word, OC-major IC-minor — note this is the
//                transpose of the naive `[K, N]`).
//   * scaling   `[IC/group_size, OC]` `__half`.
//   * zeros     `[IC/group_size, OC/8]` `int32` packed int4
//                (8 zero-points per int32 word).
//   * out       `[M, OC]` row-major `__half`.
//   * workspace `[split_k_iters, padded_M, OC]` fp16 staging;
//                `padded_M = ceil(M, 128) * 128`. Sized via
//                `baracuda_kernels_int4_awq_gemm_f16_workspace_bytes`.
//   * group_size: 64 or 128.
//   * split_k_iters: caller-chosen; typically 8.

extern "C" int32_t baracuda_kernels_int4_awq_gemm_f16_run(
    int32_t M, int32_t IC, int32_t OC,
    int32_t group_size, int32_t split_k_iters,
    const void * in_feats,
    const void * kernel_weights,
    const void * scaling_factors,
    const void * zeros,
    void * out,
    void * workspace,
    size_t workspace_bytes,
    void * stream_ptr)
{
    int32_t s = baracuda_kernels_int4_awq_gemm_f16_can_implement(
        M, IC, OC, group_size, split_k_iters);
    if (s != 0) return s;
    if (M == 0) return 0;
    if (!in_feats || !kernel_weights || !scaling_factors || !zeros || !out || !workspace) {
        return 2;
    }
    size_t need = baracuda_kernels_int4_awq_gemm_f16_workspace_bytes(M, OC, split_k_iters);
    if (workspace_bytes < need) {
        return 4;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    __half * staging = reinterpret_cast<__half *>(workspace);
    int padded_M = ((M + 127) / 128) * 128;

    // Zero the staging buffer (the kernel skips rows past M, so the
    // padded tail must be zero-initialised for the reduce step).
    cudaError_t merr = cudaMemsetAsync(
        staging, 0,
        static_cast<size_t>(split_k_iters) *
        static_cast<size_t>(padded_M) *
        static_cast<size_t>(OC) *
        sizeof(__half),
        stream);
    if (merr != cudaSuccess) return 5;

    const __half * in_h    = static_cast<const __half *>(in_feats);
    const int    * kern_i  = static_cast<const int    *>(kernel_weights);
    const __half * sfac_h  = static_cast<const __half *>(scaling_factors);
    const int    * zero_i  = static_cast<const int    *>(zeros);

    int j_factors1 = OC / 64;
    dim3 num_blocks(static_cast<unsigned int>(
        ceil_div(M, 128) * j_factors1 * split_k_iters));
    dim3 threads_per_block(32, 4);

    if (group_size == 128) {
        gemm_forward_4bit_cuda_m128n64k32<128>
            <<<num_blocks, threads_per_block, 0, stream>>>(
                split_k_iters,
                const_cast<__half *>(in_h),
                const_cast<int *>(kern_i),
                const_cast<__half *>(sfac_h),
                const_cast<int *>(zero_i),
                M, IC, OC,
                staging);
    } else {
        // group_size == 64 (validated above).
        gemm_forward_4bit_cuda_m128n64k32<64>
            <<<num_blocks, threads_per_block, 0, stream>>>(
                split_k_iters,
                const_cast<__half *>(in_h),
                const_cast<int *>(kern_i),
                const_cast<__half *>(sfac_h),
                const_cast<int *>(zero_i),
                M, IC, OC,
                staging);
    }
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) return 5;

    // Reduce sum axis 0 of [split_k, M, OC] -> [M, OC].
    __half * out_h = static_cast<__half *>(out);
    dim3 rblock(32, 4, 1);
    dim3 rgrid(
        static_cast<unsigned int>(ceil_div(OC, 32)),
        static_cast<unsigned int>(ceil_div(M, 4)),
        1);
    awq_split_k_reduce_kernel<<<rgrid, rblock, 0, stream>>>(
        staging, out_h, split_k_iters, M, OC);
    return status_from_launch(cudaPeekAtLastError());
}

// =============================================================================
// Dequant FFI — AWQ does not ship a standalone dequant kernel
// upstream (the dequant lives inside the GEMM as a per-tile staging
// step). For a host-side dequant reference, callers can launch the
// GEMM with `in_feats` being the identity matrix; or use baracuda's
// generic `DequantizePerGroupPlan` after a one-time format conversion.
//
// This stub returns "unsupported" (status 3) so the FFI surface is
// stable but the dequant path is explicitly absent.
// =============================================================================

extern "C" int32_t baracuda_kernels_int4_awq_dequantize_f16_run(
    int32_t N, int32_t K, int32_t group_size,
    const void * kernel_weights,
    const void * scaling_factors,
    const void * zeros,
    void * out,
    void * stream_ptr)
{
    (void)N; (void)K; (void)group_size;
    (void)kernel_weights; (void)scaling_factors; (void)zeros;
    (void)out; (void)stream_ptr;
    // 3 = unsupported configuration (mapped by the Rust map_status
    // helper to Error::Unsupported).
    return 3;
}
