// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda Phase 42 PyTorch-free shim for FA2 v2.8.3.
//
// FA2's `flash_fwd_launch_template.h` uses two macros from PyTorch's c10:
//
//   C10_CUDA_CHECK(cmd)             — wraps a `cudaError_t`-returning
//                                     call; throws on non-success.
//   C10_CUDA_KERNEL_LAUNCH_CHECK()  — checks `cudaGetLastError()`;
//                                     throws on non-success.
//
// PyTorch's implementations throw `c10::CUDAError`. We translate to a
// `fprintf(stderr, ...); std::abort();` to match the upstream macro's
// fail-fast intent without dragging in exception handling. This is
// acceptable for baracuda's FA2 path because the host launcher in
// `kernels/attention/fa2_launcher.cu` is always called from a context
// that already has its own error handling — a panic-equivalent here
// is no worse than the upstream behaviour for unhandled errors.

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef C10_CUDA_CHECK
#define C10_CUDA_CHECK(cmd)                                                    \
    do {                                                                       \
        cudaError_t _err = (cmd);                                              \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr,                                               \
                "baracuda FA2 shim: CUDA error %d (%s) at %s:%d\n",            \
                int(_err), cudaGetErrorString(_err), __FILE__, __LINE__);      \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
#endif

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK()                                         \
    do {                                                                       \
        cudaError_t _err = cudaGetLastError();                                 \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr,                                               \
                "baracuda FA2 shim: kernel launch failed %d (%s) at %s:%d\n",  \
                int(_err), cudaGetErrorString(_err), __FILE__, __LINE__);      \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
#endif
