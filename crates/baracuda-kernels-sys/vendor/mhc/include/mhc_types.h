// SPDX-License-Identifier: MIT
// Vendored from AndreSlavescu/mHC.cu (commit a426939c2dbc11c443db041bcff12b65d1b6482a)
// upstream: https://github.com/AndreSlavescu/mHC.cu
// Copyright (c) 2026 Andre Slavescu — MIT License (see LICENSE next to this dir)
// Brought into baracuda by Phase 43; vendored verbatim.
//
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublasLt.h>

namespace mhc {

using floatX = nv_bfloat16;
using floatN = float;

struct MHCConfig {
    int sinkhorn_iters;
    int nC;
    float eps;
    bool use_pdl;
};

struct RMSNormParams {
    int n;
    float eps;
};

inline void check_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

inline void check_cublas(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", file, line, (int)status);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(call) mhc::check_cuda((call), __FILE__, __LINE__)
#define CHECK_CUBLAS(call) mhc::check_cublas((call), __FILE__, __LINE__)
} // namespace mhc
