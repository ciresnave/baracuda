// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Shared bin-MMA building block used by both the RCR and RRR bin
// kernels. Lives in its own header so the RRR .cu translation unit
// does not pull in the RCR kernel's `__global__` definition (which
// would cause a duplicate-symbol link error).

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace baracuda {

// `mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc` wrapper.
// A is 4×b32, B is 2×b32, D accum is 4×s32 (raw popcount).
__device__ __forceinline__ void mma_m16n8k256_xor_popc(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int32_t &d0, int32_t &d1, int32_t &d2, int32_t &d3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );
}

} // namespace baracuda
