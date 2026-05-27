// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Phase 37 Gap 1b — axis-min for integer dtypes.
//
// `y = min(x, dim=k)` with keepdim=true. Same-dtype throughout (no
// widening needed — the natural min of integers never overflows). The
// init value is the dtype's max representable (so the first real
// element always wins on the first iteration).
//
// Coverage: u8, i8, u32, i16, i32, i64.

#include "../include/baracuda_elementwise.cuh"

#include <climits>
#include <cstdint>

namespace baracuda { namespace elementwise {

template <typename T>
struct MinReduceInt {
    static __device__ __forceinline__ T init();
    static __device__ __forceinline__ T finalize(T acc, int32_t /*extent*/) { return acc; }
    __device__ __forceinline__ T operator()(T acc, T x) const { return (x < acc) ? x : acc; }
};

template <> __device__ __forceinline__ uint8_t  MinReduceInt<uint8_t>::init()  { return UINT8_MAX;  }
template <> __device__ __forceinline__ int8_t   MinReduceInt<int8_t>::init()   { return INT8_MAX;   }
template <> __device__ __forceinline__ uint32_t MinReduceInt<uint32_t>::init() { return UINT32_MAX; }
template <> __device__ __forceinline__ int16_t  MinReduceInt<int16_t>::init()  { return INT16_MAX;  }
template <> __device__ __forceinline__ int32_t  MinReduceInt<int32_t>::init()  { return INT32_MAX;  }
template <> __device__ __forceinline__ int64_t  MinReduceInt<int64_t>::init()  { return INT64_MAX;  }

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_min_u8,  uint8_t,  baracuda::elementwise::MinReduceInt<uint8_t>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_min_i8,  int8_t,   baracuda::elementwise::MinReduceInt<int8_t>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_min_u32, uint32_t, baracuda::elementwise::MinReduceInt<uint32_t>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_min_i16, int16_t,  baracuda::elementwise::MinReduceInt<int16_t>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_min_i32, int32_t,  baracuda::elementwise::MinReduceInt<int32_t>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_min_i64, int64_t,  baracuda::elementwise::MinReduceInt<int64_t>)
