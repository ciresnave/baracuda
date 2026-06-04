// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Phase 37 Gap 1b — axis-max for integer dtypes.
//
// `y = max(x, dim=k)` with keepdim=true. Same-dtype throughout. Init
// is the dtype's min representable (so the first real element always
// wins on the first iteration).
//
// Coverage: u8, i8, u32, i16, i32, i64.

#include "../include/baracuda_elementwise.cuh"

#include <climits>
#include <cstdint>

namespace baracuda { namespace elementwise {

template <typename T>
struct MaxReduceInt {
    static __device__ __forceinline__ T init();
    static __device__ __forceinline__ T finalize(T acc, int32_t /*extent*/) { return acc; }
    __device__ __forceinline__ T operator()(T acc, T x) const { return (x > acc) ? x : acc; }
    static __device__ __forceinline__ T merge(T a, T b) { return (b > a) ? b : a; }
};

template <> __device__ __forceinline__ uint8_t  MaxReduceInt<uint8_t>::init()  { return 0;          }
template <> __device__ __forceinline__ int8_t   MaxReduceInt<int8_t>::init()   { return INT8_MIN;   }
template <> __device__ __forceinline__ uint32_t MaxReduceInt<uint32_t>::init() { return 0;          }
template <> __device__ __forceinline__ int16_t  MaxReduceInt<int16_t>::init()  { return INT16_MIN;  }
template <> __device__ __forceinline__ int32_t  MaxReduceInt<int32_t>::init()  { return INT32_MIN;  }
template <> __device__ __forceinline__ int64_t  MaxReduceInt<int64_t>::init()  { return INT64_MIN;  }

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_max_u8,  uint8_t,  baracuda::elementwise::MaxReduceInt<uint8_t>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_max_i8,  int8_t,   baracuda::elementwise::MaxReduceInt<int8_t>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_max_u32, uint32_t, baracuda::elementwise::MaxReduceInt<uint32_t>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_max_i16, int16_t,  baracuda::elementwise::MaxReduceInt<int16_t>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_max_i32, int32_t,  baracuda::elementwise::MaxReduceInt<int32_t>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_max_i64, int64_t,  baracuda::elementwise::MaxReduceInt<int64_t>)
