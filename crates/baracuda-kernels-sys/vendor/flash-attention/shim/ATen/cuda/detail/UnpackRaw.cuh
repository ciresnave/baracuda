// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda Phase 42 PyTorch-free shim for FA2 v2.8.3.
//
// FA2's `philox_unpack.cuh` re-exports
// `at::cuda::philox::unpack(PhiloxCudaState)` returning
// `std::tuple<uint64_t, uint64_t>`. The upstream implementation
// branches on `captured_` (CUDA-graph capture mode) to pick between
// host-side `(seed_, offset_)` and device-side
// `(*offset_extragraph_, offset_intragraph_)`.
//
// Phase 42 ships dropout disabled. The `BOOL_SWITCH(p_dropout < 1.f,
// Is_dropout, ...)` in `flash_fwd_launch_template.h:175` resolves to
// `Is_dropout=false` because the launcher sets `p_dropout = 1.0f`
// (probability of keeping = 1.0 = no dropout). The `Is_dropout=true`
// branch is compiled but never reached at runtime — `unpack` is dead
// code on the device. We provide a trivial stub returning `{0, 0}` so
// the device-side compile succeeds.

#pragma once

#include <tuple>
#include <cstdint>
#include <ATen/cuda/CUDAGeneratorImpl.h>

namespace at { namespace cuda { namespace philox {

__forceinline__ __host__ __device__
std::tuple<uint64_t, uint64_t> unpack(at::PhiloxCudaState arg) {
    (void)arg;
    return std::make_tuple<uint64_t, uint64_t>(0ULL, 0ULL);
}

}}}  // namespace at::cuda::philox
