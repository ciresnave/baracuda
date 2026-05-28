// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda Phase 42 PyTorch-free shim for FA2 v2.8.3.
//
// FA2's `flash.h` includes `<ATen/cuda/CUDAGeneratorImpl.h>` to pull in
// `at::PhiloxCudaState` — a trivially-copyable struct that holds the
// philox RNG state for dropout. Phase 42 ships dropout disabled, so
// the field is never read at runtime, but the include and the struct
// declaration must satisfy the C++ compiler.
//
// Upstream definition (PyTorch ATen/cuda/CUDAGeneratorImpl.h):
//   struct PhiloxCudaState {
//     uint64_t seed_;
//     int64_t offset_;
//     bool captured_ = false;
//     int64_t* offset_extragraph_ = nullptr;
//     uint32_t offset_intragraph_ = 0;
//   };
//
// We use the minimum subset FA2 touches via `params.philox_args = ...`
// assignment from `flash_fwd_kernel.h:69` (only via the
// `at::cuda::philox::unpack` shim).

#pragma once

#include <cstdint>

namespace at {

struct PhiloxCudaState {
    uint64_t seed_{0};
    int64_t  offset_{0};
    bool     captured_{false};
    int64_t* offset_extragraph_{nullptr};
    uint32_t offset_intragraph_{0};
};

}  // namespace at
