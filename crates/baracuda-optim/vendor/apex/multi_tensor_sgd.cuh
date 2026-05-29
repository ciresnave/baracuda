// Copyright (c) 2017-2025 NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//   * Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//   * Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//   * Neither the name of the NVIDIA CORPORATION, nor the names of their
//     contributors may be used to endorse or promote products derived
//     from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. (full text in LICENSE next to
// this file)

// baracuda Phase 49 — Apex multi_tensor_sgd vendored functor.
//
// SGD with momentum + Nesterov + weight decay. The PyTorch / Apex
// reference update (matches torch.optim.SGD exactly):
//
//   if weight_decay != 0:
//       g = g + weight_decay * w
//   if momentum != 0:
//       if first_step:
//           v = g
//       else:
//           v = momentum * v + (1 - dampening) * g
//       if nesterov:
//           g = g + momentum * v
//       else:
//           g = v
//   w = w - lr * g
//
// Apex's `multi_tensor_sgd_cuda` also supports a "param_grad_scale"
// (used by mixed-precision GradScaler), which we expose as a single
// f32 multiplier on the gradient before the weight-decay term.

#pragma once

#include <cuda_runtime.h>
#include "multi_tensor_apply.cuh"

namespace baracuda_apex {

// Vendored device-side SGD functor.
//
// Slot layout (depth = 3):
//   0: param          (in/out)
//   1: grad           (in)
//   2: momentum_buf   (in/out — may be uninitialized on first step;
//                      `first_run` controls whether it's read or
//                      seeded from `grad`)
template<typename GRAD_T, typename PARAM_T, typename MOMENT_T>
struct SgdFunctor
{
  __device__ __forceinline__ void operator()(
      int chunk_size,
      TensorListMetadata<3>& tl,
      const float weight_decay,
      const float momentum,
      const float dampening,
      const float lr,
      const bool  nesterov,
      const bool  first_run,
      const bool  wd_after_momentum,   // Apex flag — if true, decay is applied to the velocity instead of grad
      const float grad_scale)
  {
    const int tensor_loc = tl.block_to_tensor[blockIdx.x];
    const int chunk_loc  = tl.block_to_chunk[blockIdx.x];
    const int n          = tl.sizes[tensor_loc];

    PARAM_T*  p = reinterpret_cast<PARAM_T*>(tl.addresses[0][tensor_loc]) + chunk_loc * chunk_size;
    GRAD_T*   g = reinterpret_cast<GRAD_T* >(tl.addresses[1][tensor_loc]) + chunk_loc * chunk_size;
    MOMENT_T* v = reinterpret_cast<MOMENT_T*>(tl.addresses[2][tensor_loc]) + chunk_loc * chunk_size;

    const int n_this = (chunk_size < n - chunk_loc * chunk_size)
                     ? chunk_size : (n - chunk_loc * chunk_size);

    for (int i_start = 0; i_start < n_this; i_start += BLOCK_SIZE * ILP) {
      #pragma unroll
      for (int ii = 0; ii < ILP; ++ii) {
        const int i = i_start + threadIdx.x + ii * BLOCK_SIZE;
        if (i < n_this) {
          float gi = static_cast<float>(g[i]) * grad_scale;
          float pi = static_cast<float>(p[i]);

          if (!wd_after_momentum && weight_decay != 0.0f) {
            gi = gi + weight_decay * pi;
          }

          float update;
          if (momentum != 0.0f) {
            float vi;
            if (first_run) {
              // First step: seed momentum with the (possibly decayed) grad.
              vi = gi;
            } else {
              vi = static_cast<float>(v[i]);
              vi = momentum * vi + (1.0f - dampening) * gi;
            }
            v[i] = static_cast<MOMENT_T>(vi);

            if (wd_after_momentum && weight_decay != 0.0f) {
              vi = vi + weight_decay * pi;
            }

            update = nesterov ? (gi + momentum * vi) : vi;
          } else {
            update = gi;
          }

          pi = pi - lr * update;
          p[i] = static_cast<PARAM_T>(pi);
        }
      }
    }
  }
};

} // namespace baracuda_apex
