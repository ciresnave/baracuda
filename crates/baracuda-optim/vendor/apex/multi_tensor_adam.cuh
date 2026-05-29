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

// baracuda Phase 49 — Apex multi_tensor_adam vendored functor.
//
// This is the device-side Adam update functor extracted from Apex's
// `csrc/multi_tensor_adam.cu`. The original .cu's HOST-side launcher
// (`multi_tensor_adam_cuda(at::Tensor ...)`) is intentionally NOT
// vendored — baracuda's shim takes raw device pointer arrays and
// launches this functor through its own kernel template.
//
// Update rule (Adam):
//   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
//   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
//   m_hat = m_t / (1 - beta1^t)
//   v_hat = v_t / (1 - beta2^t)
//   w_t = w_{t-1} - lr * m_hat / (sqrt(v_hat) + eps) - lr * weight_decay * w_{t-1}
//
// `adamw_mode = 1` switches the weight-decay term from L2-regularization
// (added INTO the gradient before the Adam math) to AdamW-style decoupled
// decay (subtracted from the weight AFTER the Adam update).
//
// `bias_correction` switches the m_hat / v_hat normalization on/off.
// When off, the caller is expected to scale `lr` themselves
// (e.g. `lr * sqrt(1 - beta2^t) / (1 - beta1^t)`).

#pragma once

#include <cuda_runtime.h>
#include "multi_tensor_apply.cuh"

namespace baracuda_apex {

// adamMode_t — matches Apex's enum verbatim.
enum class adamMode_t {
  ADAM  = 0,   // classic Adam (weight decay folded into gradient as L2)
  ADAMW = 1,   // AdamW (weight decay decoupled)
};

// Vendored from Apex csrc/multi_tensor_adam.cu — device-side functor.
//
// Template params:
//   GRAD_T  — gradient element type (f32 / __half / __nv_bfloat16)
//   PARAM_T — parameter element type (typically f32 master weights;
//             can be __half / __nv_bfloat16 for memory-bound configs)
//   STATE_T — exp_avg / exp_avg_sq state type (always f32 in our wiring;
//             half-precision moments lose representation quickly)
//
// Memory pattern:
//   - Each block walks `n` elements of one tensor chunk.
//   - Per-thread ILP: ILP elements per loop iter, fully unrolled.
//   - Reads / writes are perfectly coalesced (no atomics, no shared mem).
template<typename GRAD_T, typename PARAM_T, typename STATE_T>
struct AdamFunctor
{
  __device__ __forceinline__ void operator()(
      int chunk_size,
      TensorListMetadata<4>& tl,
      const float beta1,
      const float beta2,
      const float beta1_correction,  // = 1 - beta1^t  (or 1.0f if !bias_correction)
      const float beta2_correction,  // = 1 - beta2^t  (or 1.0f if !bias_correction)
      const float epsilon,
      const float lr,
      const adamMode_t mode,
      const float decay)
  {
    const int tensor_loc = tl.block_to_tensor[blockIdx.x];
    const int chunk_loc  = tl.block_to_chunk[blockIdx.x];
    const int n          = tl.sizes[tensor_loc];

    // Slot 0: param, Slot 1: grad, Slot 2: exp_avg, Slot 3: exp_avg_sq.
    PARAM_T* p = reinterpret_cast<PARAM_T*>(tl.addresses[0][tensor_loc]) + chunk_loc * chunk_size;
    GRAD_T*  g = reinterpret_cast<GRAD_T* >(tl.addresses[1][tensor_loc]) + chunk_loc * chunk_size;
    STATE_T* m = reinterpret_cast<STATE_T*>(tl.addresses[2][tensor_loc]) + chunk_loc * chunk_size;
    STATE_T* v = reinterpret_cast<STATE_T*>(tl.addresses[3][tensor_loc]) + chunk_loc * chunk_size;

    // Clamp the chunk to the actual remaining elements (last chunk
    // of a tensor that isn't a perfect multiple of chunk_size).
    const int n_this = (chunk_size < n - chunk_loc * chunk_size)
                     ? chunk_size : (n - chunk_loc * chunk_size);

    // Per-thread ILP loop. Each iter touches `BLOCK_SIZE * ILP` elements.
    for (int i_start = 0; i_start < n_this; i_start += BLOCK_SIZE * ILP) {
      #pragma unroll
      for (int ii = 0; ii < ILP; ++ii) {
        const int i = i_start + threadIdx.x + ii * BLOCK_SIZE;
        if (i < n_this) {
          // Promote everything to f32 for the math — half/bf16 variances
          // lose precision catastrophically.
          float gi = static_cast<float>(g[i]);
          float mi = static_cast<float>(m[i]);
          float vi = static_cast<float>(v[i]);
          float pi = static_cast<float>(p[i]);

          if (mode == adamMode_t::ADAM && decay != 0.0f) {
            // Classic L2 regularization — fold into gradient.
            gi = gi + decay * pi;
          }

          mi = beta1 * mi + (1.0f - beta1) * gi;
          vi = beta2 * vi + (1.0f - beta2) * gi * gi;

          const float mi_hat = mi / beta1_correction;
          const float vi_hat = vi / beta2_correction;
          const float denom  = sqrtf(vi_hat) + epsilon;
          const float update = mi_hat / denom;

          if (mode == adamMode_t::ADAMW && decay != 0.0f) {
            // Decoupled weight decay — applied directly to the weight,
            // independent of the Adam adaptive lr.
            pi = pi - lr * (update + decay * pi);
          } else {
            pi = pi - lr * update;
          }

          // Store back. Half/bf16 parameter store goes through the
          // dtype's narrowing conversion (truncation is fine — Adam
          // is robust to a few bits of rounding noise per step).
          p[i] = static_cast<PARAM_T>(pi);
          m[i] = static_cast<STATE_T>(mi);
          v[i] = static_cast<STATE_T>(vi);
        }
      }
    }
  }
};

} // namespace baracuda_apex
