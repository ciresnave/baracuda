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
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// baracuda Phase 49 — Apex multi_tensor_apply scaffold.
//
// This is the vendored Apex multi_tensor_apply.cuh trimmed to the
// PyTorch-free pieces: the TensorListMetadata pack + the chunked
// dispatch loop. The original upstream file uses `at::Tensor` to
// derive the tensor pointer arrays; baracuda's shim takes those
// arrays directly from the caller (see baracuda_optim_shim.cu).
//
// Apex's chunking strategy:
//   - Each tensor (potentially millions of elements) is sliced into
//     `chunk_size`-element chunks. Each CUDA block processes one chunk
//     of one tensor. The number of blocks launched equals the total
//     number of chunks across all tensors.
//   - One launch handles up to `depth` arrays of pointers — Adam uses
//     depth=4 (param, grad, exp_avg, exp_avg_sq), SGD uses depth=3,
//     LAMB uses depth=5 with two scratch arrays.
//   - The kernel reads its (tensor, chunk) coordinate from a per-block
//     metadata index — no atomic counters, no work-stealing.

#pragma once

#include <cuda_runtime.h>

namespace baracuda_apex {

// Apex's launch geometry. Matches NVIDIA's defaults — these are tuned
// for occupancy on Volta+ and have been stable across Apex releases.
//
// BLOCK_SIZE = 512 was Apex's pick for the Volta era; we keep it for
// source-compatibility with the vendored kernels. On Ampere/Ada/Hopper
// the optimizer step is L1-bandwidth-bound and this block size is at
// or near optimal occupancy.
//
// ILP = 4 controls the inner-loop unroll factor inside each functor.
// Higher ILP improves throughput up to a register-pressure wall;
// Apex settled on 4 for all current functors.
constexpr int BLOCK_SIZE = 512;
constexpr int ILP        = 4;

// Apex's per-launch capacity. A single multi_tensor_apply call can
// dispatch up to `MAX_TENSORS_PER_LAUNCH` tensors and
// `MAX_BLOCKS_PER_LAUNCH` chunk-blocks before the caller must split
// the work into multiple sequential launches.
//
// The caller-side shim (see baracuda_optim_shim.cu) handles the
// splitting transparently: if more tensors are passed than fit in
// one metadata pack, the shim launches multiple back-to-back kernels.
constexpr int MAX_TENSORS_PER_LAUNCH = 110;
constexpr int MAX_BLOCKS_PER_LAUNCH  = 320;

// `depth` is the number of parallel pointer arrays the optimizer
// touches per tensor. Adam: param, grad, exp_avg, exp_avg_sq (4).
// SGD: param, grad, momentum_buffer (3). LAMB: param, grad, exp_avg,
// exp_avg_sq, (intermediate w_norm/g_norm work) (5).
template<int depth>
struct TensorListMetadata
{
  // Per-pointer-slot arrays. Layout:
  //   addresses[d][t] = device pointer to the start of tensor `t`'s
  //                     slot `d` (e.g. addresses[0][t] = param ptr).
  // sizes[t]          = element count of tensor `t`.
  // block_to_tensor[b] = which tensor-index this CUDA block serves.
  // block_to_chunk[b]  = which chunk of that tensor this block serves.
  // start_tensor_this_launch = caller-side bookkeeping for multi-launch
  //                            chunking; the kernel reads this verbatim.
  void*  addresses[depth][MAX_TENSORS_PER_LAUNCH];
  int    sizes[MAX_TENSORS_PER_LAUNCH];
  unsigned char block_to_tensor[MAX_BLOCKS_PER_LAUNCH];
  int    block_to_chunk[MAX_BLOCKS_PER_LAUNCH];
  int    start_tensor_this_launch;
};

// Original Apex multi_tensor_apply<...>() launcher is intentionally
// NOT vendored. It takes a `std::vector<std::vector<at::Tensor>>` and
// builds the metadata pack from PyTorch tensors. baracuda's shim does
// the same job on raw device pointer arrays — see baracuda_optim_shim.cu.

} // namespace baracuda_apex
