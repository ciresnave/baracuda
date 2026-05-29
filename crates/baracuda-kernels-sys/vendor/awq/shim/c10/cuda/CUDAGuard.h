// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda Phase 48 PyTorch-free shim for AWQ (llm-awq).
//
// Empty stand-in for upstream <c10/cuda/CUDAGuard.h>. Same rationale
// as the sibling `torch/extension.h` shim: the AWQ host-side wrapper
// that referenced c10 has been stripped from the vendored
// `src/gemm_cuda_gen.cu`; the device-side kernel does not use c10.

#pragma once
