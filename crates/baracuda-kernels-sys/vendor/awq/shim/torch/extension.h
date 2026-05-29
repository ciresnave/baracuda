// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda Phase 48 PyTorch-free shim for AWQ (llm-awq).
//
// Empty stand-in for upstream <torch/extension.h>. The vendored AWQ
// `src/gemm_cuda_gen.cu` is patched to remove the include and the
// host-side wrapper that uses torch::Tensor — only the __global__
// kernel and its lop3-dequant helpers are kept. The kernel itself
// does not reference torch / pybind11 at all.
//
// This empty header exists as defence-in-depth in case a future
// re-vendor reintroduces the include before the strip patch is
// re-applied.

#pragma once
