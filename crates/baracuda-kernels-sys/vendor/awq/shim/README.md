# PyTorch-dependency shim headers (Phase 48)

The upstream AWQ kernel TU (`src/gemm_cuda_gen.cu`) `#include`s
`<torch/extension.h>` and `<c10/cuda/CUDAGuard.h>` for its host
entry point's wrapper logic. baracuda strips both includes and the
entire host-side `gemm_forward_cuda(...)` wrapper from the vendored
copy — we use our own baracuda launcher at
`kernels/quantization/awq_launcher.cu` instead.

These shim headers exist as **belt-and-suspenders insurance** in case
a future re-vendor reintroduces the includes before the strip-patch is
re-applied. They satisfy the include paths with empty / minimal
stand-ins.

## Headers

- `torch/extension.h` — empty header. Upstream pulls this in for
  its `torch::Tensor` type and `pybind11`; the kernel itself doesn't
  use either, so the empty header is sufficient.
- `c10/cuda/CUDAGuard.h` — empty header. The upstream uses
  `at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));`
  in the stripped wrapper; the kernel itself doesn't reference c10
  at all.

## Search-path priority

When the `awq` cargo feature is on, these shims sit at the **front**
of nvcc's `-I` list. Any real PyTorch install on the system is shadowed.

## Why these are empty

Compare to the Phase 42 FA2 shim, where `c10/cuda/CUDAException.h`
provides real `C10_CUDA_CHECK` macro behaviour. AWQ does not invoke
any c10 macro from its device-side code; only the wrapper that we
deleted references c10 / torch. Empty shims are correct and complete.
