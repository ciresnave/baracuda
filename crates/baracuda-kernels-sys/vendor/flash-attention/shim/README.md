# PyTorch-dependency shim headers (Phase 42)

Vendored FA2 sources `#include <ATen/...>` and `#include <c10/...>`,
which would normally come from PyTorch's headers. baracuda must
build PyTorch-free, so these shims satisfy the `#include` paths
with minimal stand-ins.

## Headers

- `ATen/cuda/CUDAGeneratorImpl.h` — defines `at::PhiloxCudaState`
  struct. FA2 stores one inside `Flash_fwd_params` and only reads
  it when dropout is enabled. We leave the body trivially copyable
  with `seed` and `offset` fields so FA2's `params.philox_args =
  ...` assignment compiles. Phase 42 ships dropout-disabled, so the
  struct's values are never consumed.
- `ATen/cuda/detail/UnpackRaw.cuh` — provides `at::cuda::philox::unpack`
  returning `{0, 0}`. Called from the device side of
  `flash_fwd_kernel.h` only when `Is_dropout == true`. The
  `kernels/attention/fa2_launcher.cu` launcher always sets
  `p_dropout = 1.0f` (probability of keeping = 1.0 = no dropout)
  so the dropout switch resolves to the dead `Is_dropout=true`
  branch at compile time when `BOOL_SWITCH(p_dropout < 1.f, …)` =
  false, and `unpack` is never reached at runtime.
- `c10/cuda/CUDAException.h` — provides `C10_CUDA_CHECK(cmd)` and
  `C10_CUDA_KERNEL_LAUNCH_CHECK()` macros. We translate them to a
  basic `fprintf(stderr, ...); abort();` pattern — matches the
  upstream macro's semantics (abort on CUDA error) but without the
  c10 exception-throwing.

## Search-path priority

These shims are exposed at the **front** of nvcc's `-I` list. The
real ATen / c10 headers (if a PyTorch install is on the system) are
shadowed.
