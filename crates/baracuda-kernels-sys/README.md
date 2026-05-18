# baracuda-kernels-sys

Raw `extern "C"` entry points for the bespoke `.cu` kernels behind
[`baracuda-kernels`]. **You almost certainly want `baracuda-kernels`
instead** — it wraps these unsafe calls with typed plans, lifetime-
checked device buffers, the workspace contract, and a proper Rust API.

This crate exists at the layer where the workspace's other `*-sys`
crates do: just the C ABI bindings + a `build.rs` that compiles the
`.cu` sources via [`baracuda-forge`].

## What's compiled here

`kernels/` ships hand-rolled CUDA C++ for every op family that doesn't
fit (or doesn't fit well) inside the NVIDIA library SKUs:

```text
kernels/
  attention/      RoPE, ALiBi, SDPA, Flash SDPA (sm_80 + sm_89), KV-cache append
  elementwise/    unary / binary / ternary math + activations + casts + affine + fill
  embedding/      embedding + embedding_bag (FW + BW)
  fft/            fftshift / ifftshift index-permutation helpers
  gemm/           int8 RRR, FP8 (sm_89), int4, bin GEMM
  gguf/           llama.cpp GGUF block-format dequant + MMVQ (Q4_0..Q8_K + k-quants)
  image/          interpolate, grid_sample, pixel_shuffle, ROI ops, NMS
  indexing/       gather, scatter_add, index_select, masked_fill, one_hot, nonzero
  linalg/         ormqr WY-blocked + reflector-by-reflector, batched QR helpers
  loss/           CTCLoss DP forward + backward
  moe/            scalar GGUF MoE + WMMA MoE + combined WMMA+GGUF MoE
  norm/           RMSNorm, LayerNorm, BatchNorm, GroupNorm, InstanceNorm (FW + BW)
  quantize/       per-tensor / per-channel / per-token / per-group quantize / dequantize
  random/         dropout + bernoulli on top of cuRAND-uniform
  segment/        sorted + unsorted segment sum / mean / max / min / prod
  softmax/        softmax, log-softmax, Gumbel-softmax, sparsemax (FW + BW)
  sort/           block-bitonic sort + topk + kthvalue + msort + searchsorted + bincount + histogram + unique
  include/        shared headers — mma.sync wrappers, cp.async, smem swizzle, warp reductions, epilogue helpers
```

Every kernel uses the same calling convention: `extern "C" int32_t
baracuda_kernels_<family>_<spec>_run(...)` returning a status code (see
below), paired with `_can_implement` and `_workspace_size` helpers
where applicable. The safe layer in `baracuda-kernels` wraps each
launcher behind a `Plan`.

## Build script — CUDA + cuDNN auto-discovery

The build script (`build.rs`) drives `baracuda-forge::KernelBuilder` to
compile the selected kernel set with `nvcc`. The CUDA toolkit is
auto-discovered through baracuda-forge's standard probe (`CUDA_PATH` /
`CUDA_HOME` / Windows installer layouts / Linux distro paths).

When the `cudnn` feature is enabled the build script also probes for
cuDNN. Search order, first hit wins:

1. `CUDNN_PATH`, `CUDNN_ROOT`, `CUDNN_HOME` — explicit env-var roots.
2. `C:\Program Files\NVIDIA\CUDNN\v<X.Y>\lib\<cuda_ver>\x64\` — the
   standard Windows installer layout for cuDNN 9+ (versioned by both
   cuDNN release and target CUDA toolkit).
3. Legacy "drop into CUDA toolkit" layout — `$CUDA_PATH\lib\x64\` on
   Windows or `$CUDA_PATH/lib64/` on Linux. The pre-cuDNN-9 tarball
   convention.
4. Linux distro paths — `/usr/lib/x86_64-linux-gnu/` and
   `/usr/local/cuda/lib64/`.

If cuDNN can't be found and the `cudnn` feature is enabled, the build
fails with a typed error pointing at the search paths it tried.

## Cargo features

| Feature | Default | Effect |
| --- | --- | --- |
| `sm80` | yes | Ampere baseline — runs forward-compatibly on Ada and Hopper. |
| `sm89` | no | Ada Lovelace specializations (adds FP8 kernels, sm_89 Flash Attention). |
| `sm90a` | no | Hopper-specialized kernels (stubs today). |
| `cudnn` | no | Compile + link the cuDNN-backed launchers (conv / pool / CTC-cuDNN). |

## Status codes

All `*_run` and `*_can_implement` entry points return an `int32_t`:

| Code | Meaning |
| ---: | --- |
| `0` | success |
| `1` | misaligned operand |
| `2` | invalid problem (M, N, or K non-positive, or shape inconsistency) |
| `3` | not supported (this kernel doesn't implement the requested shape) |
| `4` | workspace too small or null when required |
| `5` | internal kernel error (typically a launch failure) |

The safe layer maps these to `baracuda_cutlass::Error` variants (the
shared error type for the kernel facade). See
[`ARCHITECTURE.md`](../../ARCHITECTURE.md#error-handling) for the
mapping.

## Safety

Functions take raw `void*` pointers, integer dimensions, and a
`cudaStream_t` cast to `*mut c_void`. They are unsafe because:

- Pointers are dereferenced without bounds checking.
- They're assumed to be valid device addresses.
- When a workspace pointer is non-null, it's assumed to point at the
  number of writable device bytes the caller asked `*_workspace_size`
  for.
- The stream is assumed to be a live CUDA stream owned by the calling
  thread's current context.

If any of those are violated, you'll get either `i32 == 5` (the launch
failed for a CUDA-reported reason) or undefined behavior at the GPU
level. The safe layer in `baracuda-kernels` enforces all of these.

## Vendored kernels

A subset of kernels in this crate are vendored from upstream open-
source projects (HuggingFace candle via fuel-cuda-kernels for the
elementwise core; llama.cpp's `ggml-cuda` for GGUF dequant + MMVQ;
`guoqingbao/attention.rs` for fused MoE). Each adapted source carries
an `SPDX-FileCopyrightText:` + `SPDX-License-Identifier:` header; the
consolidated provenance is in
[`LICENSE-thirdparty.md`](LICENSE-thirdparty.md).

[`baracuda-kernels`]: https://docs.rs/baracuda-kernels
[`baracuda-forge`]: https://docs.rs/baracuda-forge
