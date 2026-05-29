# Vendored: mit-han-lab/llm-awq (Phase 48)

This directory contains a curated subset of the AWQ (Activation-aware
Weight Quantization) CUDA source tree, vendored into baracuda as part
of Phase 48 to natively support the **most-deployed 4-bit format on
the Hugging Face Hub** (Llama / Mistral / Qwen prebuilts published
as `*-AWQ`). Goal B of Phase 48.

## Provenance

- **Upstream**: <https://github.com/mit-han-lab/llm-awq>
- **Pinned commit**: `d6e797a42b9ef7778de8ee2352116e0f48a78d61`
  ("[Major] set VILA deps as optional", 2025-07-17)
- **License**: MIT (see `LICENSE` next to this file).
- **Vendored**: 2026-05-28.

## License attribution — MIT (no patent grant)

The verbatim upstream `LICENSE` file is checked in alongside this
README. **Do not modify it.** AWQ is licensed under the **MIT
License**, which **does NOT include an explicit patent grant**. This
is a material distinction from the sibling Marlin vendor
(`vendor/marlin/`, Apache-2.0): Marlin grants patent rights with
termination-on-litigation; MIT is silent on patents.

baracuda's own license is dual MIT / Apache-2.0 — both arms are
compatible with the vendored AWQ MIT sources.

Per-file copyright headers in `src/gemm_cuda_gen.cu` and
`src/dequantize.cuh` are preserved verbatim from upstream. The
top-level `README.md` of baracuda's workspace lists AWQ under its
third-party attribution section.

The dequantize routine in `src/dequantize.cuh` is itself derived from
NVIDIA's FasterTransformer (`cutlass_extensions/interleaved_numeric_conversion.h`,
per the file's own header comment). FasterTransformer is Apache-2.0;
AWQ's adapter is licensed MIT.

The original `AUTHORS` file did not exist in the upstream tree —
`AUTHORS` here captures the author attribution that the upstream
README + LICENSE convey.

## Paper citation

```bibtex
@inproceedings{lin2024awq,
  title={{AWQ}: Activation-aware Weight Quantization for On-Device LLM
         Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and
          Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and
          Dang, Xingyu and Gan, Chuang and Han, Song},
  booktitle={Proceedings of Machine Learning and Systems},
  year={2024}
}
```

## Scope: what we kept

`src/` contains only the W4A16 GEMM kernel TU and its dequant header:

- `gemm_cuda_gen.cu` — the AWQ W4A16 GEMM kernel
  (`gemm_forward_4bit_cuda_m128n64k32`, single templated kernel
  parameterised on `G ∈ {64, 128}`). ~298 LOC upstream; we ship the
  device-side kernel as-is and replace the PyTorch wrapper with a
  baracuda C-ABI launcher at `kernels/quantization/awq_launcher.cu`.
- `dequantize.cuh` — the AWQ int4 → fp16 dequant routine. ~79 LOC.

## Scope: what we removed

- **`gemv_cuda.cu` / `gemv_cuda.h`** — AWQ also ships a separate GEMV
  kernel for batch=1 decode; we deferred it (the GEMM kernel handles
  batch=1 acceptably for Phase 48; GEMV path can be added in a
  follow-up if profiling shows it's worth it).
- **AWQ's other CUDA sources** — `attention/`, `layernorm/`,
  `position_embedding/`, `rope_new/`, `w8a8/`, `quantization_new/`.
  baracuda already ships all of these primitives natively under
  `crates/baracuda-kernels/src/`.
- **Python bindings** (`awq/`, `setup.py`, `entry.py`, …) —
  out of scope for a Rust crate.
- **AWQ calibration / search pipeline** — pre-quantization is
  Python-only and runs once per model; calibration is out of scope
  for the kernel layer.

## PyTorch dependency shim

Upstream `gemm_cuda_gen.cu` `#include`s `<torch/extension.h>` and
`<c10/cuda/CUDAGuard.h>` for its host entry point's wrapper logic
(allocating the split-k output, calling `.sum(0)`). The device-side
kernel itself is PyTorch-free.

baracuda strips this. The vendored `src/gemm_cuda_gen.cu` is patched
to remove both PyTorch includes and the entire host-side wrapper
`gemm_forward_cuda(...)`; we keep only the `__global__` kernel
`gemm_forward_4bit_cuda_m128n64k32<G>` and the `__pack_half2` /
`make_divisible` helpers it needs. The baracuda launcher at
`kernels/quantization/awq_launcher.cu` provides its own host-side
wrapper that:

1. Allocates the split-k staging buffer from the caller's workspace.
2. Launches the kernel.
3. Runs a row-wise reduce-sum (split_k_iters → 1) into the final
   output tensor.

A minimal shim header at `shim/torch/extension.h` and
`shim/c10/cuda/CUDAGuard.h` exists for completeness (in case a
future re-vendor reintroduces unmoved includes), but the
patched `gemm_cuda_gen.cu` does not actually need them at compile
time. The shims are documented at `shim/README.md`.

This pattern mirrors Phase 42's FA2 shim approach.

## Kernel contract

AWQ is **asymmetric int4** with explicit per-group zero-points.

- **Activations** : `[M, K]` row-major `half` (fp16). M can be any
  value; AWQ's kernel is optimised for the M < 16 decode regime but
  works for any M.
- **Weights** : `[OC, IC/8]` int32 = `[OC, IC]` packed int4
  (8 nibbles per int32 word, interleaved per AWQ's
  `dequantize_s4_to_fp16x2` lop3 pattern). **Note the upstream
  packing layout is OC-major, IC-minor**, which is the transpose
  of the `[K, N]` you might naively expect.
- **Scales** : `[IC/group_size, OC]` half.
- **Zeros** : `[IC/group_size, OC/8]` int32 = `[IC/group_size, OC]`
  packed int4 (the zero-points are themselves int4).
- **Output** : `[M, OC]` row-major half.
- **Group size** : 64 or 128 (the kernel templates on `G`).
- **OC alignment** : OC must be divisible by 64.
- **IC alignment** : IC must be divisible by `group_size`.

AWQ uses a `split_k_iters` split-k strategy: the kernel emits an
intermediate `[split_k_iters, M, OC]` tensor; the launcher does a
final reduce-sum along the split-k axis.

The kernel is **fp16-only** in upstream — there is no bf16 path.
Phase 48 documents this limitation and does NOT attempt to extend
to bf16 (the magic-number dequant trick in `dequantize.cuh` is
fp16-specific).

## Build integration

The vendored kernel is compiled when the `awq` cargo feature on
`baracuda-kernels-sys` is enabled. The build script
(`crates/baracuda-kernels-sys/build.rs`) adds the include paths and
compiles both `vendor/awq/src/gemm_cuda_gen.cu` and the baracuda
launcher `kernels/quantization/awq_launcher.cu`.

## Future scope

- **AWQ GEMV path** — `gemv_cuda.cu` for the M=1 decode hot path.
- **bf16 path** — would require a new dequant routine (the upstream
  `dequantize_s4_to_fp16x2` is fp16-specific). Deferred — caller
  can cast bf16 activations to fp16 first, at small accuracy cost.
- **AWQ v2 / GEMM-new** — `csrc/quantization_new/` ships a newer
  Marlin-inspired GEMM kernel. Deferred; the Phase 48 Marlin
  vendor (Goal A) already provides the Marlin path.

## Pruning script

To re-vendor from upstream:

```bash
git clone https://github.com/mit-han-lab/llm-awq
cd llm-awq
git checkout d6e797a42b9ef7778de8ee2352116e0f48a78d61
cp awq/kernels/csrc/quantization/gemm_cuda_gen.cu \
   <baracuda>/vendor/awq/src/
cp awq/kernels/csrc/quantization/dequantize.cuh \
   <baracuda>/vendor/awq/src/
cp LICENSE <baracuda>/vendor/awq/
# Then re-apply the patch that strips <torch/extension.h>,
# <c10/cuda/CUDAGuard.h>, and the `gemm_forward_cuda(...)` host
# wrapper from `src/gemm_cuda_gen.cu` (the device __global__ kernel
# and `__pack_half2` / `make_divisible` helpers are kept).
```
