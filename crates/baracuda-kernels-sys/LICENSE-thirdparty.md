# Third-party kernel attributions

The `baracuda-kernels-sys` crate vendors a small number of CUDA `.cu`
kernels adapted from upstream open-source projects. Each adapted source
carries an `SPDX-FileCopyrightText:` + `SPDX-License-Identifier:` header
at the top of the file; this document summarizes the upstreams in one
place for license-review tooling.

All vendored kernels are adapted to baracuda's INSTANTIATE-macro
pattern, `extern "C" baracuda_kernels_<op>_<dtype>_run` symbol naming,
and `int32_t` status-code ABI (`0` success, `2` invalid problem,
`5` internal launch failure). The adaptations are non-trivial; the
original kernel body is the load-bearing prior art.

## Vendored sources

### fuel-cuda-kernels (cast / fill / affine)

- **Upstream**: `fuel-cuda-kernels` (the project's own CUDA kernel
  crate), itself adapted from the [huggingface/candle](https://github.com/huggingface/candle)
  project's CUDA kernel set.
- **License**: dual-licensed MIT OR Apache-2.0 (matches baracuda).
- **Adapted files**:
  - `kernels/include/baracuda_cast.cuh` +
    `kernels/elementwise/cast.cu` — adapted from
    `fuel-cuda-kernels/src/cast.cu`.
  - `kernels/include/baracuda_fill.cuh` +
    `kernels/elementwise/fill.cu` — adapted from
    `fuel-cuda-kernels/src/fill.cu`.
  - `kernels/include/baracuda_affine.cuh` +
    `kernels/elementwise/affine.cu` — adapted from
    `fuel-cuda-kernels/src/affine.cu`.

  Adaptation summary:
  * Stripped Fuel's optional strided-view path (the `size_t* info` arg
    + `is_contiguous` / `get_strided_index` helpers) — baracuda's plan
    layer materializes strided views before the launch, so contig-only
    is sufficient at the kernel layer.
  * Replaced Fuel's `extern "C" __global__ void <name>(...)` direct
    kernel-as-FFI surface with the standard baracuda
    `extern "C" int32_t baracuda_kernels_<name>_run(...)` launcher
    pair (`_run` + `_can_implement`) returning status codes.
  * f16 / bf16 affine: compute through f32 to match the precision-
    guarantee contract the rest of the elementwise family follows.
  * f16 / bf16 fill: value transported as a raw `uint16_t` bit
    pattern over the FFI (memcpy'd into the half-precision wrapper
    type before the launch) to avoid Windows-x64 small-struct ABI
    issues with `__half` / `__nv_bfloat16`.
  * Cast: replaced Fuel's `cast_through<TIn, TOut, float>` site-of-use
    detour template with a single `cast_value<TIn, TOut>` entry-point
    function template + per-pair explicit specializations covering
    the half-precision endpoints.

### llama.cpp / ggml-cuda (GGUF block-format dequant + MMVQ)

- **Upstream**: [`ggerganov/llama.cpp`](https://github.com/ggerganov/llama.cpp)
  — specifically the `ggml-cuda.cu` quantized-tensor kernels (block
  formats Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q2_K / Q3_K / Q4_K / Q5_K /
  Q6_K / Q8_K + dequantize + dequantize-mul-mat-vec).
- **License**: MIT (compatible with baracuda's MIT-OR-Apache-2.0).
- **Routing**: vendored via `fuel-cuda-kernels/src/quantized.cu`, which
  itself is a near-verbatim port of the upstream llama.cpp kernels.
- **Adapted files**:
  - `kernels/include/baracuda_gguf.cuh` — block-format struct
    definitions (`block_q4_0` ... `block_q8_K`), per-block dequant
    primitives, dequantize-block templates, and the FP-activation MMVQ
    template (`dequantize_mul_mat_vec`).
  - `kernels/gguf/dequantize.cu` — 11 `extern "C"` host launchers
    (`baracuda_kernels_dequantize_<qtype>_run`) wrapping the dequant
    kernels. f32 output only (f16 output deferred).
  - `kernels/gguf/mmvq.cu` — 10 `extern "C"` host launchers
    (`baracuda_kernels_mmvq_<qtype>_run`) wrapping the FP-activation
    MMVQ kernels for Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q2_K / Q3_K /
    Q4_K / Q5_K / Q6_K. Q8_K is excluded (llama.cpp / Fuel reserve it
    as a CPU-side intermediate; no MMVQ specialization upstream).
    f32 activation, f32 output.

  Adaptation summary:
  - Replaced Fuel's `extern "C" __global__ void <name>(...)` direct
    kernel-as-FFI exports with the standard baracuda
    `extern "C" int32_t baracuda_kernels_<op>_<qtype>_run(...)`
    launcher convention returning status codes (0 success / 2 invalid
    problem / 5 internal launch failure).
  - Pinned `QK_K = 256` — the `GGML_QKK_64` variant is elided since
    every modern GGUF file (llama.cpp >= 2024) uses the 256-element
    super-block layout.
  - Dropped the q8_1-staging tile-based MMQ matmul family
    (`mul_mat_q*`, `mul_mat_vec_q*_q8_1_cuda`, the `load_tiles_q*` /
    `allocate_tiles_q*` template machinery, and the `quantize_q8_1`
    intermediate kernel). baracuda's GGUF surface today ships the
    FP-activation MMVQ path only; the q8_1-staging MMQ path is
    deferred to a follow-up milestone if benchmarks justify the
    additional 2k+ LOC.
  - Dropped Fuel's `indexed_moe_forward_*` (Fuel-specific fused MoE
    op; not on baracuda's roadmap).
  - Q4_0 + Phi-2 dequant fixup: searched Fuel's git log
    (`git log --grep="phi-2|phi2|gguf|quantized"`) and upstream
    llama.cpp PR `25805fff` ("Fix some NaNs with GGML quantized"
    `#3428`). Neither carries a Q4_0-specific bit-pattern fix that
    differs from the verbatim llama.cpp dequant. The Phi-2 inference
    path uses the same `dequantize_q4_0` routine as every other
    model; no special-case patch was applied during vendor.

### attention.rs (Mixture-of-Experts forward)

- **Upstream**: [`guoqingbao/attention.rs`](https://github.com/guoqingbao/attention.rs)
  — specifically `src/kernels/src/moe_gemm_gguf.cu`,
  `moe_gemm_wmma.cu`, and `moe_wmma_gguf.cu` (the three fused
  per-token-dispatch + expert-matmul + accumulate kernels).
- **License**: MIT (compatible with baracuda's MIT-OR-Apache-2.0).
- **Routing**: vendored via `fuel-cuda-kernels/src/moe/`. Phase 20.2
  recon (2026-05-25) confirmed Fuel's source has not changed since the
  Phase 8.5 vendor (one commit on those paths); baracuda's bodies are
  already current. Fuel's `fuel-cuda-kernels/src/moe/` is being retired
  in favour of direct calls to the `baracuda_kernels_moe_*_run` FFI
  symbols.
- **Adapted files**:
  - `kernels/include/baracuda_moe.cuh` — the three kernel templates
    plus the q8_1-staging support family (`block_q8_1`,
    `quantize_q8_1`, `vec_dot_*_q8_1`, `warp_reduce_*`, DP4A wrapper,
    `get_int_from_*` helpers), warp-level single-block dequantize
    functions (`moe_dequantize_block_q[2-6]_K`, `moe_dequantize_block_q8_0`),
    and the expert-offset histogram + Hillis-Steele scan kernels.
  - `kernels/moe/moe_gguf.cu` — scalar GGUF MoE launcher.
    `extern "C" int32_t baracuda_kernels_moe_scalar_gguf_run(...)`.
    Block formats: Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.
  - `kernels/moe/moe_wmma.cu` — FP-weights WMMA MoE launchers.
    `baracuda_kernels_moe_wmma_<f16|bf16>_run`.
  - `kernels/moe/moe_wmma_gguf.cu` — combined WMMA + GGUF MoE
    launchers (hot path for quantized LLM inference).
    `baracuda_kernels_moe_wmma_gguf_<f16|bf16>_run`. Same block-format
    coverage as `moe_gguf.cu`.

  Adaptation summary:
  - Replaced Fuel's `extern "C" void moe_gemm_*` direct FFI exports
    with baracuda's status-returning `extern "C" int32_t
    baracuda_kernels_moe_<variant>_<dtype>_run(...)` launcher
    convention (0 success / 2 invalid problem / 5 internal launch
    failure).
  - GGUF block layouts (`block_q8_0` / `block_qN_K`) are pulled from
    baracuda's `baracuda_gguf.cuh` (the Milestone 8.4 surface), NOT
    Fuel's `moe/gguf.cuh`. The struct ABI is identical (both vendor
    llama.cpp), only the namespace differs.
  - The q8_1-staging family (`block_q8_1`, `vec_dot_*_q8_1`,
    `quantize_q8_1`) was intentionally excluded from baracuda's
    Milestone 8.4 surface (the dequant + FP-activation-MMVQ-only
    header). It's vendored into `baracuda_moe.cuh` directly to keep
    the 8.4 surface clean while still allowing the scalar GGUF MoE
    path to use it.
  - Warp-level single-block dequantize functions (`moe_dequantize_block_q*_K`,
    `moe_dequantize_block_q8_0`) are distinct from the dequant
    launcher's `dequantize_block_q*_K_tmpl` variants in
    `baracuda_gguf.cuh`: same compute, different call-site contract
    (one block at the passed pointer vs `blockIdx.x` selecting from
    `nb32` blocks).
  - Expert-offset computation uses a Thrust-free path:
    `moe_count_tokens_per_expert_kernel` (atomicAdd histogram) +
    `moe_expert_prefix_sum_kernel` (single-block Hillis-Steele
    inclusive scan, requires `num_experts <= 1024`). The Fuel `_light`
    variant served as the template; the Thrust-based `calculate_expert_offsets`
    is not vendored.
  - Block-format coverage matches Fuel exactly: Q8_0 + k-quants
    (Q2_K..Q6_K). Q4_0 / Q4_1 / Q5_0 / Q5_1 would require adding
    four `vec_dot_q*_q8_1` entries Fuel itself doesn't wire for the
    MoE path.
