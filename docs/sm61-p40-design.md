# SA-13: sm_61 (Tesla P40) path, design note

Status: design, written 2026-10-30 at `origin/main` 7823c6c3. Nothing here has run on a P40.
Trigger: CireSnave 2026-10-30 (verbatim): "My desktop PC has an RTX 4060, and a PEX switch hosting 4
external PCIe slots which currently have my Radeon VII, an Intel Arc A770, and one P40, and I have both
another P40 and an MI50 (32GB) ordered." A P40 (24 GB, sm_61) is the target for the first model on that
rig (Qwen3.8-27B Q4_K_M fits one card). Background: `docs/sm61-nvrtc-toolkit-selection.md`,
`docs/sm61-parse-emit-gap-analysis.md`.

## Which kernels have an sm_61 path

| Layer | sm_61 today | Why |
|---|---|---|
| Hand-written `baracuda-kernels-sys` (473 files under `kernels/`) | none | `build.rs` targets sm_80+; at least 18 files use `mma.sync` / `cp.async` / `ldmatrix` (sm_80/sm_89 tensor-core and async-copy). |
| `baracuda-cuda-emit` (NVRTC, scalar CUDA C) | yes, compile-only, unrun | `NvrtcCompiler::new(TargetId)` accepts `cuda:sm61` -> `sm_61` (#151); half arithmetic goes through f32 on a `ViaF32` plan (`bodies_pack`, alpha.20). |
| GGUF MMVQ (`gguf/mmvq_multim.cu`, `baracuda_mmvq_multim.cuh`) | portable in principle, not built for sm_61 | uses `__dp4a` (sm_61+) through `q8_1_dp4a`; the header comment says "baracuda targets sm_80+". The file may still pull in sm_80-only includes: unverified. |

Everything the emitter generates for an sm_61 target is therefore the primary route; the hand-written
library is not an sm_61 library and this note does not propose making it one.

## What falls back

- Tensor-core GEMM / attention / conv: no sm_61 equivalent. A caller on a P40 gets a typed decline
  (arch below the kernel's minimum), never a silent wrong-arch launch.
- Half / bf16 math: Pascal GP102 has no fast fp16 (1/64 rate) and no bf16. Compute in f32 (`ViaF32`);
  store fp16 where the format needs it. Do not choose fp16 arithmetic for speed on this card.
- Quantised decode (Q4_K_M etc.): `__dp4a` is the one useful int8 primitive on sm_61. The existing MMVQ
  dot is the candidate; making it build for sm_61 is a build-matrix change, not a new kernel.
  Board 307 holds NEW int8/int4 at-bitwidth dot kernels; building the existing dp4a file for another arch
  is not that, but ask the PM before starting.
- Unified memory / peer access over the PEX switch: out of scope here.

## NVRTC requirement

CUDA 13.x NVRTC rejects `sm_61` (`NVRTC_ERROR_INVALID_OPTION`); 12.9.x compiles it. A 12.9 NVRTC must be
reachable through `BARACUDA_NVRTC_PATH` (see the toolkit-selection doc). `nvrtc-builtins64_129.dll` must
sit beside it; the loader prepends that directory to `PATH`. CUDA 12.9 is the last toolkit with Pascal.
The driver on the P40 box must support the PTX/cubin it is given (a cubin for sm_61 needs no JIT).

## Hardware-free part (doable now, no P40)

1. Compile a representative sample of emitted kernels (elementwise, reduce, matmul-scalar, softmax,
   cast, a half-typed one) with NVRTC 12.9 for `cuda:sm61` and assert success; also target sm_70 / sm_75.
   Run on any box with a 12.9 NVRTC; skip with a labelled reason where it is absent (`require!`).
   BLOCKED on this box: only CUDA 13.3 is installed (an `[ASK]` for a 12.9 NVRTC is with the PM).
2. Catch Pascal-incompatible PTX the grep-based claim in the gap analysis cannot: `__activemask`-class
   warp primitives, 64-bit float atomics, `cp.async`. A successful sm_61 compile is the test.
3. Check `unpopped::generate()` does not gate unknown `cuda:smNN` tokens before reaching `cuda.rs`.

## Hardware part (needs the P40)

Run the same kernels on device and compare to the CPU reference; then decide on the MMVQ build question
above. Evidence must be labelled "device-verified sm61" only after a P40 run.

## Open questions

- Which sm_61 build of the MMVQ file, if any, belongs in `baracuda-kernels-sys` (needs the PM / 307 view).
- Whether the Fuel dispatcher selects the emitter path automatically on a sub-sm_80 device.
