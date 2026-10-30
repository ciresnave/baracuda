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
  sm_61 MMVQ via `__dp4a`: portable in principle, BLOCKED by board 307 (the Nvidia quantized-dot-product
  patent; MMVQ is one of the two kernels it names). PM ruling 2026-10-30: no build-matrix change for MMVQ
  while 307 is unruled.
- Unified memory / peer access over the PEX switch: out of scope here.

## NVRTC requirement

CUDA 13.x NVRTC rejects `sm_61` (`NVRTC_ERROR_INVALID_OPTION`); 12.9.x compiles it. A 12.9 NVRTC must be
reachable through `BARACUDA_NVRTC_PATH` (see the toolkit-selection doc). `nvrtc-builtins64_129.dll` must
sit beside it; the loader prepends that directory to `PATH`. CUDA 12.9 is the last toolkit with Pascal.
The driver on the P40 box must support the PTX/cubin it is given (a cubin for sm_61 needs no JIT).

## Toolchain: the P40 cannot share a CUDA 13 build with the 4060

Measured by the PM on this box, 2026-10-30: CUDA 13.3 `nvcc --list-gpu-arch` starts at `compute_75`, and
`nvcc -arch=sm_61` fails with "Unsupported gpu architecture 'sm_61'". Measured here: CUDA 13.3 NVRTC
rejects `--gpu-architecture=sm_61` (`NVRTC_ERROR_INVALID_OPTION`) and the 12.9.86 NVRTC redist compiles it.
So the sm_61 path needs a PINNED 12.9 toolchain, separate from the CUDA 13 one that builds for the
RTX 4060 (sm_89).

How the build selects per device (decision for the PM; the recommendation is the first):

1. Runtime JIT, no second build (recommended, accepted by the PM 2026-10-30 for the elementwise/reduction paths). The emitter already produces source per `TargetId`; at
   run time `NvrtcCompiler` picks the NVRTC by target (`BARACUDA_NVRTC_PATH` for sm_61, the default for
   sm_75+) and loads the resulting PTX/cubin per device. The hand-written AOT library stays a CUDA 13
   build for sm_80+ and is simply not offered on the P40 (typed decline). Cost: a pinned 12.9 NVRTC
   (315 MB redist; NVIDIA licence, never copied into a repo or artifact) shipped or located on the
   P40 host.
2. Two AOT builds of the same sources, one per toolchain (CUDA 12.9 for sm_61, 13.x for the rest), chosen
   at install/link time. Only worth it if a hand-written kernel (the MMVQ dp4a file) is built for sm_61;
   doubles build time (a fresh nvcc build of `baracuda-kernels-sys` is ~1h45m on this box).
3. One fat binary with SASS/PTX per arch from two toolchains: not possible from a single nvcc
   invocation, since 13.x cannot emit sm_61 and 12.9 cannot emit what 13.x targets later; it would be
   two builds merged by hand. Not recommended.

What the driver does with old SASS: UNVERIFIED until the P40 is in the box. Expectations to test, not
facts: a cubin is matched on exact SM major plus minor at or below the device; PTX is JIT-compiled by
the driver for any newer device; a CUDA 13-era driver (needed by the 4060) may have dropped Pascal
(NVIDIA's CUDA 13 notes drop Maxwell/Pascal/Volta offline compilation, and support for the GPUs in the
driver branch is a separate question). Whether one driver install serves both the P40 and the 4060 is
therefore an open hardware question, and it decides whether both cards can share a process at all.

## Hardware-free part (doable now, no P40)

1. DONE 2026-10-30 (`crates/baracuda-cuda-emit/tests/sm61_compiles.rs`, NVRTC 12.9.86 redist via
   `BARACUDA_NVRTC_PATH`): 14 elementwise bodies (add, mul, max, relu, silu, gelu, sigmoid, tanh, exp,
   sqrt, sin, pow, floor, select) and 3 reductions (sum, max, min), each for f32, f64, f16, bf16, emitted
   for `cuda:sm61`, `sm70`, `sm75` and `sm89` (control) and compiled to PTX. Result, identical on all
   four targets: 0 compile failures; f32/f64 14+3 compiled; f16 and bf16 13+3 compiled and 1 typed
   decline each (elementwise). PTX generation only: NOT run on device. The f16/bf16 headers came from the
   CUDA 13.3 include dir (the redist ships none), so a 12.9 header set is still unchecked. Without a 12.x
   NVRTC the test declares a skip (13.x rejects sm_61).
2. Catch Pascal-incompatible PTX the grep-based claim in the gap analysis cannot: `__activemask`-class
   warp primitives, 64-bit float atomics, `cp.async`. A successful sm_61 compile is the test.
3. Check `unpopped::generate()` does not gate unknown `cuda:smNN` tokens before reaching `cuda.rs`.

## Hardware part (needs the P40)

Run the same kernels on device and compare to the CPU reference; then decide on the MMVQ build question
above. Evidence must be labelled "device-verified sm61" only after a P40 run.

## Open questions

- sm_61 MMVQ: blocked by board 307 (above); revisit on the ruling.
- Whether one NVIDIA driver serves both the P40 and the 4060 (see Toolchain).
- Whether the Fuel dispatcher selects the emitter path automatically on a sub-sm_80 device.
