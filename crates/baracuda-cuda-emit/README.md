# baracuda-cuda-emit

CUDA backend for the neutral `unpopped` kernel generator.

The `Cuda` `Backend` (IR to `.cu` source) and the NVRTC on-demand JIT compiler `NvrtcCompiler` (feature `nvrtc`).
The Fuel kernel-seam `Synthesizer` lives in `baracuda-cuda-seam`.

The crate's MAJOR.MINOR follows the `unpopped` release it builds against; the PATCH is baracuda's own counter.
Targets are open `cuda:smNN` tokens; the generated source is scalar CUDA C. NVRTC 13.x rejects `sm_61`, so Pascal needs a 12.x NVRTC (see `docs/sm61-p40-design.md` in the repository).

Part of [baracuda](https://github.com/ciresnave/baracuda); licensed under MIT OR Apache-2.0 (see `LICENSE-MIT`, `LICENSE-APACHE`).
