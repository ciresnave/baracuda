# Vendored: AndreSlavescu/mHC.cu (Phase 43)

This directory contains a curated subset of the mHC.cu CUDA source
tree, vendored into baracuda as part of Phase 43 to give baracuda a
native implementation of DeepSeek-AI's Manifold-Constrained
Hyper-Connections residual-stream mixing op.

mHC replaces the bare `y = x + sublayer(x)` residual connection in
transformer blocks with a learned `n`-stream mixing scheme that flows
information through a small (`n × n`) hyper-connection matrix `M` that
is itself constrained to be doubly-stochastic (via Sinkhorn-Knopp
iterations on the manifold of doubly-stochastic matrices). The paper
reports that this small change improves training stability and
downstream task scores across multiple base architectures.

## Provenance

- **Upstream**: <https://github.com/AndreSlavescu/mHC.cu>
- **Pinned commit**: `a426939c2dbc11c443db041bcff12b65d1b6482a` ("fix build")
- **License**: MIT (see `LICENSE` next to this file).
- **Vendored**: 2026-05-28.

## License attribution

The verbatim upstream `LICENSE` file is checked in alongside this
README. **Do not modify it.** The original `AUTHORS` file did not
exist in the upstream tree — `AUTHORS` here captures the author
attribution that the upstream README + LICENSE conveys.

Per-file copyright headers in every `src/*.cuh` / `include/*.{h,cuh}`
file are preserved verbatim from upstream. baracuda's own license
(dual MIT / Apache-2.0) sits alongside; the vendored mHC sources
retain MIT independently.

The top-level `README.md` of baracuda's workspace lists mHC.cu under
its third-party attribution section.

## Paper citation

```bibtex
@article{xie2025mhc,
  title={mHC: Manifold-Constrained Hyper-Connections},
  author={Xie, Zhenda and Wei, Yixuan and Cao, Huanqi and Zhao, Chenggang and
          Deng, Chengqi and Li, Jiashi and Dai, Damai and Gao, Huazuo and
          Chang, Jiang and Zhao, Liang and Zhou, Shangyan and Xu, Zhean and
          Zhang, Zhengyan and Zeng, Wangding and Hu, Shengding and
          Wang, Yuqing and Yuan, Jingyang and Wang, Lean and Liang, Wenfeng},
  journal={arXiv preprint arXiv:2512.24880},
  year={2025}
}
```

## Scope: what we kept

`src/` contains the kernel header tree (all `.cuh` files are
header-only template kernels) needed for the **Tier-1 integration**
(static-H path, bf16 only):

- `mhc_layer.cuh` — top-level `MHCLayer` struct + static-H + dynamic-H
  forward dispatch + backward. The Tier-1 launcher only invokes the
  static-H FW path.
- `stream_ops.cuh` — fused stream aggregate / distribute / mix kernels
  (the per-stream residual mixing where `M` lives).
- `rmsnorm.cuh` — fused RMSNorm with rms-save (reused by mHC).
- `sinkhorn_knopp.cuh` — Sinkhorn-Knopp doubly-stochastic projection
  iterations (the manifold-constraint enforcement).
- `fused_rmsnorm_matmul.cuh` — fused RMS + cuBLAS-Lt matmul (only
  exercised on the dynamic-H code path; included for completeness so
  the `mhc_layer.cuh` template instantiations type-check).

`include/` contains shared helpers:

- `mhc_types.h` — `floatX = nv_bfloat16` typedef + cuBLAS error checks.
- `utils.cuh` — fused-H activation kernels, FP→BF16 convert, L2-flush
  helper, sigmoid/exp intrinsic wrappers, the device profiler harness.
- `profiling.cuh` — CUDA event-based profiler timestamps.

## Scope: what we removed

- **Python bindings** (`src/python/`) — we expose mHC through
  `baracuda_kernels_mhc_*` FFI symbols, not Python.
- **Modal deploy harness** (`runmodal.py`, `setup.py`, `pyproject.toml`,
  `Dockerfile`, `Makefile`) — out of scope for a Rust crate.
- **PyTorch trainer** (`src/python/mhc/trainer.py`, `model.py`,
  `layer.py`, `ops.py`) — we provide our own Rust-side wrapper.
- **C++ tests** (`src/csrc/tests/`) — baracuda ships its own GPU
  smoke tests under `crates/baracuda-kernels/tests/`.
- **C++ benchmark scaffolding** (`src/csrc/benchmarks/`,
  `src/csrc/include/bench_harness.cuh`) — baracuda has its own
  criterion-based bench under `crates/baracuda-kernels-bench/`.
- **CMakeLists.txt** — built through `baracuda-forge` instead.

## Future scope (Tier-2+)

- **Backward**: extend the launcher to call `MHCLayer::backward` and
  surface `baracuda_kernels_mhc_layer_static_bf16_backward`. The
  upstream `MHCLayer::init(.., enable_backward = true)` already
  allocates the gradient buffers.
- **Dynamic-H FW + BW**: vendor the dynamic-H path. Requires cuBLAS-Lt
  (which baracuda already links for cuBLAS).
- **f16 / f32 paths**: the upstream is hardcoded to `floatX = bf16`.
  Adding fp16 / fp32 needs either an additional templated launcher
  layer or in-launcher cast kernels.

## Build integration

The vendored sources are compiled when the `mhc` cargo feature on
`baracuda-kernels-sys` is enabled. The build script
(`crates/baracuda-kernels-sys/build.rs`) adds the include paths and
compiles `kernels/attention/mhc_launcher.cu` which is the baracuda
C-ABI shim wrapping the upstream `MHCLayer` API. mHC requires
**cuBLAS-Lt** which is part of cuBLAS (already linked).

## Pruning script

To re-vendor from upstream:

```bash
git clone https://github.com/AndreSlavescu/mHC.cu
cd mHC.cu
git checkout a426939c2dbc11c443db041bcff12b65d1b6482a
cp src/csrc/include/{mhc_types.h,utils.cuh,profiling.cuh}    <baracuda>/vendor/mhc/include/
cp src/csrc/kernels/{mhc_layer,fused_rmsnorm_matmul,rmsnorm,sinkhorn_knopp,stream_ops}.cuh \
    <baracuda>/vendor/mhc/src/
cp LICENSE <baracuda>/vendor/mhc/
```
