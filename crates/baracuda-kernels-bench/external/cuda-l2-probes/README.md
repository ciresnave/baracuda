# CUDA-L2 standalone probes

Standalone `.cu` validation probes used to decide whether to vendor
[`deepreinforce-ai/CUDA-L2`](https://github.com/deepreinforce-ai/CUDA-L2)
HGEMM kernels as a third `GemmPlan` backend (alongside `Bespoke` /
`Cublas`).

**Decision: SKIP.** See `BENCHMARKS.md` Phase 44 section and the
`gemm_vs_cuda_l2` bench file for the head-to-head numbers + rationale.

## Provenance

CUDA-L2 upstream pinned at commit `dbe017722194bb33bafadfbcbb4a65ab6df95dc3`
(2026-03-30 "Release H100 HGEMM (16 bit) kernels with 32-bit accumulator").
Full upstream lives at `crates/baracuda-kernels-bench/external/cuda-l2/`
(MIT license, copy of LICENSE.txt preserved alongside).

The probes here are stripped versions of `external/cuda-l2/kernels/3090_F32F16F16F32/{128,2048}_4096_4096.cu`
with the torch::Tensor pybind shim removed (it pulls in libtorch headers
and isn't needed for the kernel itself) and a head-to-head cuBLAS
gemmEx timing loop bolted on.

## Files

| Probe | Shape | What it times |
|---|---|---|
| `probe_m128.cu` | M=128, N=K=4096 | CUDA-L2 only (validates compile + correctness on sm_89) |
| `probe_m128_vs_cublas.cu` | M=128, N=K=4096 | CUDA-L2 vs cuBLAS gemmEx side-by-side |
| `probe_m2048.cu` | M=2048, N=K=4096 | CUDA-L2 vs cuBLAS gemmEx side-by-side |

## Building & running

Requires the same CUTLASS headers that `baracuda-cutlass-sys` already
caches at `~/.baracuda-cutlass-sys/checkouts/cutlass-4_2_0/`.

```powershell
# Set up MSVC environment so nvcc finds cl.exe.
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'

# Build a probe (example: M=2048 vs cuBLAS).
nvcc -std=c++17 -O3 -arch=sm_89 `
     --expt-relaxed-constexpr --expt-extended-lambda `
     -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ `
     -U__CUDA_NO_HALF2_OPERATORS__ `
     -I"$env:USERPROFILE/.baracuda-cutlass-sys/checkouts/cutlass-4_2_0/include" `
     -I"$env:USERPROFILE/.baracuda-cutlass-sys/checkouts/cutlass-4_2_0/tools/util/include" `
     probe_m2048.cu -lcublas -o probe_m2048.exe

./probe_m2048.exe
```

## Recorded measurements (RTX 4070, sm_89, CUDA 13.0, 2026-05-28)

| Shape | CUDA-L2 (us) | cuBLAS gemmEx (us) | ratio | Verdict |
|---|---:|---:|---:|---|
| M=128, N=K=4096 (f16/fp32acc) | 175.20 | 177.37 | 0.988 | **CUDA-L2 +1.2%** |
| M=2048, N=K=4096 (f16/fp32acc) | 2452.73 | 2621.46 | 0.936 | **CUDA-L2 +6.4%** |

CUDA-L2 ships kernels for `M ∈ {64, 128, 256, 512, 1024, 2048, 4096, 8192,
12288}` × `K=N=4096` in the 3090 set. It does **not** ship kernels for
`M ∈ {1, 8, 32}` — the decode regime where baracuda's Phase 30 cuBLAS
fast-path already won 3× over the bespoke CUTLASS path. CUDA-L2's
upstream FAQ recommends "find the nearest neighbor configuration (larger
than yours) and pad with zeros" for shapes outside their grid — which
would be silly for M=1 (pad to M=64, 64× the work).
