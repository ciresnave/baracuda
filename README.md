# baracuda

![A great barracuda — the project's namesake, minus one letter.](assets/barracuda.png)

> **About the name.** Yes, we know — it's spelled **barracuda** (two Rs). That
> name was taken on crates.io, so we dropped one R and kept swimming.

Idiomatic, ergonomic Rust wrappers for the entire NVIDIA CUDA stack: the Driver
and Runtime APIs plus the ML-adjacent libraries (NVRTC, cuBLAS + cuBLASLt + cuBLASXt,
cuFFT, cuSPARSE, cuSOLVER + cuSOLVERMg, cuRAND, cuTENSOR, cuDNN, NCCL, NPP,
nvJPEG, nvCOMP, CV-CUDA, NVML, CUPTI, cuFile, nvJitLink, TensorRT, cuDF).

Every library is wrapped as a two-crate pair — `baracuda-<lib>-sys` for the
raw FFI + dynamic loader, and `baracuda-<lib>` for the safe, idiomatic API —
and re-exported through the umbrella [`baracuda`] crate behind cargo features.

![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)
![CUDA](https://img.shields.io/badge/CUDA-11.4%E2%80%9313.x-76b900)

## Why baracuda

- **Breadth.** Every library in NVIDIA's compute stack, wrapped with the same
  conventions — handles are RAII, scalars are generic over `f32` / `f64` /
  `Complex32` / `Complex64` where the math allows, errors flow through a shared
  `CudaStatus` trait.
- **Dynamic loading everywhere.** `cargo build` needs *no* CUDA toolkit
  installed. The resulting binary runs against whatever NVIDIA driver + CUDA
  runtime the user has — or fails loudly with a typed loader error if they
  don't.
- **Version-flexible.** CUDA 11.4 floor, targets latest 13.x. Newer APIs (green
  contexts, multicast, cuDNN Graph API, conditional graph nodes, etc.) are
  feature-gated at the safe layer but always present in the FFI.
- **Two-layer crates.** If you only need the type vocabulary (`Half`,
  `BFloat16`, `Complex32/64`, `DeviceRepr`, `CudaVersion`), depend on
  [`baracuda-types`] alone — no loader, no CUDA, no transitive weight.

## Getting started

Add the umbrella crate to your `Cargo.toml`:

```toml
[dependencies]
baracuda = { version = "0.1", features = ["driver", "runtime"] }
```

```rust
use baracuda::runtime::{DeviceBuffer, set_device};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_device(0)?;
    let host = vec![1.0f32, 2.0, 3.0];
    let d = DeviceBuffer::from_slice(&host)?;

    let mut back = vec![0.0f32; 3];
    d.copy_to_host(&mut back)?;
    assert_eq!(host, back);
    Ok(())
}
```

Turn on additional libraries with features:

```toml
baracuda = { version = "0.1", features = ["ml"] }
# ml = driver + runtime + nvrtc + nvjitlink + cublas + cufft + cusparse
#      + cusolver + curand + cudnn + nccl + tensorrt
```

Full guides live under [`docs/guides/`](docs/guides/):

- [Getting started](docs/guides/getting-started.md) — install, first kernel, common pitfalls.
- [Matrix multiply with cuBLAS](docs/guides/matmul-cublas.md) — SGEMM + GemmEx + cuBLASLt.
- [NVRTC runtime compilation](docs/guides/nvrtc.md) — compile CUDA C++ to PTX at runtime.
- [Async pipelines](docs/guides/streams-and-events.md) — overlap H2D / compute / D2H.
- [CUDA graphs](docs/guides/cuda-graphs.md) — capture and replay.
- [Memory patterns](docs/guides/memory-patterns.md) — device / unified / pinned / pool.
- [Using feature flags](docs/guides/feature-flags.md) — picking the right bundle.

See [`examples/`](examples/) for end-to-end runnable programs.

## Workspace layout

```text
baracuda/
├── crates/
│   ├── baracuda/                 umbrella crate — re-exports per feature
│   ├── baracuda-types/           pure-data types: Half, BFloat16, Complex, DeviceRepr
│   ├── baracuda-types-derive/    #[derive(DeviceRepr)], #[derive(KernelArg)]
│   ├── baracuda-core/            loader (libloading), Error plumbing, stream_mode
│   ├── baracuda-build/           build.rs helpers
│   ├── baracuda-cuda-sys/        Driver + Runtime FFI
│   ├── baracuda-driver/          safe Driver API
│   ├── baracuda-runtime/         safe Runtime API
│   ├── baracuda-nvrtc{,-sys}     runtime CUDA C++ → PTX
│   ├── baracuda-nvjitlink{,-sys} CUDA 12+ JIT linker
│   ├── baracuda-cublas{,-sys}    cuBLAS + cuBLASLt + cuBLASXt
│   ├── baracuda-cufft{,-sys}     cuFFT
│   ├── baracuda-cusparse{,-sys}  generic sparse API
│   ├── baracuda-cusolver{,-sys}  Dn + Sp + Rf + Mg
│   ├── baracuda-curand{,-sys}    RNG + quasi-random
│   ├── baracuda-cudnn{,-sys}     classic ops + Graph API
│   ├── baracuda-nccl{,-sys}      multi-GPU collectives
│   ├── baracuda-npp{,-sys}       image / signal primitives (workhorse subset)
│   ├── baracuda-nvjpeg{,-sys}    JPEG codec (single + batched + hybrid + encoder)
│   ├── baracuda-nvml{,-sys}      device monitoring, ECC, NVLink, events
│   ├── baracuda-cupti{,-sys}     profiling: activity + callback + event/metric + profiler host
│   ├── baracuda-cutensor{,-sys}  contraction / reduction / elementwise / permutation
│   ├── baracuda-nvcomp{,-sys}    LZ4 / Snappy / Zstd / GDeflate / Deflate / Bitcomp / ANS / Cascaded / Gzip / CRC32
│   ├── baracuda-cvcuda{,-sys}    computer-vision operator catalog
│   ├── baracuda-cufile{,-sys}    GPUDirect Storage (Linux-only)
│   ├── baracuda-tensorrt{,-sys}  runtime inference (engine deserialize + execute)
│   └── baracuda-cudf{,-sys}      RAPIDS GPU DataFrames (skeleton — tracks libcudf_c)
├── assets/                       README imagery
├── docs/                         design notes + per-feature guides
└── examples/                     runnable end-to-end programs
```

## Feature flags (umbrella crate)

```toml
default   = ["driver", "runtime"]

# Core APIs
driver    nvrtc   nvjitlink   runtime

# Math libraries
cublas    cufft   cusparse    cusolver   curand   cutensor

# Deep learning
cudnn     nccl    tensorrt

# Data / I/O / imaging
cudf      npp     nvjpeg      nvcomp     cvcuda

# System tooling
nvml      cupti   cufile

# Convenience bundles
math      = ["cublas", "cufft", "cusparse", "cusolver", "curand"]
dl        = ["cudnn", "nccl", "tensorrt"]
imaging   = ["npp", "nvjpeg", "cvcuda"]
ml        = ["driver", "runtime", "nvrtc", "nvjitlink", "math", "dl"]
full      = ["ml", "imaging", "nvcomp", "nvml", "cupti", "cufile", "cudf", "cutensor"]
```

## Coverage status

Legend: ✅ comprehensive — full host-facing surface wrapped idiomatically · 🟢
workhorse — enough for 95% of uses, with a documented tail · 🟡 scaffolding —
loader + smoke coverage, specific ops on request.

| API / library    | Status | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---------------- | :----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CUDA Driver API  |   ✅    | Device / Context (primary + explicit + green) / Module / Library / Function / Kernel / Memory (device + host pinned + managed + pool + VMM + IPC) / Stream / Event / Graph (capture + exec + conditional + switch nodes) / Launch / Occupancy / Texture / Surface / Tensor Map / Multicast / External memory + semaphore / Peer / Graphics interop / Profiler                                                                                                                                                                                                |
| CUDA Runtime API |   ✅    | Mirror of Driver API for everything that Runtime exposes; zero-cost interop of `Stream`/`Event`/`Graph` between the two                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| NVRTC            |   ✅    | `Program`, compile options, include-path registration, error-log retrieval                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| nvJitLink        |   ✅    | Full linker API                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| cuBLAS + LT + XT |   ✅    | L1/L2/L3 S/D/C/Z + real-only L2 (symv/trmv/trsv/ger/syr) + Ex variants (axpy/dot/nrm2/scal/rot) + batched GEMM + batched direct solvers (getrf / getrs / getri / matinv) + GemmEx + GemmStridedBatchedEx + cuBLASLt (descriptors, preference, heuristics, matmul) + cuBLASXt multi-GPU GEMM                                                                                                                                                                                                                                                                  |
| cuFFT            |   ✅    | Plan1d / Plan2d / Plan3d / PlanMany + R2C / C2R / C2C / D2Z / Z2D / Z2Z + XT multi-GPU + callbacks                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| cuSPARSE         |   ✅    | CSR / CSC / COO / BSR + DnVec / DnMat + SpMV / SpMM / SpGEMM (3-phase) / SpSV / SpSM / SDDMM + sparse↔dense / CSR↔CSC conversions + sparse-BLAS-1 (axpby / gather / scatter / rot)                                                                                                                                                                                                                                                                                                                                                                           |
| cuSOLVER         |   ✅    | Dense: LU / QR (geqrf + orgqr + ormqr) / Cholesky (potrf + potrs + potri) / SVD / syevd / syevj / gesvdj / syevj\_batched / gesvdj\_batched / gels S/D/C/Z + generic 64-bit X… (Xgetrf/Xgetrs/Xgeqrf/Xpotrf/Xpotrs/Xsyevd); Sparse: csrlsvchol / csrlsvqr; Refactor: full Rf; Multi-GPU: cuSOLVERMg (getrf / potrf / syevd)                                                                                                                                                                                                                                  |
| cuRAND           |   ✅    | Host + device generators, quasi-random (Sobol + scrambled), all documented distributions (uniform, normal, log-normal, Poisson, binomial)                                                                                                                                                                                                                                                                                                                                                                                                                    |
| cuTENSOR         |   ✅    | `Handle`, `ComputeDescriptor`, `TensorDescriptor`, `OperationDescriptor`, `PlanPreference`, `Plan` + full op catalog: Contraction, Reduction, ElementwiseBinary, ElementwiseTrinary, Permutation, BlockSparseContraction, TrinaryContraction + plan-cache I/O                                                                                                                                                                                                                                                                                                |
| cuDNN            |   🟢    | Classic ops (activation, conv forward + backward data + backward filter + backward bias, pooling, softmax, BN inference + training + backward, LRN, dropout, op-tensor, reduce, transform, add, scale), spatial transformer, CTC loss, RNN forward + backward, N-D tensor / filter / pooling / conv descriptors, full Graph (backend) API scaffolding with `BackendDescriptor::set_attribute_raw`                                                                                                                                                            |
| NCCL             |   ✅    | Communicators + all collectives (all-reduce, reduce, reduce-scatter, all-gather, broadcast, reduce, send/recv) + p2p + group API + memory alloc/free + comm split + register/deregister + abort/finalize                                                                                                                                                                                                                                                                                                                                                     |
| nvJPEG           |   ✅    | Single + batched decode (simple + three-phase hybrid: host / transfer / device), planar decode, buffer pools (pinned + device), DecodeParams (output format / ROI / CMYK), JpegStream parser, full encoder (quality, chroma subsampling, optimized Huffman, encode image / YUV, retrieve bitstream)                                                                                                                                                                                                                                                          |
| NPP              |   🟡    | `version()` + sample arithmetic; NPP has thousands of function variants — expansion on request                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| nvCOMP           |   ✅    | LZ4, Snappy, Zstd, GDeflate, Gzip, Deflate, Bitcomp, ANS, Cascaded (compress + decompress async, temp-size, max-chunk-size, alignment queries) + CRC32 + status / property APIs                                                                                                                                                                                                                                                                                                                                                                              |
| CV-CUDA          |   🟢    | ~30 operators — geometric (Resize / PillowResize / Warp\* / Remap / Rotate / Crop / CopyMakeBorder), filters (Gaussian / Median / Average / Laplacian / Bilateral / Motion / Conv2D), morphology, edges (Canny), thresholds, color (ColorTwist / BrightnessContrast / GammaContrast), stats, composite, misc                                                                                                                                                                                                                                                 |
| NVML             |   ✅    | Enumeration, memory / temperature / power / fan / utilization, clock info (current / max / applications / default), applications-clock set, power-limit get / set / range, P-states (power + performance), temperature thresholds, ECC (per-location + total error counts, mode get / set), PCI info / link gen / link width / throughput, NVLink state / version / capability, running processes (compute + graphics), compute mode, UUID / serial / index / minor number, handle lookup by UUID / PCI bus ID, event set with register + wait, field values |
| CUPTI            |   ✅    | Activity API + record walker, Callback API (`Subscriber` RAII), Event API (Group RAII + device/domain enumeration + read), Metric API (id-by-name, attributes, value), Profiler Host API (initialize / de-initialize / begin session / end session / set config / begin pass / end pass / enable / disable / push range / pop range / flush / counter availability), `cuptiGetResultString` helper                                                                                                                                                           |
| cuFile           |   ✅    | Driver lifecycle + properties, file-handle register, buffer / stream register, sync + async read / write, BatchIO (setup / submit / poll / cancel / destroy), configurable direct-I/O / cache / pinned-mem limits, op-status error-string helper                                                                                                                                                                                                                                                                                                             |
| TensorRT         |   🟢    | Runtime-side inference: Runtime, Engine (deserialize + inspect IO / shapes / dtypes + serialize back to bytes), ExecutionContext (with allocation strategy), set input shape, set tensor address, enqueueV3. **Builder side remains C++-only by TensorRT's design** — use `trtexec` or the Python API to produce engine blobs, then load them here.                                                                                                                                                                                                          |
| cuDF             |   🟡    | Tracks RAPIDS `libcudf_c`'s emerging C ABI — CSV / Parquet I/O, Column, Table, TypeId. Anything not exposed through libcudf\_c itself is bounded by upstream progress.                                                                                                                                                                                                                                                                                                                                                                                       |

## Examples

| Example                                          | Demonstrates                                    |
| ------------------------------------------------ | ----------------------------------------------- |
| [`hello_kernel`](examples/hello_kernel.rs)       | Driver API: load PTX, launch kernel, readback   |
| [`hello_runtime`](examples/hello_runtime.rs)     | Runtime API: allocation + H2D + D2H             |
| [`matmul_cublas`](examples/matmul_cublas.rs)     | SGEMM via `baracuda-cublas`                     |
| [`nvrtc_jit`](examples/nvrtc_jit.rs)             | Compile CUDA C++ at runtime and run it          |
| [`stream_pipeline`](examples/stream_pipeline.rs) | Overlap H2D / compute / D2H on multiple streams |
| [`graph_capture`](examples/graph_capture.rs)     | Record and replay a graph                       |

## Comparison

| Feature             |   `baracuda`   | `cust`  |    `cudarc`     | `rustacuda` |
| ------------------- | :------------: | :-----: | :-------------: | :---------: |
| CUDA Driver API     |       ✅        |    ✅    |        ✅        |  ✅ (stale)  |
| CUDA Runtime API    |       ✅        |    —    |     partial     |      —      |
| ML-stack libraries  | ✅ (everything) |    —    | ✅ (opinionated) |      —      |
| cuBLASLt / cuBLASXt |       ✅        |    —    |        —        |      —      |
| TensorRT            |   🟢 runtime    |    —    |        —        |      —      |
| CUPTI profiling     |       ✅        |    —    |        —        |      —      |
| Dynamic loading     |       ✅        | ✅ (opt) |        ✅        |      —      |
| Active maintenance  |       ✅        |    ✅    |        ✅        |  abandoned  |

See [`docs/compared-to-cust.md`](docs/compared-to-cust.md) and
[`docs/compared-to-cudarc.md`](docs/compared-to-cudarc.md) for detailed
positioning.

## Platform support

| Platform        | Driver | Runtime | Libraries                                              |
| --------------- | :----: | :-----: | ------------------------------------------------------ |
| Linux x86\_64   |   ✅    |    ✅    | all                                                    |
| Linux aarch64   |   ✅    |    ✅    | all (Jetson: full NvSci support)                       |
| Windows x86\_64 |   ✅    |    ✅    | all except cuFile and CV-CUDA (both are Linux-primary) |
| macOS           |   —    |    —    | NVIDIA dropped Mac CUDA support in 2019                |

## Design highlights

- **Every handle is RAII.** Contexts, streams, events, plans, descriptors,
  communicators — they all have `Drop` impls and `as_raw()` escape hatches.
- **Ownership flows through lifetimes.** Descriptors borrow their backing
  `DeviceBuffer`s; you cannot free a matrix while cuSPARSE still holds a
  pointer to it.
- **Send / !Sync everywhere it matters.** Handles that NVIDIA documents as
  "single-thread at a time" (cuBLAS, cuDNN, cuSPARSE) are `Send` but not
  `Sync`; the ones that are thread-safe (NCCL, most CUDA Graph types) are both.
- **No hidden global state.** The loader's singletons are the only global
  state; `baracuda_core::stream_mode::init` is the one knob you set per
  process if you want non-default per-thread-default-stream semantics.
- **Errors are typed per-library but unified by `CudaStatus`.** An
  application-level error type can absorb all of them through `baracuda::types::CudaStatus`.

## Testing

```bash
# Host-only tests (no GPU required):
cargo test --workspace --lib

# Full workspace build, proves the dynamic-load story works without CUDA:
cargo build --workspace

# GPU-backed integration tests (requires a working NVIDIA driver + CUDA runtime):
BARACUDA_GPU_TESTS=1 cargo test --workspace -- --ignored
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE). Pick
whichever fits your project. Contributions are accepted under the same terms.

NVIDIA CUDA libraries (`libcuda`, `libcudart`, `libcublas`, `libcudnn`,
`libnccl`, `cutensor.so`, `nvcomp64_5.dll`, etc.) are **not redistributed** by
this project. You must obtain them from NVIDIA separately — either through the
CUDA Toolkit or through the individual library's download page. The dynamic
loader opens whatever the driver / toolkit has installed.

The barracuda image in `assets/` is the project's namesake mascot; replace it
with any freely-licensed barracuda photo of your choice.
