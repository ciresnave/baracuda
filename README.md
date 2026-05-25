# baracuda

![A great barracuda — the project's namesake, minus one letter.](https://raw.githubusercontent.com/ciresnave/baracuda/refs/heads/main/assets/barracuda.png)

> **About the name.** Yes, we know — it's spelled **barracuda** (two Rs). That
> name was taken on crates.io, so we dropped one R and kept swimming.

A unified Rust ML-op facade over the NVIDIA CUDA ecosystem.

![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)
![Status](https://img.shields.io/badge/status-alpha.37-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900)
![Tests](https://img.shields.io/badge/regression-1958%2F0-success)

## What baracuda is

baracuda is a Rust workspace that exposes every primitive an ML framework
expects — the union of PyTorch (`torch.*` + `nn.functional`) and JAX
(`jax.lax.*` + `jax.numpy.*`) — through a single `Plan`-based crate surface
called [`baracuda-kernels`]. Internally each plan dispatches to:

1. The appropriate NVIDIA-library wrapper crate (cuBLAS, cuDNN, cuFFT,
   cuSOLVER, cuRAND, cuSPARSE, cuTENSOR, NPP, CV-CUDA, CUTLASS) when one
   already covers the op well, or
2. A bespoke hand-rolled `.cu` kernel shipped in [`baracuda-kernels-sys`]
   when no NVIDIA library covers the op (or covers it poorly at the shapes
   that matter for modern transformer / vision / GNN workloads).

Callers import **one** crate (`baracuda-kernels`) and reach for **one** API
style. The dispatch decision — which is observable through
`Plan::sku()` for telemetry — is otherwise invisible. Switching from a
CUTLASS-backed SKU to a bespoke-backed SKU is a layout flag, not an import
change.

baracuda is for downstream Rust ML / inference / training frameworks that
need access to the full CUDA stack without re-vendoring it themselves. The
workspace also ships idiomatic stand-alone wrappers for every CUDA library
under `crates/baracuda-<lib>` if you want to skip the kernel facade and
talk to one library directly.

## Status

**In active development — alpha.37.** Roughly **1958 GPU tests passing**
on an RTX 4070 (sm_89), across **616 binary targets**.

Phase coverage (see [`ARCHITECTURE.md`](ARCHITECTURE.md) for the phase
matrix):

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Crate scaffolding, shared type vocabulary | done |
| 1 | int8 GEMM RRR (Fuel-blocking, 18 SKUs) | done |
| 2 | FP8 / int4 / bin GEMM completion | done |
| 3 | Elementwise + shape / layout (Categories B, B', C, C', D, N) | done |
| 4 | Reductions + scans + random (Categories E, F, Q) | done |
| 5 | Normalization + softmax + loss (Categories G, H, R) | done |
| 6 | Attention + linalg + FFT (Categories K, Linalg, U) | done |
| 7 | Convolution + pooling + indexing + embedding + segment (Categories I, J, L, M, S) | done |
| 8 | Quantization helpers + GGUF + MoE (Category P, V) | done |
| 9 | Sort / topk / image / NMS (Categories O, T) | done |
| 10 | sm_89 (Ada Lovelace) tuning sweep | done |
| 11 | Fuel feedback integration (alpha.27) — ScalarType ergonomics, Conv/Pool fanout, GGUF Q8_K MMVQ, i64 indices, Sparsemax cap lift, atomicAdd-via-CAS, build-env probe | done |
| 12 | PowI + ArgMax/Min u32/i32 outputs (alpha.28) — `IndexOutputElement` sealed trait | done |
| 13 | WriteSlice + Contiguize + sub-byte casts + Triu/Tril (alpha.29) — KV-cache fast path, retires Fuel's D2H/CPU/H2D fallback, plus `DeviceBuffer::zero()` (alpha.30) | done |
| 14 | Strided FFI siblings (alpha.31) — Affine, PowI, Triu/Tril, RoPE+SDPA, GGUF MMVQ activation-strided + W byte offset; 56 new FFI symbols | done |
| 15 | Quick wins + correctness cleanup (alpha.32) — MMVQ alignment guard, OneHot/Nonzero i64 wrappers, MoE fixture race fix | done |
| 16 | Pool completion (alpha.33) — bit-exact adaptive pool {1,2,3}d, bespoke LpPool {1,2}d, bespoke FractionalMaxPool {2,3}d; 48 new FFI symbols | done |
| 17 | SDPA / attention completion (alpha.34) — Flash SDPA sm_89 strided FW + SDPA BW GQA-broadcast atomicAdd | done |
| 18 | Sub-byte / quantized completeness (alpha.35) — f16/bf16 activations for `GgufMmvqPlan` across all 11 block formats × contig + strided; 44 new FFI symbols | done |
| 19 | Fuel retirement asks (alpha.36) — pool/conv FFI facade for cuDNN-backed plans + Upsample Nearest2d + NEW im2col/im2col1d/col2im1d bespoke; vendored Fuel Q8_1 for inspection; 140 new FFI symbols. Surfaced 1.0-freeze prereq for broader library-backed FFI facade audit | done |
| 20 | MoE — Item 4 from Fuel retirement (alpha.37): batched MMVQ × N-experts (36 new FFI symbols across 11 GGUF block formats × 3 activation dtypes + 3 pure-FP); MoE absorb-and-expose proved to be a no-op (Fuel hadn't evolved their kernels since Phase 8.5 vendor; 5 baracuda-side symbols already match) + 2 direct-FFI smoke tests | done |
| 21+ | Broader FFI facade audit (1.0-freeze prereq), segment + embedding BW completion, linalg completion, Hopper / Blackwell, 1.0 freeze | pending (see [`ROADMAP.md`](ROADMAP.md)) |

API stability is **not** promised before beta.0. Breaking changes ship in
each alpha bump and are documented in the workspace `CHANGELOG.md`.

## Quick start

Add the kernel facade and the driver crate:

```toml
[dependencies]
baracuda-kernels = { version = "0.0.1-alpha.37", features = ["sm89", "cudnn"] }
baracuda-driver  = "0.0.1-alpha.37"
```

A representative example — single-axis numerically stable softmax over a
device-resident tensor:

```rust,no_run
use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    PlanPreference, SoftmaxArgs, SoftmaxDescriptor, SoftmaxKind, SoftmaxPlan,
    TensorMut, TensorRef, Workspace,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Standard CUDA bring-up via baracuda-driver.
    let ctx = Context::new(&Device::get(0)?)?;
    let stream = Stream::new(&ctx)?;

    // 2. Allocate device input + output buffers (rank-2: rows × cols).
    let rows = 32i32;
    let cols = 1024i32;
    let n_elems = (rows * cols) as usize;
    let dev_x: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_elems)?;
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_elems)?;

    // 3. Build the descriptor — pure shape + dtype + op-kind, no handles.
    let desc = SoftmaxDescriptor::<2> {
        kind: SoftmaxKind::Softmax,
        input_shape: [rows, cols],
        softmax_axis: 1,
        element: <f32 as baracuda_kernels::Element>::KIND,
    };

    // 4. Plan selection — picks a kernel SKU (bespoke softmax kernel here).
    let plan = SoftmaxPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())?;

    // 5. Args carry the per-call tensor handles + strides.
    let args = SoftmaxArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: [rows, cols], stride: [cols as i64, 1] },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: [rows, cols], stride: [cols as i64, 1] },
    };

    // 6. Launch. Workspace::None for plans that need no scratch.
    plan.run(&stream, Workspace::None, args)?;
    stream.synchronize()?;
    Ok(())
}
```

The same `select` → `run` shape applies to every op. GEMM, attention,
conv2d, FFT, scatter — the descriptor / args fields differ per family but
the lifecycle is identical. See the [`crates/baracuda-kernels`
README](crates/baracuda-kernels/README.md) for the int8-GEMM variant of
the same example.

## Workspace layout

The user-facing crates a typical caller will reach for:

```text
baracuda-kernels             # the unified Plan-based ML op facade
baracuda-kernels-types       # shared type vocabulary (Element, TensorRef, KernelSku, ...)
baracuda-kernels-sys         # raw FFI to bespoke .cu kernels
baracuda-kernels-bench       # criterion harness for sm_89 perf sweeps (not published)
baracuda-cutlass             # safe wrapper for CUTLASS GEMM (float, int8 RCR, batched, grouped)
baracuda-driver              # safe wrapper for the CUDA Driver API
baracuda-runtime             # safe wrapper for the CUDA Runtime API
```

The per-library wrappers used internally by the facade (you can also use
them stand-alone):

```text
baracuda-cublas{,-sys}       # cuBLAS + cuBLASLt + cuBLASXt
baracuda-cudnn{,-sys}        # cuDNN classic + Graph API
baracuda-cufft{,-sys}        # cuFFT
baracuda-cusolver{,-sys}     # cuSOLVER dense + sparse + Rf + Mg
baracuda-cusparse{,-sys}     # cuSPARSE
baracuda-curand{,-sys}       # cuRAND
baracuda-cutensor{,-sys}     # cuTENSOR
baracuda-npp{,-sys}          # NPP
baracuda-nccl{,-sys}         # NCCL
baracuda-cvcuda{,-sys}       # CV-CUDA
baracuda-nvjpeg{,-sys}       # nvJPEG
baracuda-nvcomp{,-sys}       # nvCOMP
```

And the supporting low-level crates (FFI, build infrastructure, profiling):

```text
baracuda-cuda-sys            # Driver + Runtime FFI
baracuda-nvrtc{,-sys}        # runtime CUDA C++ → PTX
baracuda-nvjitlink{,-sys}    # CUDA 12+ JIT linker
baracuda-cupti{,-sys}        # profiling APIs
baracuda-nvml{,-sys}         # device monitoring
baracuda-cufile{,-sys}       # GPUDirect Storage (Linux-only)
baracuda-tensorrt{,-sys}     # TensorRT inference runtime
baracuda-forge              # build-time .cu → PTX compiler driver
baracuda-build              # build.rs helpers
baracuda-core                # loader + Error plumbing
baracuda-types{,-derive}     # pure-data types: Half, BFloat16, Complex, DeviceRepr
```

The full umbrella crate (`baracuda`) re-exports everything behind cargo
features — convenient when you want everything; overkill when you don't.

## Hardware support

baracuda targets **Ampere and newer** by design. Pre-Ampere GPUs lack the
tensor-core instructions and async-copy primitives the bespoke kernels are
written against (`mma.sync.m16n8k*`, `cp.async`, `ldmatrix`), and we have
no desire to ship a slower SIMT fallback for hardware that's eight years
old.

| Compute capability | NVIDIA marketing names | baracuda support |
| --- | --- | --- |
| sm_80 | Ampere (A100, A40, A30, RTX 30xx) | **default baseline** |
| sm_89 | Ada Lovelace (RTX 40xx, L40, L4) | feature-gated specialized kernels (FP8, larger Flash Attention tiles) |
| sm_90a | Hopper async (H100, H200) | stubs in place; full specialization pending Phase 11 |
| sm_100 | Blackwell | post-Phase-11 |
| ≤ sm_75 (Turing, Volta, Pascal, …) | — | **unsupported** |

The default `sm80` build runs forward-compatibly on Ada and Hopper through
JIT-compiled PTX; turn on `sm89` to pick up the FP8 and Flash-Attention
sibling plans tuned for Ada's larger register file.

## Cargo features

The kernel facade exposes a small feature set:

| Feature | Default | Effect |
| --- | --- | --- |
| `sm80` | yes | Build the Ampere-baseline kernel set. |
| `sm89` | no | Build the Ada Lovelace specializations (FP8 GEMM, `FlashSdpaSm89Plan`). |
| `sm90a` | no | Build the Hopper-specialized kernels (stubs today). |
| `cudnn` | no | Link cuDNN and enable conv / pool / `CtcLossCudnnPlan`. |

`cudnn` is off by default because cuDNN is a separate NVIDIA download not
bundled with the stock CUDA toolkit installer. Enabling it without cuDNN
installed produces a linker error on `cudnn.lib` / `libcudnn.so` — see
the building section for the auto-discovery paths the build script probes.

## Building

Requirements:

- **CUDA Toolkit ≥ 12.0** with `nvcc` on `PATH`. baracuda is tested on
  12.x and 13.x.
- **cuDNN 9.x** (only if you enable the `cudnn` feature) — separate
  NVIDIA download, not bundled with the toolkit.
- **A working Rust toolchain ≥ 1.85** (workspace MSRV pinned in
  `rust-toolchain.toml`).
- **Windows users**: `lld-link.exe` somewhere on `PATH`. The CUDA `nvcc`
  invocation links through it; the install location is typically
  `C:\Program Files\LLVM\bin`. Install the LLVM Windows package and add
  that directory to `PATH` if `cargo build` complains about
  `lld-link.exe` not being found.

A typical full build with all GPU-side features (CUDA toolkit + cuDNN
present):

```bash
cargo build -p baracuda-kernels --features sm89,cudnn --release
```

Or, to verify the public API surface compiles without the full kernel
build (fast — type-check only):

```bash
cargo check -p baracuda-kernels --features sm89,cudnn
```

The `baracuda-kernels-sys` build script auto-discovers cuDNN at the
following paths in order: `CUDNN_PATH` / `CUDNN_ROOT` / `CUDNN_HOME` env
vars, then `C:\Program Files\NVIDIA\CUDNN\v<X.Y>\` on Windows, then the
CUDA toolkit's own `lib/` directory (pre-cuDNN-9 layout), then the
standard Linux distro paths under `/usr/lib/`.

## Troubleshooting

### Windows: Git-for-Windows fake `link.exe` shadowing the MSVC linker

Git-for-Windows ships a GNU coreutils binary named `link.exe` at
`C:\Program Files\Git\usr\bin\link.exe` — its job is to create a hard
link, **not** to link object files. If that directory appears on `PATH`
ahead of the MSVC linker (or LLVM's `lld-link.exe`), `cargo build`
invokes the coreutils binary instead of the real linker and fails with a
cryptic error (it doesn't understand `/OUT:` and friends).

baracuda's `baracuda-kernels-sys` and `baracuda-cutlass-sys` build
scripts probe `PATH` on Windows and emit a `cargo:warning` if they
detect this shadowing. **Fix:** re-order `PATH` so the MSVC linker
(typically reached via the Visual Studio "x64 Native Tools Command
Prompt") or LLVM's `lld-link.exe` (`C:\Program Files\LLVM\bin\`) appears
before `C:\Program Files\Git\usr\bin\`. Building from the VS x64 Native
Tools prompt is the most reliable option; alternatively, install LLVM
and put its `bin` directory ahead of Git's on the user/system `PATH`.

## Testing

baracuda's GPU integration tests are gated behind `#[ignore]` so a
host-only `cargo test` doesn't try to launch a kernel on a machine
without an NVIDIA driver. To run them you need a working GPU plus the
`--ignored` flag:

```bash
# Host-only tests (compile + reference logic; no GPU access):
cargo test -p baracuda-kernels --lib

# Full GPU integration sweep — RTX 30xx / 40xx / 50xx required:
cargo test -p baracuda-kernels --release -- --ignored

# Verify the workspace-level API surface compiles (no GPU needed):
cargo check -p baracuda-kernels --features sm89,cudnn
```

The full regression on an RTX 4070 covers 324 binary targets at
~1630 tests passing. Individual op-family suites take 30–90 seconds;
the full sweep is 25–40 minutes.

## Benchmarks

The `baracuda-kernels-bench` crate is a criterion-based harness with
CUDA-event-timed throughput sweeps across GEMM, Flash Attention, and
Conv2d at LLM-typical and ResNet-typical shapes. It is **not** published
to crates.io (it depends on a working GPU).

```bash
cargo bench -p baracuda-kernels-bench --features sm89,cudnn
```

The full sweep takes ~30 minutes on an RTX 4070. Scope to a single family
with `--bench gemm` / `--bench flash_attention` / `--bench conv2d`. See
[`crates/baracuda-kernels-bench/BENCH-sm89.md`](crates/baracuda-kernels-bench/BENCH-sm89.md)
for the baseline table format and methodology.

## Project documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — layered design, Plan-Descriptor-Args
  pattern, `KernelSku` taxonomy, dispatcher design, workspace contract,
  sibling-plan pattern, vendoring convention, phase roadmap.
- `OP-MATRIX.md` — full op × dtype × backend coverage matrix (planned).
- `LESSONS.md` — postmortems, ABI footguns, performance traps (planned).
- Per-crate `README.md` files under `crates/<name>/`.

## License

Dual-licensed under [MIT](LICENSE-MIT) **or** [Apache-2.0](LICENSE-APACHE).
Pick whichever fits your project. Contributions accepted under the same
terms.

NVIDIA's CUDA libraries (`libcuda`, `libcudart`, `libcublas`, `libcudnn`,
…) are **not** redistributed by this project. You obtain them from NVIDIA
separately — either through the CUDA Toolkit installer or through each
library's dedicated download page. baracuda's loader opens whatever the
host driver / toolkit has installed.

## Vendor attribution

A small number of bespoke kernels in `baracuda-kernels-sys` are vendored
from upstream open-source projects (huggingface/candle's CUDA kernel set
via `fuel-cuda-kernels`; llama.cpp's `ggml-cuda` GGUF block-format
quantization + MMVQ; `guoqingbao/attention.rs`'s fused MoE expert
kernels). Each adapted source carries an `SPDX-FileCopyrightText:` +
`SPDX-License-Identifier:` header; the consolidated provenance is in
[`crates/baracuda-kernels-sys/LICENSE-thirdparty.md`](crates/baracuda-kernels-sys/LICENSE-thirdparty.md).

The [`baracuda-forge`](crates/baracuda-forge) build-time kernel-compiler
crate is a vendored fork of [`cudaforge`](https://github.com/guoqingbao/cudaforge)
by **Guoqing Bao** — see [`crates/baracuda-forge/NOTICE`](crates/baracuda-forge/NOTICE)
for the upstream commit hash.

The [`baracuda-cutlass`](crates/baracuda-cutlass) safe wrapper for NVIDIA
CUTLASS — plan-based GEMM and grouped-GEMM with caller-supplied
workspace, MoE-friendly variable-M-per-group dispatch — was specified
by the **Fuel ML library team**. See
[`crates/baracuda-cutlass/NOTICE`](crates/baracuda-cutlass/NOTICE) for
the design lineage.

[`baracuda-kernels`]: crates/baracuda-kernels
[`baracuda-kernels-sys`]: crates/baracuda-kernels-sys
