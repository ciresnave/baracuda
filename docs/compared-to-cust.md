# baracuda vs cust

[`cust`](https://docs.rs/cust) is the safe host-side Driver-API binding of the
**Rust-CUDA** project ([`Rust-GPU/Rust-CUDA`](https://github.com/Rust-GPU/Rust-CUDA)).
Its defining purpose is to be the runtime companion to **`rustc_codegen_nvvm`** —
the rustc backend that compiles ordinary **Rust into PTX**, so you can write GPU
kernels *in Rust* rather than in CUDA C.

Status matters here and is easy to get wrong. Rust-CUDA went dormant after early
2022, then was **rebooted in January 2025** under the Rust-GPU org (maintainers
jorge-ortega, LegNeato). The git repo is actively developed again — pinned to a
recent Rust nightly, CUDA 12.x in CI, kernels-in-Rust demonstrably working. **But
the published crate is still `cust 0.3.2` from February 2022**; the reboot's
progress is git-only, so consuming it today means pinning a git revision.

The headline difference from baracuda is a **capability**, not a coverage gap:

- **Rust-CUDA compiles Rust → NVVM IR → PTX.** You author kernels in Rust, share
  types and logic between CPU and GPU, and can even use some crates.io crates
  on-device. This is the whole reason the project exists.
- **baracuda's kernel generator emits CUDA C / portable-C** and wraps bespoke
  `.cu`. It never compiles Rust to PTX.

If "kernels in the same language as the host, no CUDA C" is your requirement,
Rust-CUDA is the only option and baracuda is not a substitute. If you don't need
that, baracuda avoids the costs that capability imposes (a pinned nightly and a
build-time toolkit — see below).

## When cust / Rust-CUDA is the better choice

- **You want to write kernels in Rust** — the `rustc_codegen_nvvm` + `cuda_std` +
  `cuda_builder` path. `cust` is the host half that loads and launches them.
- **You're in the Rust-GPU / SPIR-V ecosystem.** Rust-CUDA deliberately shares
  its pinned nightly and code with rust-gpu, and interops with the `glam` /
  `mint` / `bytemuck` math types. One Rust source tree targeting both CUDA and
  SPIR-V → `cust` is the native CUDA host.
- **You need OptiX raytracing** — the repo ships `optix` / `optix-sys`, a domain
  baracuda's compute/ML facade doesn't target at all.

## When baracuda is the better choice

- **You don't want a nightly toolchain or a build-time toolkit.** Rust-CUDA's
  kernel path needs a specific Rust nightly + `libnvvm` (it *is* a rustc
  backend), and `cust` links `libcuda` at build time. baracuda is stable Rust and
  `dlsym`s every symbol at first use, so a binary builds on a CUDA-less CI runner
  and runs against whatever driver is present, picking up library updates without
  a rebuild.
- **You need both the Driver and Runtime APIs.** `cust` is Driver-only; baracuda
  ships separate first-class `baracuda-driver` and `baracuda-runtime` crates.
- **You want ready ops + generated kernels + a correctness oracle**, not just a
  place to launch your own PTX. That's baracuda's `baracuda-kernels` op facade +
  the `baracuda-kernelgen` IR/emitter/oracle stack; Rust-CUDA has no equivalent
  op or codegen-from-IR layer — you bring the kernels, it runs them.
- **You want breadth of NVIDIA libraries under one error model.** Rust-CUDA has
  `cudnn` and `blastoff` (cuBLAS) in-tree (narrow, git-only); baracuda wraps ~22
  libraries (cuBLAS, cuDNN, cuTENSOR, nvCOMP, NCCL, CV-CUDA, cuFile, …) behind a
  shared `CudaStatus` / `Error<S>` convention. (Note: the older "cust ships no
  ML-adjacent libraries at all" framing was wrong for the *project* — it has a
  few; they're just narrower and git-only.)
- **You want releases that reflect current work.** baracuda publishes alpha
  releases; `cust`'s crates.io artifact is four years stale.

## API shape

The host handles line up closely — `cust`'s `Context` / `Stream` /
`Module::from_ptx*` / `DeviceBuffer<T>` + launch builder map onto
`baracuda-driver`'s equivalents almost 1:1. `cust` has Rust-GPU-specific
conveniences (the `DeviceCopy` derive); baracuda has a broader marker-trait
vocabulary — `DeviceRepr`, `KernelArg`, `CudaStatus` in
[`baracuda-types`](../crates/baracuda-types/). The biggest surface difference
isn't the handles: `cust` assumes you'll feed it PTX (ideally Rust-compiled),
while baracuda assumes it (or its generator) produced the kernel for you.

## Mixing the two

Both wrap the same opaque driver handles (`CUdevice` / `CUcontext` / `CUstream` /
`CUevent`) and can move them across in **either ownership direction**: `as_raw()`
hands baracuda's handle out, and `unsafe from_raw` (baracuda takes ownership,
destroys on drop) / `unsafe borrow_raw` (non-owning, for a handle Rust-CUDA still
owns) adopt a foreign one — on `Context` / `Stream` / `Event`, plus
`Device::from_raw`. For context sharing specifically,
`baracuda_driver::PrimaryContext::retain(device)` gives baracuda a handle to the
device's primary context. A typical reason to mix: you write kernels in Rust with
Rust-CUDA but want baracuda's cuDNN / cuTENSOR wrappers alongside, driving them on
the same context and stream.
