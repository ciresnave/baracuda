# baracuda vs cust

[`cust`](https://docs.rs/cust) is the Driver-API binding maintained by
the Rust-GPU project. It predates baracuda and is well-maintained.

## When to use `cust`

- You're specifically targeting the Rust-GPU PTX emitter; `cust` is
  the canonical host-side companion.
- You only need the Driver API and no ML-adjacent libraries.

## When to use `baracuda`

- You need **both** the Driver and Runtime APIs. `cust` is Driver-only.
- You need **any** NVIDIA library (cuBLAS, cuDNN, cuTENSOR, nvCOMP,
  NCCL, CV-CUDA, cuFile, …) — `cust` doesn't ship these at all.
- You want the same `Error` / `Result` conventions across every
  library. baracuda shares a `CudaStatus` trait and a generic
  `Error<S>` enum across all safe crates.
- You want dynamic loading. `cust` requires a CUDA Toolkit at build
  time; baracuda doesn't (every symbol is `dlsym`'d at first use).

## API shape

| concept          | cust                     | baracuda                                 |
| ---------------- | ------------------------ | ---------------------------------------- |
| initialization   | `cust::init()`           | lazy; first call to any API              |
| context          | `cust::context::Context` | `baracuda_driver::Context`               |
| device pointer   | `DeviceBuffer<T>`        | `baracuda_driver::memory::DeviceBuffer<T>` (same name, same semantics) |
| stream           | `Stream`                 | `baracuda_driver::Stream`                |
| module load      | `Module::from_ptx_cstr`  | `baracuda_driver::module::Module::from_ptx` |
| kernel launch    | builder → `.launch()`    | same builder, same feel                  |

The core handles line up almost 1:1. Biggest surface difference: cust
has richer Rust-GPU-specific conveniences (`DeviceCopy` derive, etc.);
baracuda has a broader marker-trait vocabulary in
[`baracuda_types`](../crates/baracuda-types/) (DeviceRepr, KernelArg,
CudaStatus).

## Dynamic loading

cust uses `build.rs` to link against `libcuda` at build time — you
must have a CUDA Toolkit installed on the build machine, and the
resulting binary is pinned to the driver version you built against.

baracuda uses `libloading` at runtime. A binary compiled on a
CUDA-less CI runner runs fine on any target with a driver installed,
and it picks up library updates without rebuild.

## When to mix

You can depend on both. `cust` and `baracuda-driver` wrap the same C
types (`CUdevice`, `CUcontext`, `CUstream`, `CUevent`) as opaque
handles; both expose `as_raw()` / `from_raw()` escape hatches that let
you hand a handle between the two layers. Typical reason: you have
existing code that wraps `cust::Context` but want baracuda's cuDNN /
cuTENSOR wrappers alongside.
