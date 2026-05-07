# baracuda-cuda-sys

Raw FFI bindings + dynamic loader for the **NVIDIA CUDA Driver and Runtime APIs**.

This is the foundational `-sys` crate of the baracuda workspace. It
covers `libcuda` (Driver API: `cu*` symbols) and `libcudart` (Runtime
API: `cuda*` symbols) in a single crate, with a versioned symbol
resolver that uses `cuGetProcAddress_v2` to pick the right ABI when CUDA
exposes multiple versions of the same call (e.g. `cuCtxCreate_v3` vs
`cuCtxCreate_v2`).

`cargo build` succeeds with **no CUDA installed**: symbols resolve at
runtime via [`libloading`](https://docs.rs/libloading), and the
resulting binary opens whatever `libcuda.so.1` / `nvcuda.dll` /
`libcudart.so` the host has.

**Most users want [`baracuda-driver`] or [`baracuda-runtime`] instead** —
those crates expose RAII handles, typed memory, lifetime-checked slices,
and a kernel-launch builder over the same symbols.

## What's exposed

- **Driver API** (`cu_*`): contexts, modules, kernels, memory, streams,
  events, graphs, textures, surfaces, tensor maps, multicast, external
  memory + semaphores, IPC, peer access, graphics interop, profiler,
  green contexts.
- **Runtime API** (`cuda_*`): the same surface area where it overlaps,
  plus runtime-only symbols (memory pools, set-device, runtime graph
  capture).
- Type definitions for every `CU*` / `cuda*` enum, struct, and handle.

## Version policy

CUDA floor: **11.4**. Symbols added after 11.4 are bound but resolve to
`None` at load time on older drivers — the safe-wrapper crates surface
that as `Feature::Unavailable` rather than panicking.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-driver`]: https://docs.rs/baracuda-driver
[`baracuda-runtime`]: https://docs.rs/baracuda-runtime
