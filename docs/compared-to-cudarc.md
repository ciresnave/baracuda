# baracuda vs cudarc

[`cudarc`](https://docs.rs/cudarc) is the ML-focused CUDA binding from
the `coreylowman/cudarc` ecosystem. It pioneered dynamic loading in
Rust-on-CUDA and is used by `candle`, `burn`, and related ML crates.

## When to use `cudarc`

- You want a mature ML stack today with a large existing user base.
- You like its `CudaSlice<T>` + implicit-device-context design.
- You only need Driver + cuBLAS + cuDNN + cuRAND + NCCL (cudarc's
  primary coverage).

## When to use `baracuda`

- You need a library `cudarc` doesn't wrap — cuTENSOR, nvCOMP, CV-CUDA,
  cuFile, NPP, NVML, CUPTI, nvJPEG, nvJitLink.
- You want **both** the Driver and Runtime APIs as first-class. cudarc
  is Driver-primary; some Runtime APIs are wrapped but the Runtime is
  a second-class citizen.
- You want a less-opinionated memory model. cudarc has a single
  `CudaSlice` type that owns device memory plus a context reference;
  baracuda has separate `DeviceBuffer`, `UnifiedBuffer`, `PinnedVec`,
  `MemoryPool` types that reflect CUDA's own allocator family.
- You want uniform conventions across every library. baracuda's
  `-sys` + safe pair pattern, `CudaStatus` trait, and generic
  `Error<S>` are consistent across 16+ libraries.

## API shape

| concept              | cudarc                          | baracuda                                       |
| -------------------- | ------------------------------- | ---------------------------------------------- |
| context              | `CudaDevice` (owns + manages)   | `Device` + explicit primary-context retain     |
| device-owned buffer  | `CudaSlice<T>`                  | `DeviceBuffer<T>` (Driver) / same name (Runtime) |
| H2D upload           | `dev.htod_copy(&host)?`         | `DeviceBuffer::from_slice(&host)?`             |
| D2H download         | `dev.dtoh_sync_copy(&slice)?`   | `slice.copy_to_host(&mut host)?`               |
| stream               | `CudaStream`                    | `Stream`                                       |
| cuBLAS               | `CudaBlas::new(dev)`            | `baracuda_cublas::Handle::new()`               |
| kernel launch        | `.launch_async(cfg, args)?`     | `LaunchBuilder::new(...).arg(...).launch()?`   |

## Memory semantics

cudarc's `CudaSlice` holds an `Arc<CudaDevice>`, so passing a slice
across threads keeps the device context alive. This is convenient but
bakes a specific ownership model into the type.

baracuda's `DeviceBuffer<T>` holds only a raw pointer + size. The
owning `Context` / `Device` lifetime is the user's responsibility
(typically a top-level singleton). This gives you full control at the
cost of one extra thing to hold onto, and it matches what the
equivalent C code looks like.

## Dynamic loading

Both crates load NVIDIA shared libraries at runtime. The mechanics
differ slightly — cudarc generates static function pointers via
`bindgen` + a shim; baracuda uses `libloading` + a `runtime_fns!`
macro with `OnceLock`-cached PFNs. Behavior is equivalent for users.

## Framework compatibility

`candle`, `burn`, and `dfdx` are built on cudarc. If you're targeting
those ecosystems today, use cudarc.

baracuda's niche is:

1. Applications that need libraries cudarc doesn't cover
   (cuTENSOR for HPC / quantum, nvCOMP for data pipelines, CV-CUDA for
   video, cuFile for GDS).
2. Future ML frameworks / tooling that want a **uniform** base for
   every NVIDIA compute library rather than building per-library
   bespoke wrappers.

## When to mix

Both crates wrap the same opaque C handles. You can hold a
`CudaDevice` from cudarc and get its underlying `CUcontext` via
`dev.cu_primary_ctx()`, then feed that into
`baracuda_driver::Context::from_raw(...)` for baracuda's cuTENSOR /
nvCOMP wrappers. The two don't know about each other, but they both
work against the same driver.
