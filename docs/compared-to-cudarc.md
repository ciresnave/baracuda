# baracuda vs cudarc

[`cudarc`](https://docs.rs/cudarc) (repo [`chelsea0x3b/cudarc`](https://github.com/chelsea0x3b/cudarc),
formerly `coreylowman/cudarc`) is the mature, minimal, safe Rust binding
layer over the CUDA toolkit. As of this writing it is at **0.19.x** (CUDA
11.4–13.3, selectable by feature), actively maintained, ~6M downloads, and the
CUDA backend under HuggingFace **candle**. If you are choosing a Rust CUDA
*binding*, cudarc is the incumbent and a safe default.

The most important thing to understand up front: **cudarc and baracuda are not
the same kind of thing.** cudarc is a binding library — safe wrappers over the
driver and the NVIDIA math/DL libraries, and nothing above that. baracuda
contains a binding layer *and* two layers cudarc does not attempt:

- **bindings** (`baracuda-driver` / `-runtime` + ~22 library wrappers) — the
  layer cudarc competes at;
- an **op facade** (`baracuda-kernels`) — ready-made ML ops (softmax, layernorm,
  the GEMM families, attention, …) as `Plan` types, not just library handles you
  assemble yourself;
- a **kernel generator + correctness stack** (`baracuda-kernelgen`) — a neutral
  op IR that *emits* specialized CUDA (and portable-C) kernels, validated against
  an independent CPU oracle, with precision-tiered variants and a live JIT seam.

So the honest comparison is layered: at the binding layer they overlap heavily;
above it they diverge. cudarc's binding breadth has grown a lot — the old "we
wrap more libraries" pitch mostly no longer holds — so baracuda's real
differentiation today is the op + codegen layers, not the bindings.

## When cudarc is the better choice

- **You target candle** (or its model zoo). candle is built directly on cudarc's
  context/stream/`CudaSlice` model; baracuda can't substitute without
  reimplementing that integration. This is the one true lock-in, and it's a good
  reason. (burn has moved to its own `cubecl` stack and dfdx is largely dormant,
  so candle is the live downstream that matters.)
- **You want a stable, battle-tested binding shipping today.** cudarc is
  semver-versioned, has years of production mileage, and tracks new CUDA toolkits
  within weeks. baracuda is `0.0.1-alpha` with fluid internals and an explicit
  "pin exact versions" warning. For anything you need to ship now, cudarc is
  lower-risk.
- **You want minimal footprint.** cudarc is one focused dependency with three
  predictable tiers (`sys` / `result` / `safe`) and no op / IR / oracle / codegen
  concepts to learn. If you just need safe raw CUDA + the math libraries and will
  write your own kernels (via cudarc's `nvrtc` or precompiled PTX), it's lighter
  to adopt than a full facade + codegen stack.
- **You only need libraries cudarc already covers.** Its coverage is now broad:
  driver, `nvrtc`, cuBLAS / cuBLASLt, cuDNN, cuRAND, cuFFT, cuSPARSE,
  cuSOLVER(+Mg), cuTENSOR, cuFILE (GDS), NCCL, CUPTI, nvtx — plus CUDA graphs,
  stream priorities, and P2P.

## When baracuda is the better choice

- **You want ops, not just handles.** cudarc hands you a cuBLAS handle; you
  assemble the op yourself (candle exists precisely to add that layer).
  `baracuda-kernels` presents hundreds of ready ops through one `Plan` API, each
  routing to the right NVIDIA library or a bespoke kernel.
- **You want generated / specialized kernels with a correctness guarantee.**
  `baracuda-kernelgen` emits specialized kernels from an IR, checks them
  bit-for-bit against an independent CPU oracle, and can offer precision-first
  variants (e.g. a more-accurate, bitwise-reproducible reduction). cudarc
  generates nothing and has no oracle — it is a pass-through to the vendor
  libraries.
- **You need a library cudarc still doesn't wrap** — NPP, nvJPEG / nvJPEG2000,
  nvCOMP, CV-CUDA, NVML, nvJitLink. (This list is now *short*: cudarc has since
  added cuTENSOR and cuFile, which earlier versions of this doc wrongly claimed
  as baracuda-exclusive.)
- **You want both the Driver and Runtime APIs as first-class.** cudarc's safe
  surface is Driver-centric; its Runtime module is thin / unsafe. baracuda ships
  separate first-class `baracuda-driver` and `baracuda-runtime` crates.
- **You want one uniform error / convention model across every library** — the
  `-sys` + safe-pair pattern, the `CudaStatus` trait, and a generic `Error<S>`
  across all ~22 wrappers.

## API shape (cudarc ≈ 0.19, 2026 — a moving target)

cudarc restructured to a **stream-centric** model; the older
`CudaDevice`-owns-everything design is gone. Its API has changed across releases,
so check its current docs for exact signatures — the feel today:

| concept       | cudarc (0.19)                                   | baracuda                                   |
| ------------- | ----------------------------------------------- | ------------------------------------------ |
| context       | `CudaContext::new(id) -> Arc<CudaContext>`      | `Device` + explicit primary-context retain |
| stream        | `ctx.default_stream() -> CudaStream`            | `Stream`                                   |
| device buffer | `CudaSlice<T>`                                  | `DeviceBuffer<T>` (raw ptr + len)          |
| H2D upload    | `stream.memcpy_stod(&host)?`                    | `DeviceBuffer::from_slice(&host)?`         |
| D2H download  | `stream.memcpy_dtov(&slice)?`                   | `slice.copy_to_host(&mut host)?`           |
| kernel launch | `stream.launch_builder(&f).arg(&x).launch(cfg)` | `LaunchBuilder::new(..).arg(..).launch()?` |
| cuBLAS        | `CudaBlas::new(stream)`                         | `baracuda_cublas::Handle::new()`           |

Two deliberate design differences, unchanged by cudarc's restructure:

- **Memory ownership.** cudarc's `CudaSlice` carries `Arc`s to its context and
  stream, so a slice keeps the context alive across threads — foolproof, but it
  bakes one ownership model into the type. baracuda's `DeviceBuffer<T>` holds
  only a raw pointer + length; the owning `Device` / `Context` lifetime is yours
  to manage (typically a top-level singleton). More control, one more thing to
  hold, closer to what the equivalent C looks like. (baracuda also offers
  `ManagedBuffer<T>` for unified/managed memory, `PinnedBuffer<T>` for pinned
  host memory, and a `MemoryPool` — a small allocator family mirroring CUDA's
  own, rather than a single slice type.)
- **Dynamic loading.** Both load NVIDIA shared libraries at runtime, so neither
  needs the CUDA toolkit at build time. cudarc uses a bindgen shim; baracuda uses
  `libloading` + a `runtime_fns!` macro with `OnceLock`-cached PFNs. Equivalent
  for users.

## Mixing the two

Both wrap the same opaque driver handles and can coexist against one device. For
sharing a **context**, the clean path is the primary context:
`baracuda_driver::PrimaryContext::retain(device)` returns baracuda's handle to
the device's primary context — the same one cudart-based frameworks use — so you
don't juggle raw pointers. Otherwise, baracuda broadly exposes `as_raw()` to hand
a raw `CUcontext` / `CUstream` to another library's API that accepts one;
adopting a *foreign* raw handle back into a safe baracuda type is currently
limited (the owning RAII handles — `Context` / `Stream` / `Event` — expose
`as_raw()` but no `from_raw`), so mixing leans on baracuda-hands-out rather than
baracuda-wraps-in. A common setup: keep candle-on-cudarc for your model and add
baracuda's cuTENSOR / nvCOMP / CV-CUDA wrappers on the shared primary context.
