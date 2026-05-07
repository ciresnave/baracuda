# baracuda-driver

Safe Rust wrappers for the **CUDA Driver API**.

Takes the raw FFI from [`baracuda-cuda-sys`] and dresses it up with RAII
handles, typed memory, lifetime-checked slices, kernel-launch builders,
async memcpy, and graph capture. Doesn't hide the Driver-API model:
contexts are explicit, modules are explicit, streams are explicit.

```rust,no_run
use baracuda_driver::{Context, Device, DeviceBuffer, Module, Stream};

# fn demo() -> baracuda_driver::Result<()> {
let device = Device::get(0)?;
let ctx = Context::new(&device)?;
let stream = Stream::new(&ctx)?;

let host = vec![1.0f32, 2.0, 3.0, 4.0];
let dev = DeviceBuffer::from_slice(&ctx, &host)?;

let mut back = vec![0.0f32; host.len()];
dev.copy_to_host(&mut back)?;
stream.synchronize()?;
assert_eq!(host, back);
# Ok(()) }
```

## Coverage

Comprehensive. Every Driver-API surface area is wrapped:

- **Device** — enumeration, attributes, compute capability, P2P queries.
- **Context** — explicit + primary-context reuse.
- **Module / Function** — PTX, CUBIN, fatbin loading; kernel entry-point lookup.
- **Memory** — `DeviceBuffer<T>`, `DeviceSlice<T>`, `DeviceSliceMut<T>`,
  `PinnedBuffer<T>`, `ManagedBuffer<T>`, memory-pool allocations,
  IPC, VMM, multicast.
- **Stream + Event** — ordered async work + timing/synchronization.
- **Graph** — capture, conditional nodes, switch nodes, exec instantiation,
  graph cloning.
- **LaunchBuilder** — `cuLaunchKernel` + `cuLaunchKernelEx` with launch
  attributes (cluster dims, programmatic stream serialization, priority).
- **Texture / Surface / Tensor map / Array** — full texture pipeline.
- **Green contexts** — SM-partitioned execution (CUDA 12.4+).
- **External memory + semaphores** — Vulkan/D3D12 interop.
- **Coredump, profiler control, occupancy queries.**

## Quickstart vs `baracuda-runtime`

If you want to compile kernels at build time and never think about
contexts or modules, use [`baracuda-runtime`] instead — it's the safe
wrapper for the higher-level Runtime API. baracuda-driver is for
people who want explicit control over module loading, kernel launch
parameters, graph capture, etc.

Both are first-class; pick whichever fits the workload. The two crates
share `Stream` / `Event` / `Graph` interop so you can mix them.

## Async feature

Enable the `async` feature to get a `futures-core::Future` impl on
`Event`, so you can `.await` a CUDA event from any async runtime
without baracuda picking one for you.

```toml
baracuda-driver = { version = "0.0.1-alpha.7", features = ["async"] }
```

## Dynamic loading

Like every safe-wrapper crate in baracuda, this one uses
[`baracuda-cuda-sys`]'s dynamic loader: `cargo build` succeeds with no
CUDA installed, and the resulting binary opens whatever `libcuda.so` /
`nvcuda.dll` is on the host.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cuda-sys`]: https://docs.rs/baracuda-cuda-sys
[`baracuda-runtime`]: https://docs.rs/baracuda-runtime
