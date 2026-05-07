# baracuda-runtime

Safe Rust wrappers for the **CUDA Runtime API**.

The Runtime API is "higher level" than the Driver API: contexts are
implicit (each device has a primary context the runtime manages),
kernels are typically linked at build time, and most operations dispatch
to the current thread's current device. baracuda-runtime mirrors the
Driver-side types where it makes sense (`Device`, `Stream`, `Event`,
`DeviceBuffer`) and provides Runtime-API-specific facilities (memory
pools, graph capture from the runtime side, set-device per thread).

```rust,no_run
use baracuda_runtime::{set_device, DeviceBuffer};

# fn demo() -> Result<(), Box<dyn std::error::Error>> {
set_device(0)?;
let host = vec![1.0f32, 2.0, 3.0];
let dev = DeviceBuffer::from_slice(&host)?;
let mut back = vec![0.0f32; 3];
dev.copy_to_host(&mut back)?;
assert_eq!(host, back);
# Ok(()) }
```

## Coverage

Mirror of [`baracuda-driver`] for everything the Runtime API exposes:

- **Device + set_device** — enumeration, attributes, current-device
  control per thread.
- **Memory** — typed `DeviceBuffer<T>`, slices, async copies, pinned
  host memory, managed memory, memory pools, multicast.
- **Stream + Event** — ordered async work + sync/timing.
- **Graph** — capture and replay; cross-API interop with
  `baracuda-driver::Graph` via `from_raw` / `as_raw`.
- **Launch** — `LaunchBuilder` (typed, lifetime-checked) for typed
  kernel invocations.
- **External memory + semaphores, IPC, VMM, query helpers.**

## Driver / Runtime interop

Enable the `driver-interop` feature (forwarded from the umbrella crate's
`driver-runtime-interop`) to get zero-cost conversion between
`baracuda_driver::Stream` ↔ `baracuda_runtime::Stream`,
`baracuda_driver::Event` ↔ `baracuda_runtime::Event`, etc. Same handle
under the hood — the conversion just changes the Rust type.

## Dynamic loading

Same model as the rest of baracuda: `cargo build` needs no CUDA
installed; the resulting binary opens `libcudart.so` / `cudart64_*.dll`
at runtime via the loader in [`baracuda-cuda-sys`].

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-driver`]: https://docs.rs/baracuda-driver
[`baracuda-cuda-sys`]: https://docs.rs/baracuda-cuda-sys
