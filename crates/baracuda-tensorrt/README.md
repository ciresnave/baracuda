# baracuda-tensorrt

Safe Rust wrappers for **NVIDIA TensorRT** — the high-performance
inference runtime. Loads pre-built engine blobs and executes them on
GPU.

## Status: target API, pending a C-ABI shim

TensorRT exposes **no flat C ABI** — its public headers are C++-only, and
`libnvinfer` exports only `getInferLibVersion` and
`createInferRuntime_INTERNAL` as `extern "C"`. The runtime calls this crate
makes (`deserializeCudaEngine`, `enqueueV3`, tensor binding, …) therefore go
through baracuda-defined `trt*` symbols that a small C++ shim must supply by
forwarding to the C++ vtable API. **That shim is not built yet**, so on a
stock `libnvinfer` these calls fail at symbol-resolution time. The Rust
surface below compiles and is type-correct, but inference cannot run until the
shim lands. See [`AUDIT.md`](AUDIT.md) for the shim spec, the full symbol
list, and the TensorRT install requirement.

## Scope: runtime side only

TensorRT's **builder** (network construction, optimization passes, plan
serialization) is C++-only by NVIDIA's design. Use `trtexec` or the Python
bindings to produce engine blobs, then load them through this crate at
inference time.

The runtime side is what this crate wraps:

- **`Runtime`**: deserialize an engine blob (`deserialize_engine`). Created
  with a raw `ILogger*` (`unsafe Runtime::new`) or none
  (`Runtime::with_null_logger`). A typed safe `Logger` is deferred (it needs
  the shim — see `AUDIT.md`).
- **`Engine`**: inspect IO bindings (names, shapes, dtypes), query the engine
  name + number of optimization profiles, and serialize back to bytes
  (round-trip) via `Engine::serialize` → `HostMemory::as_slice`.
- **`ExecutionContext`**: create with an allocation strategy
  (`Static` / `OnProfileChange` / `UserManaged`), set input shape, set tensor
  address, and enqueue work via `enqueue_v3` (accepts a baracuda `&Stream`) or
  `enqueue_v3_raw` (raw `cudaStream_t`).

```rust,no_run
use baracuda_tensorrt::Runtime;
use baracuda_driver::{Context, Device, Stream};

# fn demo() -> Result<(), Box<dyn std::error::Error>> {
let ctx = Context::new(&Device::get(0)?)?;
let stream = Stream::new(&ctx)?;

// No logger (TRT 10 accepts null); use `unsafe Runtime::new(logger_ptr)` to
// attach an existing C++ ILogger.
let runtime = Runtime::with_null_logger()?;

let engine_bytes = std::fs::read("model.engine")?;
let engine = runtime.deserialize_engine(&engine_bytes)?;
let exec = engine.create_execution_context()?;

// Bind each IO tensor's device address (unsafe — caller owns the memory),
// set dynamic input shapes if any, then enqueue on a baracuda Stream.
// for name in engine IO tensors:
//     unsafe { exec.set_tensor_address(&name, dev_ptr)?; }
unsafe { exec.enqueue_v3(&stream)?; }
# Ok(()) }
```

Pairs with [`baracuda-tensorrt-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-tensorrt-sys`]: https://docs.rs/baracuda-tensorrt-sys
