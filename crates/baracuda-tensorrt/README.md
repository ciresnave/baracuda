# baracuda-tensorrt

Safe Rust wrappers for **NVIDIA TensorRT** — the high-performance
inference runtime. Loads pre-built engine blobs and executes them on
GPU.

## How it links: dynamic libnvinfer + a C++ shim

TensorRT exposes **no flat C ABI** — its public headers are C++-only, and
`libnvinfer` exports only `getInferLibVersion` and
`createInferRuntime_INTERNAL` as `extern "C"`. The runtime calls this crate
makes (`deserializeCudaEngine`, `enqueueV3`, tensor binding, …) go through a
small C++ shim (`baracuda-tensorrt-sys/shim/trt_shim.cpp`) that forwards flat
`trt*` symbols to TensorRT's C++ vtable API. The shim references **no**
libnvinfer symbol (pure vtable dispatch on pointers from Rust), so `libnvinfer`
is still loaded **dynamically at runtime** via `libloading` — no link-time
TensorRT dependency.

The shim is compiled (and statically linked) only with the **`shim` feature**,
which needs the TensorRT SDK headers at build time:

```bash
cargo build -p baracuda-tensorrt --features shim   # with TENSORRT_PATH + CUDA_PATH set
```

Without it (the default), `version()` and `Runtime` construction work, but
`Runtime::deserialize_engine` returns `Error::ShimNotBuilt` (check via
`shim_built()`). See [`AUDIT.md`](AUDIT.md) for the full symbol map, the
header-verified ABI, and the build/test recipe.

## Scope: runtime side only

TensorRT's **builder** (network construction, optimization passes, plan
serialization) is C++-only by NVIDIA's design. Use `trtexec` or the Python
bindings to produce engine blobs, then load them through this crate at
inference time.

The runtime side is what this crate wraps:

- **`Runtime`**: deserialize an engine blob (`deserialize_engine`). Created
  with a raw `ILogger*` (`unsafe Runtime::new`) or none
  (`Runtime::with_null_logger`). A typed safe `Logger` is a future addition
  (it needs an extra shim object that constructs an `ILogger` — see `AUDIT.md`).
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
