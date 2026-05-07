# baracuda-tensorrt

Safe Rust wrappers for **NVIDIA TensorRT** — the high-performance
inference runtime. Loads pre-built engine blobs and executes them on
GPU.

## Scope: runtime side only

TensorRT's **builder** (network construction, optimization passes, plan
serialization) is C++-only by NVIDIA's design — there's no stable C ABI
for it. Use `trtexec` or the Python bindings to produce engine blobs,
then load them through this crate at inference time.

The runtime side has a clean C ABI which this crate wraps:

- **`Runtime`**: create with a typed Logger, deserialize an engine blob.
- **`Engine`**: inspect IO bindings (names, shapes, dtypes), serialize
  back to bytes (round-trip), query optimization profiles, query memory
  pool limits.
- **`ExecutionContext`**: create with allocation strategy
  (`OnProfileChange` / `Static`), set input shape, set tensor address,
  enqueue work via `enqueueV3`.
- **`Logger`**: callback-based logger with severity filtering.

```rust,no_run
use baracuda_tensorrt::{Runtime, Logger, Severity};
use baracuda_driver::{Context, Device, Stream};

# fn demo() -> Result<(), Box<dyn std::error::Error>> {
let ctx = Context::new(&Device::get(0)?)?;
let stream = Stream::new(&ctx)?;

let logger = Logger::new(Severity::Warning, |sev, msg| {
    eprintln!("[trt {sev:?}] {msg}");
});
let runtime = Runtime::new(&logger)?;

let engine_bytes = std::fs::read("model.engine")?;
let engine = runtime.deserialize_engine(&engine_bytes)?;
let mut ctx_exec = engine.create_execution_context()?;

// Bind tensors via set_input_shape + set_tensor_address, then enqueue.
unsafe { ctx_exec.enqueue_v3(&stream)?; }
# Ok(()) }
```

Pairs with [`baracuda-tensorrt-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-tensorrt-sys`]: https://docs.rs/baracuda-tensorrt-sys
