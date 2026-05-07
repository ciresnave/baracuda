# baracuda-tensorrt-sys

Raw FFI bindings + dynamic loader for **NVIDIA TensorRT** — the
high-performance inference runtime that consumes pre-built engine blobs
and executes them on GPU.

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libnvinfer.so` / `nvinfer64_*.dll`.

**Most users want [`baracuda-tensorrt`]** — that crate exposes typed
`Runtime`, `Engine`, `ExecutionContext`, `Logger`, and tensor-binding
APIs in idiomatic Rust.

## Scope: runtime side only

TensorRT's **builder** (network construction, optimization passes, plan
serialization) is C++-only by NVIDIA's design — there is no stable C
ABI for it. Use `trtexec` or the Python bindings to produce engine blobs,
then load them through this crate at inference time.

The runtime side has a clean C ABI (`libnvinfer`'s `_C` symbols), which
is what this crate wraps:

- `Runtime` create / destroy.
- `Engine` deserialize / inspect (IO names, shapes, dtypes,
  optimization profiles) / serialize back to bytes.
- `ExecutionContext` create with allocation strategy, set input shape,
  set tensor address, `enqueueV3`.
- `Logger` callback registration.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-tensorrt`]: https://docs.rs/baracuda-tensorrt
