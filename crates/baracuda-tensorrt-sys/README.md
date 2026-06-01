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

**Important:** the runtime side has **no flat C ABI either.** `libnvinfer`
exports only `getInferLibVersion` and `createInferRuntime_INTERNAL` as
`extern "C"`; the runtime methods are C++ vtable calls. This crate's loader
resolves baracuda-defined `trt*` symbols that a small C++ shim must supply
(forwarding to the C++ API) — that shim is not built yet. See
`../baracuda-tensorrt/AUDIT.md` for the spec. The symbol surface wrapped here:

- `Runtime` create / destroy / deserialize.
- `Engine` inspect (IO names, shapes, dtypes, optimization profiles) /
  serialize back to bytes.
- `ExecutionContext` create with allocation strategy, set input shape,
  set tensor address, `enqueueV3`.
- `HostMemory` data / size / destroy (for serialized blobs).

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-tensorrt`]: https://docs.rs/baracuda-tensorrt
