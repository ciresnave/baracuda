# baracuda-tensorrt-sys

Raw FFI bindings + dynamic loader for **NVIDIA TensorRT** — the
high-performance inference runtime that consumes pre-built engine blobs
and executes them on GPU.

The two real `libnvinfer` exports resolve lazily via
[`libloading`](https://docs.rs/libloading); the rest of the runtime API is
reached through a bundled C++ shim. Either way there is **no link-time
dependency** on `libnvinfer.so` / `nvinfer_10.dll` — it is loaded at runtime.

**Most users want [`baracuda-tensorrt`]** — that crate exposes typed
`Runtime`, `Engine`, `ExecutionContext`, and tensor-binding APIs in idiomatic
Rust.

## Scope: runtime side only

TensorRT's **builder** (network construction, optimization passes, plan
serialization) is C++-only by NVIDIA's design — there is no stable C
ABI for it. Use `trtexec` or the Python bindings to produce engine blobs,
then load them through this crate at inference time.

**Important:** the runtime side has **no flat C ABI either.** `libnvinfer`
exports only `getInferLibVersion` and `createInferRuntime_INTERNAL` as
`extern "C"`; the runtime methods are C++ vtable calls. This crate bundles a
C++ shim (`shim/trt_shim.cpp`) that forwards flat `trt*` symbols to the C++
API, compiled by `build.rs` behind the **`shim` feature** (which needs the
TensorRT SDK headers — set `TENSORRT_INCLUDE_DIR`/`TENSORRT_PATH` + `CUDA_PATH`;
no import library is linked). The shim does pure vtable dispatch, so libnvinfer
stays dynamically loaded at runtime. Without the feature (default), the runtime
ops are stubs. See `../baracuda-tensorrt/AUDIT.md` for the spec + ABI
verification. The symbol surface wrapped here:

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
