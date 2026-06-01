# baracuda-tensorrt — runtime-surface audit (Phase 68)

Audit date: 2026-06-01. Scope: verify `baracuda-tensorrt` exposes a usable
safe wrapper over the TensorRT **inference runtime** surface, and fill Tier-1
gaps. Builder/network/ONNX/INT8/plugin authoring are explicitly out of scope.

## TL;DR — the blocking finding

**TensorRT ships no flat C ABI.** Its public headers (`NvInfer.h`,
`NvInferRuntime.h`, `NvInferRuntimeBase.h`, …) are C++-only. Confirmed against
`NVIDIA/TensorRT/include` on GitHub: there is **no `NvInferRuntimeCAPI.h` / no
`NvInferCAPI.h`** in the standard distribution. The only `extern "C"` symbols
`libnvinfer` exports are:

| Symbol | Real? |
|---|---|
| `getInferLibVersion` | ✅ real export |
| `createInferRuntime_INTERNAL` | ✅ real export (the factory behind the inline `createInferRuntime`) |
| everything else this crate loads (`trtRuntimeDeserializeCudaEngine`, `trtCudaEngine*`, `trtExecutionContext*`, `trtHostMemory*`, `destroyInferRuntime`) | ❌ **fictional** — baracuda-invented names, not present in `libnvinfer` |

So the safe wrapper is **structurally complete and type-correct, but cannot
execute against a stock `libnvinfer`**: ~23 of the 25 loaded symbols resolve to
`LoaderError::SymbolUnavailable` (surfaced as `Error::Loader`). Making the
inference path actually run requires a **C++ shim translation unit** that
defines those `trt*` `extern "C"` functions over the real C++ vtable API, built
by `build.rs` and linked against TensorRT. That shim does not exist yet
(`build.rs` only calls `baracuda_build::emit_rerun_hints()`).

This is the dominant Tier-1 gap. It is deferred (not implemented this session)
because authoring + validating ~250 lines of C++ requires a TensorRT install,
which is **not present on the dev machine** (no `nvinfer*.dll`, no `CUDA_PATH`,
no `TENSORRT_PATH`) — and committing unvalidated C++ against headers we cannot
compile contradicts baracuda's verify-on-real-hardware rule. See the shim spec
below; it is ready to implement on a TensorRT-equipped box.

## Inventory table

Legend: **Exposed** = working safe wrapper (modulo the shim); **Partial** =
wrapper exists but limited / blocked on shim; **Missing** = no exposure.

### Engine build path (out of runtime-crate scope by design)

| Capability | Status | Notes |
|---|---|---|
| `INetworkDefinition` (layer API) | Missing | C++-only; belongs to a future `baracuda-tensorrt-builder` crate + shim. |
| `IBuilder` / `IBuilderConfig` | Missing | Precision flags, workspace size — build side. |
| ONNX import (`IParser`) | Missing | Separate lib `libnvonnxparser`; out of scope. |
| INT8 calibration (`IInt8EntropyCalibrator2`) | Missing | Requires a C++ subclass callback shim; Tier-2. |
| Plugin registration (`IPluginRegistry`) | Missing | `trtIPluginRegistry_t` typedef exists in `-sys` but no functions wired. Out of scope per task. |

### Engine run path

| Capability | Status | Notes |
|---|---|---|
| `IExecutionContext` lifecycle | Partial (shim) | `create_execution_context`, `create_execution_context_with_strategy`, `Drop`. Type-correct; needs shim symbols. |
| Tensor binding (`set_tensor_address`) | Partial (shim) | `set_tensor_address` (unsafe), `tensor_address`, `set_input_shape`, `tensor_shape`. |
| `enqueueV3` | **Partial (shim)** — improved this phase | Now accepts a baracuda `&Stream` (`enqueue_v3`) **and** a raw `cudaStream_t` (`enqueue_v3_raw`). Previously raw-only. |
| Multi-stream concurrent | Partial | Achievable today: one `ExecutionContext` + one `Stream` per concurrent inference (contexts are independent). No dedicated helper; the type lifetimes already allow it. |

### Memory + tooling

| Capability | Status | Notes |
|---|---|---|
| Engine serialize / deserialize | Partial (shim) | `Runtime::deserialize_engine` (+ `deserialize` alias added this phase), `Engine::serialize` → `HostMemory::as_slice`. Round-trip to disk supported. |
| Optimization profile | Partial | Query-only: `Engine::num_optimization_profiles`. No profile *selection* (`setOptimizationProfileAsync`) or build-time profile creation. |
| `setOutputType` / `setMemoryPoolLimit` dials | Missing | Build-config side (`IBuilderConfig`); not a runtime knob. |
| Runtime version | Exposed ✅ | `version()` → real `getInferLibVersion`. The one call that works without the shim. |
| Plugin compatibility check | Missing | Needs `IPluginRegistry` wiring. |

## Changes made this phase (Tier-1, compilable without TensorRT)

1. **`enqueue_v3(&self, stream: &Stream)`** — the coordination ask. The binding
   now accepts baracuda's `Stream` (added `baracuda-driver` dep), casting
   `CUstream`→`cudaStream_t` the same way `baracuda-cudnn::Handle::set_stream`
   does. The old raw-pointer form is preserved as `enqueue_v3_raw`.
2. **`Runtime::deserialize_engine`** — promoted to the primary, documented name
   (matches the README + the `ICudaEngine::deserializeCudaEngine` Tier-1 ask);
   `deserialize` retained as a thin source-compat alias.
3. **Module docs** now carry a prominent "requires a C-ABI shim" status block so
   callers are not surprised by `SymbolUnavailable` at runtime.
4. **README reconciled** with the real API (it previously documented a `Logger`
   type, `Runtime::new(&logger)`, and memory-pool queries that do not exist).
5. **Smoke tests** added (`tests/runtime_smoke.rs`): one always-on pure-Rust
   `Dims` test, plus two `#[ignore]`d inference tests (version probe + graceful
   garbage-blob deserialize) for TensorRT-equipped boxes.

## Deferred (next session, needs a TensorRT box)

### The C-ABI shim (the real Tier-1 unblock)

Add `crates/baracuda-tensorrt-sys/shim/trt_shim.cpp` and compile it from
`build.rs` with `cc` (mirroring `baracuda-ozimmu-sys`'s `baracuda_shim.cu`
convention), gated behind a `shim` feature and an env-probe for the TensorRT
SDK. Each baracuda symbol forwards to the C++ method:

```cpp
// trt_shim.cpp — compiled & linked against libnvinfer when feature `shim` is on.
#include "NvInferRuntime.h"
using namespace nvinfer1;

extern "C" {
ICudaEngine* trtRuntimeDeserializeCudaEngine(IRuntime* rt, const void* blob, size_t n) {
    return rt->deserializeCudaEngine(blob, n);
}
void trtCudaEngineDestroy(ICudaEngine* e)            { delete e; }
int32_t trtCudaEngineGetNbIOTensors(ICudaEngine* e)  { return e->getNbIOTensors(); }
const char* trtCudaEngineGetIOTensorName(ICudaEngine* e, int32_t i) { return e->getIOTensorName(i); }
TensorIOMode trtCudaEngineGetTensorIOMode(ICudaEngine* e, const char* n) { return e->getTensorIOMode(n); }
DataType trtCudaEngineGetTensorDataType(ICudaEngine* e, const char* n)   { return e->getTensorDataType(n); }
Dims trtCudaEngineGetTensorShape(ICudaEngine* e, const char* n)          { return e->getTensorShape(n); }
int32_t trtCudaEngineGetTensorBytesPerComponent(ICudaEngine* e, const char* n) { return e->getTensorBytesPerComponent(n); }
IExecutionContext* trtCudaEngineCreateExecutionContext(ICudaEngine* e)   { return e->createExecutionContext(); }
IExecutionContext* trtCudaEngineCreateExecutionContextWithStrategy(ICudaEngine* e, ExecutionContextAllocationStrategy s)
    { return e->createExecutionContext(s); }
const char* trtCudaEngineGetName(ICudaEngine* e)                 { return e->getName(); }
int32_t trtCudaEngineGetNbOptimizationProfiles(ICudaEngine* e)  { return e->getNbOptimizationProfiles(); }
IHostMemory* trtCudaEngineSerialize(ICudaEngine* e)             { return e->serialize(); }

void trtExecutionContextDestroy(IExecutionContext* c)           { delete c; }
bool trtExecutionContextSetInputShape(IExecutionContext* c, const char* n, const Dims* d) { return c->setInputShape(n, *d); }
Dims trtExecutionContextGetTensorShape(IExecutionContext* c, const char* n) { return c->getTensorShape(n); }
bool trtExecutionContextSetTensorAddress(IExecutionContext* c, const char* n, void* p) { return c->setTensorAddress(n, p); }
void* trtExecutionContextGetTensorAddress(IExecutionContext* c, const char* n) { return c->getTensorAddress(n); }
bool trtExecutionContextEnqueueV3(IExecutionContext* c, cudaStream_t s) { return c->enqueueV3(s); }

void  trtRuntimeDestroy(IRuntime* r)        { delete r; }   // back `destroyInferRuntime`
void* trtHostMemoryData(IHostMemory* m)     { return m->data(); }
size_t trtHostMemorySize(IHostMemory* m)    { return m->size(); }
void  trtHostMemoryDestroy(IHostMemory* m)  { delete m; }
}
```

Notes for the implementer:
- `Dims` returned by value matches `-sys`'s `#[repr(C)] trtDims_t` only if
  `nvinfer1::Dims` is `{ int32_t nbDims; int64_t d[8]; }` — verify against the
  installed `NvInferRuntimeBase.h` (it is, for TRT 10).
- `destroyInferRuntime` in `-sys` should be repointed at `trtRuntimeDestroy`
  (current symbol name is also fictional).
- `IRuntime::deserializeCudaEngine` has overloads; the `(blob, size)` one is the
  target.

### TensorRT install requirement (for tests + shim build)

- TensorRT **10.x** (matches the `nvinfer` 10/9/8 loader candidates in `-sys`).
- Windows: install the TRT zip, set `TENSORRT_PATH`, put `nvinfer_10.dll` (and
  `nvinfer_plugin_10.dll` if plugins are used) on `PATH`; headers at
  `%TENSORRT_PATH%\include`, import libs at `%TENSORRT_PATH%\lib`.
- Linux: `libnvinfer.so.10` on the loader path; headers under the SDK `include/`.
- Produce a test engine with `trtexec --onnx=model.onnx --saveEngine=model.engine`
  for the `#[ignore]`d round-trip test.

### Other deferred items
- Optimization-profile *selection* (`setOptimizationProfileAsync`) for dynamic
  shapes (needs a shim symbol + safe method).
- A typed `Logger` (C++ `ILogger` subclass that trampolines to a Rust callback)
  — requires a shim; the README example's logger was removed until then.
- Builder path (`INetworkDefinition` / `IBuilder` / ONNX `IParser`) — a separate
  `baracuda-tensorrt-builder` phase.
- INT8 calibration + plugin registry — Tier-2.
