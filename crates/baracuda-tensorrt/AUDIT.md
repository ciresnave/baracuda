# baracuda-tensorrt — runtime-surface audit (Phase 68)

Audit date: 2026-06-01. Scope: verify `baracuda-tensorrt` exposes a usable
safe wrapper over the TensorRT **inference runtime** surface, and fill Tier-1
gaps. Builder/network/ONNX/INT8/plugin authoring are explicitly out of scope.

## TL;DR — finding + resolution

**TensorRT ships no flat C ABI.** Its public headers (`NvInfer.h`,
`NvInferRuntime.h`, `NvInferRuntimeBase.h`, …) are C++-only. Confirmed against
`NVIDIA/TensorRT/include`: there is **no `NvInferRuntimeCAPI.h` / no
`NvInferCAPI.h`** in the standard distribution. The only `extern "C"` symbols
`libnvinfer` exports are:

| Symbol | Real? |
|---|---|
| `getInferLibVersion` | ✅ real export |
| `createInferRuntime_INTERNAL` | ✅ real export (the factory behind the inline `createInferRuntime`) |
| everything else on the runtime path (`deserializeCudaEngine`, the engine/context getters, tensor binding, `enqueueV3`, …) | ❌ C++ vtable methods — **no flat symbol** |

**Resolution (this phase): ship the C++ shim.** Following baracuda's
established convention for C++-only libraries, `baracuda-tensorrt-sys` now
carries `shim/trt_shim.cpp` — a translation unit that defines the flat `trt*`
`extern "C"` symbols by forwarding to the C++ API. `build.rs` compiles it with
`cc` and statically links it, behind the `shim` feature.

The shim is **link-clean by design**: it references *no* libnvinfer symbol.
The Rust side resolves the two real exports (`createInferRuntime_INTERNAL`,
`getInferLibVersion`) via `libloading`, then hands the resulting `IRuntime*`
(as an opaque `void*`) into the shim functions, which dispatch every operation
through the object's **vtable** (the inline public wrappers forward to
`mImpl->…`). A vtable call needs only the class layout from the headers — no
link-time symbol — and lands in the libnvinfer module the Rust loader has
already mapped. Teardown is `delete` (the public `virtual ~X()`), referencing
only the C++ runtime's `operator delete`. Net effect:

- **Build time:** the `shim` feature needs the TensorRT SDK *headers* (and the
  CUDA headers they include for `cudaStream_t`) — but **no import library**.
- **Run time:** `libnvinfer` is loaded **dynamically** via `libloading`, exactly
  as the rest of the baracuda `-sys` crates do. No link-time TensorRT dependency.
- **Default build (`shim` off):** nothing is compiled, no SDK required; the
  runtime ops are stubs and `Runtime::deserialize_engine` returns
  `Error::ShimNotBuilt`. `version()` + `Runtime` construction still work.

### Verification status

The shim could not be *compiled* in this session — the dev box has no TensorRT
(no `nvinfer*.dll`, no `CUDA_PATH`/`TENSORRT_PATH`), and the SDK headers
transitively include CUDA headers that aren't present either. Instead, **every
shim signature was verified line-by-line against the real TensorRT 10.7
headers** (downloaded from `NVIDIA/TensorRT@release/10.7`):

- `Dims64 { int32_t nbDims; int64_t d[8]; }` ⟷ baracuda `trtDims_t` — exact
  layout match (incl. the 4-byte pad before the `int64_t[]`), so struct-by-value
  return of `Dims` is ABI-compatible.
- `DataType` (kFLOAT=0 … kINT4=9) and `TensorIOMode` (kNONE/kINPUT/kOUTPUT=0/1/2)
  ⟷ baracuda `trtDataType_t` / `trtTensorIOMode_t` — exact value match.
- `ExecutionContextAllocationStrategy` kSTATIC=0/kON_PROFILE_CHANGE=1 ⟷
  `Static`/`OnProfileChange`.
- Method signatures (return type, params, const-ness) for `IRuntime`,
  `ICudaEngine`, `IExecutionContext`, `IHostMemory` — all confirmed, including
  the public `virtual ~X()` destructors that make `delete` correct.
- **Bug fixed:** `createInferRuntime_INTERNAL` is `(void* logger, int32_t
  version)`; the old binding omitted `version` (the factory would have rejected
  the call). The safe wrapper now passes `getInferLibVersion()` as the version.

The default (`shim` off) build is `cargo check`-clean. The `shim`-on path
compiles the same Rust extern block whose signatures are exercised by the
identical-signature stubs in the default build; the remaining unverified step is
the one-time C++ compile/link, which runs on a TensorRT-equipped box (CI or the
RTX 4070 dev box once the SDK is installed).

## Inventory table

Legend: **Exposed** = safe wrapper present, runs when built with `--features
shim` against a TensorRT install; **Partial** = wrapper exists but limited;
**Missing** = no exposure.

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
| `IExecutionContext` lifecycle | Exposed (shim) | `create_execution_context`, `create_execution_context_with_strategy`, `Drop`. |
| Tensor binding (`set_tensor_address`) | Exposed (shim) | `set_tensor_address` (unsafe), `tensor_address`, `set_input_shape`, `tensor_shape`. |
| `enqueueV3` | Exposed (shim) — improved this phase | Accepts a baracuda `&Stream` (`enqueue_v3`) **and** a raw `cudaStream_t` (`enqueue_v3_raw`). Previously raw-only. |
| Multi-stream concurrent | Exposed (shim) | One `ExecutionContext` + one `Stream` per concurrent inference (contexts are independent). The type lifetimes already allow it; no dedicated helper. |

### Memory + tooling

| Capability | Status | Notes |
|---|---|---|
| Engine serialize / deserialize | Exposed (shim) | `Runtime::deserialize_engine` (+ `deserialize` alias), `Engine::serialize` → `HostMemory::as_slice`. Round-trip to disk supported. |
| Tensor bytes-per-component | Exposed (shim) — new | `Engine::tensor_bytes_per_component` (added this phase; helps size device buffers). |
| Optimization profile | Partial | Query-only: `Engine::num_optimization_profiles`. No profile *selection* (`setOptimizationProfileAsync`) yet. |
| `setOutputType` / `setMemoryPoolLimit` dials | Missing | Build-config side (`IBuilderConfig`); not a runtime knob. |
| Runtime version | Exposed ✅ | `version()` → real `getInferLibVersion`. Works without the shim. |
| Plugin compatibility check | Missing | Needs `IPluginRegistry` wiring. |

## Changes made this phase

1. **C-ABI shim shipped** — `baracuda-tensorrt-sys/shim/trt_shim.cpp` (24
   `extern "C"` forwarders) + `build.rs` `cc` compile behind the `shim` feature
   + a TensorRT/CUDA header probe (`TENSORRT_INCLUDE_DIR` / `TENSORRT_PATH`,
   `CUDA_PATH`). Link-clean (vtable dispatch only; libnvinfer stays
   dynamically loaded). See the TL;DR for the design + verification.
2. **`-sys` loader trimmed** to the two real exports; the ~23 runtime ops are
   now `extern "C"` shim symbols (feature on) or graceful stubs (feature off,
   default). Added `SHIM_BUILT` const. Fixed the `createInferRuntime_INTERNAL`
   `(logger, version)` signature.
3. **`enqueue_v3(&self, stream: &Stream)`** — the coordination ask. Accepts
   baracuda's `Stream` (added `baracuda-driver` dep), casting
   `CUstream`→`cudaStream_t` like `baracuda-cudnn::Handle::set_stream`. Raw form
   preserved as `enqueue_v3_raw`.
4. **`Runtime::deserialize_engine`** — primary documented name (matches the
   `ICudaEngine::deserializeCudaEngine` Tier-1 ask); `deserialize` retained as a
   source-compat alias. Returns `Error::ShimNotBuilt` on a shim-less build;
   `shim_built()` lets callers detect it.
5. **Docs** — module docs + both READMEs describe the dynamic-load + shim
   design (previously documented a `Logger` type and APIs that did not exist).
6. **Smoke tests** (`tests/runtime_smoke.rs`): one always-on `Dims` test plus
   two `#[ignore]`d inference tests (version probe + graceful garbage-blob
   deserialize) for `--features shim` on a TensorRT box.

## Building + testing the shim (needs a TensorRT box)

The shim compile/link is the one step not exercisable on a TensorRT-less
machine. To finish validation:

- Install TensorRT **10.x** (matches the `nvinfer` 10/9/8 loader candidates).
  - Windows: set `TENSORRT_PATH` (its `include/` + `lib/`), put `nvinfer_10.dll`
    on `PATH`. CUDA headers via `CUDA_PATH`.
  - Linux: `libnvinfer.so.10` on the loader path; SDK `include/` reachable via
    `TENSORRT_INCLUDE_DIR` or `TENSORRT_PATH`; CUDA via `CUDA_PATH` or
    `/usr/local/cuda`.
- Build: `cargo build -p baracuda-tensorrt --features shim`.
- Make a test engine: `trtexec --onnx=model.onnx --saveEngine=model.engine`.
- Run: `cargo test -p baracuda-tensorrt --features shim -- --ignored`.

## Deferred (future phases)
- Optimization-profile *selection* (`setOptimizationProfileAsync`) for dynamic
  shapes (add a shim symbol + safe method).
- A typed `Logger` (a C++ `ILogger` subclass trampolining to a Rust callback) —
  needs an extra shim object that *creates* a logger; the current path uses a
  null logger or a caller-supplied `ILogger*`.
- Builder path (`INetworkDefinition` / `IBuilder` / ONNX `IParser`) — a separate
  `baracuda-tensorrt-builder` phase.
- INT8 calibration + plugin registry — Tier-2.
