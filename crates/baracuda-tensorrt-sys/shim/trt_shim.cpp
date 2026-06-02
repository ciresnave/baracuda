// baracuda-tensorrt-sys — C-ABI shim over the TensorRT 10 C++ runtime API.
//
// TensorRT ships no flat C ABI: its public API is C++ only, and `libnvinfer`
// exports just `getInferLibVersion` + `createInferRuntime_INTERNAL` as
// `extern "C"`. This translation unit supplies the flat `trt*` symbols the
// baracuda Rust side calls, each forwarding to a TensorRT C++ method.
//
// Link discipline — this object references NO libnvinfer symbol:
//   * It never calls the factory; the Rust side resolves
//     `createInferRuntime_INTERNAL` via libloading and hands the resulting
//     `IRuntime*` (as an opaque `void*`) into the functions below.
//   * Every operation is a *virtual* call dispatched through the object's
//     vtable (the inline public wrappers in NvInferRuntime.h forward to
//     `mImpl->...`). A vtable call needs only the class layout from the
//     headers — no link-time symbol. The vtable itself lives inside the
//     libnvinfer module that the Rust loader has already mapped into the
//     process, so the calls land in libnvinfer's code at runtime.
//   * Teardown uses `delete`, which dispatches the public `virtual ~X()` and
//     frees with TensorRT's own allocator (the deleting-destructor in the
//     vtable). Only the C++ runtime's `operator delete` is referenced, which
//     the `cc`/host C++ toolchain links automatically.
//
// Consequence: building this shim needs the TensorRT SDK *headers* (and the
// CUDA headers they include for `cudaStream_t`), but no import library, and no
// TensorRT at all when the `shim` feature is off.
//
// Signatures verified against TensorRT 10.7 headers (NvInferRuntime.h,
// NvInferRuntimeBase.h) — see crates/baracuda-tensorrt/AUDIT.md.

#include "NvInferRuntime.h"

#include <cstddef>
#include <cstdint>

using namespace nvinfer1;

// Handles cross the boundary as opaque `void*`; `cudaStream_t` is itself a
// pointer, so it crosses as `void*` too. All structs returned/taken by value
// (`Dims`) use the platform C struct ABI, which matches the Rust `#[repr(C)]`
// `trtDims_t` (same `{ int32_t nbDims; int64_t d[8]; }` layout).

extern "C" {

// ---- IRuntime -----------------------------------------------------------

ICudaEngine* trtRuntimeDeserializeCudaEngine(
    void* runtime, const void* blob, std::size_t size) noexcept {
    return static_cast<IRuntime*>(runtime)->deserializeCudaEngine(blob, size);
}

void trtRuntimeDestroy(void* runtime) noexcept {
    delete static_cast<IRuntime*>(runtime);
}

// ---- ICudaEngine --------------------------------------------------------

void trtCudaEngineDestroy(void* engine) noexcept {
    delete static_cast<ICudaEngine*>(engine);
}

std::int32_t trtCudaEngineGetNbIOTensors(void* engine) noexcept {
    return static_cast<ICudaEngine*>(engine)->getNbIOTensors();
}

const char* trtCudaEngineGetIOTensorName(void* engine, std::int32_t index) noexcept {
    return static_cast<ICudaEngine*>(engine)->getIOTensorName(index);
}

TensorIOMode trtCudaEngineGetTensorIOMode(void* engine, const char* name) noexcept {
    return static_cast<ICudaEngine*>(engine)->getTensorIOMode(name);
}

DataType trtCudaEngineGetTensorDataType(void* engine, const char* name) noexcept {
    return static_cast<ICudaEngine*>(engine)->getTensorDataType(name);
}

Dims trtCudaEngineGetTensorShape(void* engine, const char* name) noexcept {
    return static_cast<ICudaEngine*>(engine)->getTensorShape(name);
}

std::int32_t trtCudaEngineGetTensorBytesPerComponent(void* engine, const char* name) noexcept {
    return static_cast<ICudaEngine*>(engine)->getTensorBytesPerComponent(name);
}

IExecutionContext* trtCudaEngineCreateExecutionContext(void* engine) noexcept {
    return static_cast<ICudaEngine*>(engine)->createExecutionContext();
}

IExecutionContext* trtCudaEngineCreateExecutionContextWithStrategy(
    void* engine, std::int32_t strategy) noexcept {
    return static_cast<ICudaEngine*>(engine)->createExecutionContext(
        static_cast<ExecutionContextAllocationStrategy>(strategy));
}

const char* trtCudaEngineGetName(void* engine) noexcept {
    return static_cast<ICudaEngine*>(engine)->getName();
}

std::int32_t trtCudaEngineGetNbOptimizationProfiles(void* engine) noexcept {
    return static_cast<ICudaEngine*>(engine)->getNbOptimizationProfiles();
}

IHostMemory* trtCudaEngineSerialize(void* engine) noexcept {
    return static_cast<ICudaEngine*>(engine)->serialize();
}

// ---- IExecutionContext --------------------------------------------------

void trtExecutionContextDestroy(void* context) noexcept {
    delete static_cast<IExecutionContext*>(context);
}

bool trtExecutionContextSetInputShape(void* context, const char* name, const Dims* dims) noexcept {
    return static_cast<IExecutionContext*>(context)->setInputShape(name, *dims);
}

Dims trtExecutionContextGetTensorShape(void* context, const char* name) noexcept {
    return static_cast<IExecutionContext*>(context)->getTensorShape(name);
}

bool trtExecutionContextSetTensorAddress(void* context, const char* name, void* data) noexcept {
    return static_cast<IExecutionContext*>(context)->setTensorAddress(name, data);
}

void* trtExecutionContextGetTensorAddress(void* context, const char* name) noexcept {
    // Public method returns `void const*`; baracuda surfaces a mutable pointer
    // (it is the device address the caller themselves bound).
    return const_cast<void*>(static_cast<IExecutionContext*>(context)->getTensorAddress(name));
}

bool trtExecutionContextEnqueueV3(void* context, void* stream) noexcept {
    return static_cast<IExecutionContext*>(context)->enqueueV3(
        reinterpret_cast<cudaStream_t>(stream));
}

// ---- IHostMemory --------------------------------------------------------

void* trtHostMemoryData(void* mem) noexcept {
    return static_cast<IHostMemory*>(mem)->data();
}

std::size_t trtHostMemorySize(void* mem) noexcept {
    return static_cast<IHostMemory*>(mem)->size();
}

void trtHostMemoryDestroy(void* mem) noexcept {
    delete static_cast<IHostMemory*>(mem);
}

}  // extern "C"
