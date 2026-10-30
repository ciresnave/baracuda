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

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

using namespace nvinfer1;

namespace {

// An `ILogger` that forwards every message to a Rust callback. TensorRT keeps
// one logger per process and may keep the `ILogger*` it was first handed for as
// long as any builder or runtime lives, so the Rust side creates exactly one of
// these and never destroys it.
class CallbackLogger final : public ILogger {
public:
    using Callback = void (*)(std::int32_t severity, const char* msg, void* user);

    CallbackLogger(Callback cb, void* user) noexcept : cb_(cb), user_(user) {}

    void log(Severity severity, const char* msg) noexcept override {
        cb_(static_cast<std::int32_t>(severity), msg, user_);
    }

private:
    Callback cb_;
    void* user_;
};

}  // namespace

// Handles cross the boundary as opaque `void*`; `cudaStream_t` is itself a
// pointer, so it crosses as `void*` too. All structs returned/taken by value
// (`Dims`) use the platform C struct ABI, which matches the Rust `#[repr(C)]`
// `trtDims_t` (same `{ int32_t nbDims; int64_t d[8]; }` layout).

extern "C" {

// ---- ILogger ------------------------------------------------------------

// Returns an `ILogger*` (upcast before it becomes `void*`, so the pointer
// `createInferRuntime_INTERNAL` receives is a real `ILogger*`).
void* trtLoggerCreate(CallbackLogger::Callback cb, void* user) noexcept {
    return static_cast<ILogger*>(new (std::nothrow) CallbackLogger(cb, user));
}

void trtLoggerDestroy(void* logger) noexcept {
    delete static_cast<ILogger*>(logger);
}

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

// ---- Builder side (IBuilder / IBuilderConfig / INetworkDefinition) -------
//
// Same discipline as the runtime half: every call is a vtable dispatch on a
// pointer the Rust side obtained from libnvinfer's `createInferBuilder_INTERNAL`
// (resolved through libloading), so this object references no libnvinfer
// symbol. Layers cross the boundary as `ILayer*` (never a subclass pointer)
// and are cast back to their concrete type only inside the matching setter.

static inline IBuilder* asBuilder(void* p) noexcept { return static_cast<IBuilder*>(p); }
static inline IBuilderConfig* asConfig(void* p) noexcept { return static_cast<IBuilderConfig*>(p); }
static inline INetworkDefinition* asNetwork(void* p) noexcept { return static_cast<INetworkDefinition*>(p); }
static inline ITensor* asTensor(void* p) noexcept { return static_cast<ITensor*>(p); }
static inline ILayer* asLayer(void* p) noexcept { return static_cast<ILayer*>(p); }
static inline void* Out(ILayer* l) noexcept { return l; }

extern "C" {

// ---- IBuilder -----------------------------------------------------------

void trtBuilderDestroy(void* builder) noexcept { delete asBuilder(builder); }

void* trtBuilderCreateNetwork(void* builder, std::uint32_t flags) noexcept {
    return asBuilder(builder)->createNetworkV2(flags);
}

void* trtBuilderCreateConfig(void* builder) noexcept { return asBuilder(builder)->createBuilderConfig(); }

IHostMemory* trtBuilderBuildSerialized(void* builder, void* network, void* config) noexcept {
    return asBuilder(builder)->buildSerializedNetwork(*asNetwork(network), *asConfig(config));
}

// ---- IBuilderConfig -----------------------------------------------------

void trtConfigDestroy(void* config) noexcept { delete asConfig(config); }

void trtConfigSetMemoryPoolLimit(void* config, std::int32_t pool, std::size_t size) noexcept {
    asConfig(config)->setMemoryPoolLimit(static_cast<MemoryPoolType>(pool), size);
}

std::size_t trtConfigGetMemoryPoolLimit(void* config, std::int32_t pool) noexcept {
    return asConfig(config)->getMemoryPoolLimit(static_cast<MemoryPoolType>(pool));
}

void trtConfigSetFlag(void* config, std::int32_t flag, bool on) noexcept {
    if (on) {
        asConfig(config)->setFlag(static_cast<BuilderFlag>(flag));
    } else {
        asConfig(config)->clearFlag(static_cast<BuilderFlag>(flag));
    }
}

bool trtConfigGetFlag(void* config, std::int32_t flag) noexcept {
    return asConfig(config)->getFlag(static_cast<BuilderFlag>(flag));
}

void trtConfigSetOptimizationLevel(void* config, std::int32_t level) noexcept {
    asConfig(config)->setBuilderOptimizationLevel(level);
}

std::int32_t trtConfigGetOptimizationLevel(void* config) noexcept {
    return asConfig(config)->getBuilderOptimizationLevel();
}

// ---- INetworkDefinition -------------------------------------------------

void trtNetworkDestroy(void* network) noexcept { delete asNetwork(network); }

void* trtNetworkAddInput(void* network, const char* name, std::int32_t type, const Dims* dims) noexcept {
    return asNetwork(network)->addInput(name, static_cast<DataType>(type), *dims);
}

void trtNetworkMarkOutput(void* network, void* tensor) noexcept { asNetwork(network)->markOutput(*asTensor(tensor)); }

std::int32_t trtNetworkGetNbInputs(void* network) noexcept { return asNetwork(network)->getNbInputs(); }
std::int32_t trtNetworkGetNbOutputs(void* network) noexcept { return asNetwork(network)->getNbOutputs(); }
std::int32_t trtNetworkGetNbLayers(void* network) noexcept { return asNetwork(network)->getNbLayers(); }

void* trtNetworkAddElementWise(void* network, void* a, void* b, std::int32_t op) noexcept {
    return Out(asNetwork(network)->addElementWise(*asTensor(a), *asTensor(b), static_cast<ElementWiseOperation>(op)));
}

void* trtNetworkAddActivation(void* network, void* in, std::int32_t type) noexcept {
    return Out(asNetwork(network)->addActivation(*asTensor(in), static_cast<ActivationType>(type)));
}

void* trtNetworkAddUnary(void* network, void* in, std::int32_t op) noexcept {
    return Out(asNetwork(network)->addUnary(*asTensor(in), static_cast<UnaryOperation>(op)));
}

void* trtNetworkAddMatrixMultiply(
    void* network, void* a, std::int32_t op_a, void* b, std::int32_t op_b) noexcept {
    return Out(asNetwork(network)->addMatrixMultiply(
        *asTensor(a), static_cast<MatrixOperation>(op_a), *asTensor(b), static_cast<MatrixOperation>(op_b)));
}

void* trtNetworkAddConstant(void* network, const Dims* dims, std::int32_t type, const void* values,
                            std::int64_t count) noexcept {
    Weights w{static_cast<DataType>(type), values, count};
    return Out(asNetwork(network)->addConstant(*dims, w));
}

void* trtNetworkAddShuffle(void* network, void* in) noexcept {
    return Out(asNetwork(network)->addShuffle(*asTensor(in)));
}

void* trtNetworkAddReduce(void* network, void* in, std::int32_t op, std::uint32_t axes,
                          bool keep_dims) noexcept {
    return Out(asNetwork(network)->addReduce(*asTensor(in), static_cast<ReduceOperation>(op), axes, keep_dims));
}

void* trtNetworkAddSoftMax(void* network, void* in) noexcept {
    return Out(asNetwork(network)->addSoftMax(*asTensor(in)));
}

void* trtNetworkAddConcatenation(void* network, void* const* inputs, std::int32_t n) noexcept {
    // `ITensor* const*` and `void* const*` have the same representation; copy
    // into a typed array rather than rely on that.
    ITensor* tensors[64];
    if (n < 0 || n > 64) {
        return nullptr;
    }
    for (std::int32_t i = 0; i < n; ++i) {
        tensors[i] = asTensor(inputs[i]);
    }
    return Out(asNetwork(network)->addConcatenation(tensors, n));
}

void* trtNetworkAddCast(void* network, void* in, std::int32_t type) noexcept {
    return Out(asNetwork(network)->addCast(*asTensor(in), static_cast<DataType>(type)));
}

void* trtNetworkAddIdentity(void* network, void* in) noexcept {
    return Out(asNetwork(network)->addIdentity(*asTensor(in)));
}

void* trtNetworkAddConvolutionNd(void* network, void* in, std::int64_t nbOutputMaps,
                                 const Dims* kernelSize, const void* kernel, std::int64_t kernelCount,
                                 const void* bias, std::int64_t biasCount) noexcept {
    Weights k{DataType::kFLOAT, kernel, kernelCount};
    Weights b{DataType::kFLOAT, bias, biasCount};
    return Out(asNetwork(network)->addConvolutionNd(*asTensor(in), nbOutputMaps, *kernelSize, k, b));
}

void* trtNetworkAddPoolingNd(void* network, void* in, std::int32_t type, const Dims* windowSize) noexcept {
    return Out(asNetwork(network)->addPoolingNd(*asTensor(in), static_cast<PoolingType>(type), *windowSize));
}

void* trtNetworkAddDeconvolutionNd(void* network, void* in, std::int64_t nbOutputMaps,
                                   const Dims* kernelSize, const void* kernel, std::int64_t kernelCount,
                                   const void* bias, std::int64_t biasCount) noexcept {
    Weights k{DataType::kFLOAT, kernel, kernelCount};
    Weights b{DataType::kFLOAT, bias, biasCount};
    return Out(asNetwork(network)->addDeconvolutionNd(*asTensor(in), nbOutputMaps, *kernelSize, k, b));
}

void* trtNetworkAddSlice(void* network, void* in, const Dims* start, const Dims* size,
                         const Dims* stride) noexcept {
    return Out(asNetwork(network)->addSlice(*asTensor(in), *start, *size, *stride));
}

void* trtNetworkAddGather(void* network, void* data, void* indices, std::int32_t axis) noexcept {
    return Out(asNetwork(network)->addGather(*asTensor(data), *asTensor(indices), axis));
}

// ---- ILayer / ITensor ---------------------------------------------------

void* trtLayerGetOutput(void* layer, std::int32_t index) noexcept { return asLayer(layer)->getOutput(index); }
std::int32_t trtLayerGetNbOutputs(void* layer) noexcept { return asLayer(layer)->getNbOutputs(); }
void trtLayerSetName(void* layer, const char* name) noexcept { asLayer(layer)->setName(name); }

void trtTensorSetName(void* tensor, const char* name) noexcept { asTensor(tensor)->setName(name); }
const char* trtTensorGetName(void* tensor) noexcept { return asTensor(tensor)->getName(); }
Dims trtTensorGetDimensions(void* tensor) noexcept { return asTensor(tensor)->getDimensions(); }
std::int32_t trtTensorGetType(void* tensor) noexcept {
    return static_cast<std::int32_t>(asTensor(tensor)->getType());
}

// ---- layer-specific setters --------------------------------------------

void trtActivationSetAlpha(void* layer, float alpha) noexcept {
    static_cast<IActivationLayer*>(asLayer(layer))->setAlpha(alpha);
}

void trtActivationSetBeta(void* layer, float beta) noexcept {
    static_cast<IActivationLayer*>(asLayer(layer))->setBeta(beta);
}

void trtShuffleSetReshapeDims(void* layer, const Dims* dims) noexcept {
    static_cast<IShuffleLayer*>(asLayer(layer))->setReshapeDimensions(*dims);
}

void trtShuffleSetFirstTranspose(void* layer, const std::int32_t* order) noexcept {
    Permutation p{};
    for (int i = 0; i < Dims::MAX_DIMS; ++i) {
        p.order[i] = order[i];
    }
    static_cast<IShuffleLayer*>(asLayer(layer))->setFirstTranspose(p);
}

void trtShuffleSetSecondTranspose(void* layer, const std::int32_t* order) noexcept {
    Permutation p{};
    for (int i = 0; i < Dims::MAX_DIMS; ++i) {
        p.order[i] = order[i];
    }
    static_cast<IShuffleLayer*>(asLayer(layer))->setSecondTranspose(p);
}

// ---- network inputs / outputs by index ----------------------------------

void* trtNetworkGetInput(void* network, std::int32_t index) noexcept {
    return asNetwork(network)->getInput(index);
}

void* trtNetworkGetOutput(void* network, std::int32_t index) noexcept {
    return asNetwork(network)->getOutput(index);
}

// ---- ONNX parser (nvonnxparser::IParser) --------------------------------
//
// Created from Rust through `createNvOnnxParser_INTERNAL` in
// nvonnxparser_10.dll / libnvonnxparser.so.10; every call here is a vtable
// dispatch, so this object references no parser symbol either.

static inline nvonnxparser::IParser* asParser(void* p) noexcept {
    return static_cast<nvonnxparser::IParser*>(p);
}

void trtOnnxParserDestroy(void* parser) noexcept { delete asParser(parser); }

bool trtOnnxParserParse(void* parser, const void* data, std::size_t size, const char* path) noexcept {
    return asParser(parser)->parse(data, size, path);
}

std::int32_t trtOnnxParserGetNbErrors(void* parser) noexcept { return asParser(parser)->getNbErrors(); }
void trtOnnxParserClearErrors(void* parser) noexcept { asParser(parser)->clearErrors(); }

bool trtOnnxParserSupportsOperator(void* parser, const char* op) noexcept {
    return asParser(parser)->supportsOperator(op);
}

// Error `index` fields. Strings are owned by the parser and valid until the
// next parse or clearErrors.
std::int32_t trtOnnxParserErrorCode(void* parser, std::int32_t index) noexcept {
    return static_cast<std::int32_t>(asParser(parser)->getError(index)->code());
}
const char* trtOnnxParserErrorDesc(void* parser, std::int32_t index) noexcept {
    return asParser(parser)->getError(index)->desc();
}
std::int32_t trtOnnxParserErrorNode(void* parser, std::int32_t index) noexcept {
    return asParser(parser)->getError(index)->node();
}
const char* trtOnnxParserErrorNodeName(void* parser, std::int32_t index) noexcept {
    return asParser(parser)->getError(index)->nodeName();
}
const char* trtOnnxParserErrorNodeOperator(void* parser, std::int32_t index) noexcept {
    return asParser(parser)->getError(index)->nodeOperator();
}

// ---- optimization profiles ----------------------------------------------

// The profile is owned by the builder that created it: never delete it.
void* trtBuilderCreateOptimizationProfile(void* builder) noexcept {
    return asBuilder(builder)->createOptimizationProfile();
}

bool trtProfileSetDimensions(void* profile, const char* name, std::int32_t selector,
                             const Dims* dims) noexcept {
    return static_cast<IOptimizationProfile*>(profile)->setDimensions(
        name, static_cast<OptProfileSelector>(selector), *dims);
}

Dims trtProfileGetDimensions(void* profile, const char* name, std::int32_t selector) noexcept {
    return static_cast<IOptimizationProfile*>(profile)->getDimensions(
        name, static_cast<OptProfileSelector>(selector));
}

bool trtProfileIsValid(void* profile) noexcept {
    return static_cast<IOptimizationProfile*>(profile)->isValid();
}

std::int32_t trtConfigAddOptimizationProfile(void* config, void* profile) noexcept {
    return asConfig(config)->addOptimizationProfile(static_cast<IOptimizationProfile*>(profile));
}

std::int32_t trtConfigGetNbOptimizationProfiles(void* config) noexcept {
    return asConfig(config)->getNbOptimizationProfiles();
}

Dims trtCudaEngineGetProfileShape(void* engine, const char* name, std::int32_t profile,
                                  std::int32_t selector) noexcept {
    return static_cast<ICudaEngine*>(engine)->getProfileShape(
        name, profile, static_cast<OptProfileSelector>(selector));
}

bool trtExecutionContextSetOptimizationProfileAsync(void* context, std::int32_t profile,
                                                    void* stream) noexcept {
    return static_cast<IExecutionContext*>(context)->setOptimizationProfileAsync(
        profile, reinterpret_cast<cudaStream_t>(stream));
}

std::int32_t trtExecutionContextGetOptimizationProfile(void* context) noexcept {
    return static_cast<IExecutionContext*>(context)->getOptimizationProfile();
}

void trtConcatenationSetAxis(void* layer, std::int32_t axis) noexcept {
    static_cast<IConcatenationLayer*>(asLayer(layer))->setAxis(axis);
}

void trtSoftMaxSetAxes(void* layer, std::uint32_t axes) noexcept {
    static_cast<ISoftMaxLayer*>(asLayer(layer))->setAxes(axes);
}

void trtConvolutionSetStrideNd(void* layer, const Dims* d) noexcept {
    static_cast<IConvolutionLayer*>(asLayer(layer))->setStrideNd(*d);
}
void trtConvolutionSetPaddingNd(void* layer, const Dims* d) noexcept {
    static_cast<IConvolutionLayer*>(asLayer(layer))->setPaddingNd(*d);
}
void trtConvolutionSetDilationNd(void* layer, const Dims* d) noexcept {
    static_cast<IConvolutionLayer*>(asLayer(layer))->setDilationNd(*d);
}
void trtConvolutionSetNbGroups(void* layer, std::int64_t n) noexcept {
    static_cast<IConvolutionLayer*>(asLayer(layer))->setNbGroups(n);
}
void trtPoolingSetStrideNd(void* layer, const Dims* d) noexcept {
    static_cast<IPoolingLayer*>(asLayer(layer))->setStrideNd(*d);
}
void trtPoolingSetPaddingNd(void* layer, const Dims* d) noexcept {
    static_cast<IPoolingLayer*>(asLayer(layer))->setPaddingNd(*d);
}

void trtDeconvolutionSetStrideNd(void* layer, const Dims* d) noexcept {
    static_cast<IDeconvolutionLayer*>(asLayer(layer))->setStrideNd(*d);
}
void trtDeconvolutionSetPaddingNd(void* layer, const Dims* d) noexcept {
    static_cast<IDeconvolutionLayer*>(asLayer(layer))->setPaddingNd(*d);
}
void trtDeconvolutionSetDilationNd(void* layer, const Dims* d) noexcept {
    static_cast<IDeconvolutionLayer*>(asLayer(layer))->setDilationNd(*d);
}
void trtDeconvolutionSetNbGroups(void* layer, std::int64_t n) noexcept {
    static_cast<IDeconvolutionLayer*>(asLayer(layer))->setNbGroups(n);
}

}  // extern "C"

// ---- plugin registry (nvinfer1::IPluginRegistry) -------------------------
//
// The registry is a process-wide singleton reached through libnvinfer's exported
// `getPluginRegistry`, which the Rust loader resolves; every call here is a
// vtable dispatch. Creators come back as `IPluginCreatorInterface*`; the concrete
// creator type is read from `getInterfaceInfo()` (no RTTI crosses the boundary).

namespace {

enum PluginCreatorKind : std::int32_t { kUnknown = 0, kV1 = 1, kV3One = 2 };

PluginCreatorKind creator_kind(void* creator) noexcept {
    const char* kind = static_cast<IPluginCreatorInterface*>(creator)->getInterfaceInfo().kind;
    if (kind == nullptr) {
        return kUnknown;
    }
    if (std::strcmp(kind, "PLUGIN CREATOR_V1") == 0) {
        return kV1;
    }
    if (std::strcmp(kind, "PLUGIN CREATOR_V3ONE") == 0) {
        return kV3One;
    }
    return kUnknown;
}

}  // namespace

extern "C" {

std::int32_t trtPluginCreatorKind(void* creator) noexcept { return creator_kind(creator); }

// The three accessors below return null for a creator kind this shim cannot read.
const char* trtPluginCreatorName(void* creator) noexcept {
    switch (creator_kind(creator)) {
        case kV1: return static_cast<IPluginCreator*>(creator)->getPluginName();
        case kV3One: return static_cast<IPluginCreatorV3One*>(creator)->getPluginName();
        default: return nullptr;
    }
}

const char* trtPluginCreatorVersion(void* creator) noexcept {
    switch (creator_kind(creator)) {
        case kV1: return static_cast<IPluginCreator*>(creator)->getPluginVersion();
        case kV3One: return static_cast<IPluginCreatorV3One*>(creator)->getPluginVersion();
        default: return nullptr;
    }
}

const char* trtPluginCreatorNamespace(void* creator) noexcept {
    switch (creator_kind(creator)) {
        case kV1: return static_cast<IPluginCreator*>(creator)->getPluginNamespace();
        case kV3One: return static_cast<IPluginCreatorV3One*>(creator)->getPluginNamespace();
        default: return nullptr;
    }
}

// Returns a pointer to the registry's array of creators and writes the count.
void* const* trtPluginRegistryGetAllCreators(void* registry, std::int32_t* count) noexcept {
    return reinterpret_cast<void* const*>(
        static_cast<IPluginRegistry*>(registry)->getAllCreators(count));
}

void* trtPluginRegistryGetCreator(
    void* registry, const char* name, const char* version, const char* ns) noexcept {
    return static_cast<IPluginRegistry*>(registry)->getCreator(name, version, ns);
}

// Returns an opaque library handle, or null if the library could not be loaded.
void* trtPluginRegistryLoadLibrary(void* registry, const char* path) noexcept {
    return static_cast<IPluginRegistry*>(registry)->loadLibrary(path);
}

void trtPluginRegistryDeregisterLibrary(void* registry, void* handle) noexcept {
    static_cast<IPluginRegistry*>(registry)->deregisterLibrary(handle);
}

bool trtPluginRegistryDeregisterCreator(void* registry, void* creator) noexcept {
    return static_cast<IPluginRegistry*>(registry)->deregisterCreator(
        *static_cast<IPluginCreatorInterface*>(creator));
}

}  // extern "C"
