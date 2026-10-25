use std::cell::RefCell;
use std::ffi::CString;
use std::marker::PhantomData;
use std::sync::Arc;

use baracuda_tensorrt::{DataType, Dims, Error, HostMemory, Logger, Result};
use baracuda_tensorrt_sys::{self as sys, builder as ffi};

/// Define a fieldless `#[repr(i32)]` enum whose discriminants are TensorRT's.
macro_rules! trt_enum {
    ($(#[$meta:meta])* $name:ident { $($(#[$vmeta:meta])* $variant:ident = $val:expr),* $(,)? }) => {
        $(#[$meta])*
        #[repr(i32)]
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
        pub enum $name {
            $($(#[$vmeta])* $variant = $val,)*
        }
    };
}

trt_enum! {
    /// `nvinfer1::ElementWiseOperation`.
    ElementWiseOp {
        /// `a + b`.
        Sum = 0,
        /// `a * b`.
        Prod = 1,
        /// `max(a, b)`.
        Max = 2,
        /// `min(a, b)`.
        Min = 3,
        /// `a - b`.
        Sub = 4,
        /// `a / b`.
        Div = 5,
        /// `a ^ b`.
        Pow = 6,
        /// Floor division.
        FloorDiv = 7,
        /// Logical and.
        And = 8,
        /// Logical or.
        Or = 9,
        /// Logical xor.
        Xor = 10,
        /// `a == b`.
        Equal = 11,
        /// `a > b`.
        Greater = 12,
        /// `a < b`.
        Less = 13,
    }
}

trt_enum! {
    /// `nvinfer1::ActivationType`.
    Activation {
        /// `max(0, x)`.
        Relu = 0,
        /// Logistic sigmoid.
        Sigmoid = 1,
        /// `tanh`.
        Tanh = 2,
        /// `x >= 0 ? x : alpha * x`.
        LeakyRelu = 3,
        /// `x >= 0 ? x : alpha * (exp(x) - 1)`.
        Elu = 4,
        /// Selu.
        Selu = 5,
        /// `x / (1 + |x|)`.
        Softsign = 6,
        /// `alpha * log(exp(beta * x) + 1)`.
        Softplus = 7,
        /// `max(alpha, min(beta, x))`.
        Clip = 8,
        /// `max(0, min(1, alpha * x + beta))`.
        HardSigmoid = 9,
        /// `alpha * tanh(beta * x)`.
        ScaledTanh = 10,
        /// `x > alpha ? x : 0`.
        ThresholdedRelu = 11,
        /// Exact (erf) GELU.
        GeluErf = 12,
        /// Tanh-approximate GELU.
        GeluTanh = 13,
    }
}

trt_enum! {
    /// `nvinfer1::UnaryOperation`.
    UnaryOp {
        /// `exp`.
        Exp = 0,
        /// Natural log.
        Log = 1,
        /// Square root.
        Sqrt = 2,
        /// Reciprocal.
        Recip = 3,
        /// Absolute value.
        Abs = 4,
        /// Negation.
        Neg = 5,
        /// Sine.
        Sin = 6,
        /// Cosine.
        Cos = 7,
        /// Tangent.
        Tan = 8,
        /// Hyperbolic sine.
        Sinh = 9,
        /// Hyperbolic cosine.
        Cosh = 10,
        /// Inverse sine.
        Asin = 11,
        /// Inverse cosine.
        Acos = 12,
        /// Inverse tangent.
        Atan = 13,
        /// Inverse hyperbolic sine.
        Asinh = 14,
        /// Inverse hyperbolic cosine.
        Acosh = 15,
        /// Inverse hyperbolic tangent.
        Atanh = 16,
        /// Ceiling.
        Ceil = 17,
        /// Floor.
        Floor = 18,
        /// Error function.
        Erf = 19,
        /// Logical not.
        Not = 20,
    }
}

trt_enum! {
    /// `nvinfer1::ReduceOperation`.
    ReduceOp {
        /// Sum.
        Sum = 0,
        /// Product.
        Prod = 1,
        /// Maximum.
        Max = 2,
        /// Minimum.
        Min = 3,
        /// Mean.
        Avg = 4,
    }
}

trt_enum! {
    /// `nvinfer1::MatrixOperation`.
    MatrixOp {
        /// Use the operand as given.
        None = 0,
        /// Transpose the last two axes first.
        Transpose = 1,
        /// Treat the operand as a vector.
        Vector = 2,
    }
}

trt_enum! {
    /// `nvinfer1::MemoryPoolType`.
    MemoryPool {
        /// Scratch memory a layer may use while it runs.
        Workspace = 0,
        /// DLA managed SRAM.
        DlaManagedSram = 1,
        /// DLA local DRAM.
        DlaLocalDram = 2,
        /// DLA global DRAM.
        DlaGlobalDram = 3,
        /// Host memory the builder may use for tactics' timing caches.
        TacticSharedMemory = 5,
    }
}

trt_enum! {
    /// The `nvinfer1::BuilderFlag` values this crate exposes.
    BuilderFlag {
        /// Allow FP16 tactics (deprecated in TensorRT 10 in favour of a
        /// strongly typed network, but still honoured for weakly typed ones).
        Fp16 = 0,
        /// Allow INT8 tactics (see the calibration row of the roadmap).
        Int8 = 1,
        /// Add debug-tensor outputs.
        Debug = 2,
        /// Let layers fall back to the GPU when DLA cannot run them.
        GpuFallback = 3,
    }
}

/// `nvinfer1::NetworkDefinitionCreationFlag`s, as a bit set.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub struct NetworkFlags(pub u32);

impl NetworkFlags {
    /// No flags. TensorRT 10 networks are always explicit-batch.
    pub const NONE: Self = Self(0);
    /// Every tensor carries its own type; precision is not inferred.
    pub const STRONGLY_TYPED: Self = Self(1 << 1);
}

impl std::ops::BitOr for NetworkFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

fn nul_terminated(name: &str) -> Result<CString> {
    Ok(CString::new(name)?)
}

fn non_null<T>(ptr: *mut T, op: &'static str) -> Result<*mut T> {
    if ptr.is_null() {
        Err(Error::NullHandle { op })
    } else {
        Ok(ptr)
    }
}

/// An `IBuilder`: creates networks and configs and builds engines.
#[derive(Debug)]
pub struct Builder {
    raw: sys::trtIBuilder_t,
    // TensorRT keeps the logger pointer; it must outlive the builder and everything made from it.
    _logger: Arc<Logger>,
}

impl Builder {
    /// Create a builder that reports through `logger`.
    pub fn new(logger: Arc<Logger>) -> Result<Self> {
        // SAFETY: both functions were resolved from libnvinfer by the loader; the logger is the process-wide `ILogger*`, and the version is the library's own.
        let raw = unsafe {
            let t = sys::tensorrt()?;
            let version = (t.get_infer_lib_version()?)();
            (t.create_infer_builder()?)(logger.as_raw(), version)
        };
        Ok(Self {
            raw: non_null(raw, "createInferBuilder")?,
            _logger: logger,
        })
    }

    /// Create an empty network definition.
    pub fn create_network(&self, flags: NetworkFlags) -> Result<Network<'_>> {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        let raw = unsafe { ffi::trtBuilderCreateNetwork(self.raw, flags.0) };
        Ok(Network {
            raw: non_null(raw, "createNetworkV2")?,
            weights: RefCell::new(Vec::new()),
            _builder: PhantomData,
        })
    }

    /// Create a build configuration.
    pub fn create_config(&self) -> Result<BuilderConfig<'_>> {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        let raw = unsafe { ffi::trtBuilderCreateConfig(self.raw) };
        Ok(BuilderConfig {
            raw: non_null(raw, "createBuilderConfig")?,
            _builder: PhantomData,
        })
    }

    /// Build `network` under `config` into a serialized engine plan.
    ///
    /// Fails with [`Error::NullHandle`] when TensorRT rejects the network
    /// (the logger receives the reason).
    pub fn build_serialized(
        &self,
        network: &Network<'_>,
        config: &BuilderConfig<'_>,
    ) -> Result<HostMemory> {
        // SAFETY: `self.raw`, `network.raw` and `config.raw` are live handles (the network and config
        // borrow this builder); TensorRT reads them for the call and returns a new `IHostMemory*` or null.
        let raw = unsafe { ffi::trtBuilderBuildSerialized(self.raw, network.raw, config.raw) };
        let raw = non_null(raw, "buildSerializedNetwork")?;
        // SAFETY: TensorRT returned a fresh `IHostMemory*` that only this value owns.
        Ok(unsafe { HostMemory::from_raw(raw) })
    }

    /// Raw `IBuilder*`.
    pub fn as_raw(&self) -> sys::trtIBuilder_t {
        self.raw
    }
}

impl Drop for Builder {
    fn drop(&mut self) {
        // SAFETY: `self.raw` came from the matching create call, is destroyed exactly once here, and nothing can still borrow it (children borrow `&self`).
        unsafe { ffi::trtBuilderDestroy(self.raw) };
    }
}

/// An `IBuilderConfig`.
#[derive(Debug)]
pub struct BuilderConfig<'b> {
    raw: sys::trtIBuilderConfig_t,
    _builder: PhantomData<&'b Builder>,
}

impl BuilderConfig<'_> {
    /// Cap `pool` at `bytes`.
    pub fn set_memory_pool_limit(&self, pool: MemoryPool, bytes: usize) {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        unsafe { ffi::trtConfigSetMemoryPoolLimit(self.raw, pool as i32, bytes) };
    }

    /// The current cap on `pool`, in bytes.
    pub fn memory_pool_limit(&self, pool: MemoryPool) -> usize {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        unsafe { ffi::trtConfigGetMemoryPoolLimit(self.raw, pool as i32) }
    }

    /// Turn a builder flag on or off.
    pub fn set_flag(&self, flag: BuilderFlag, on: bool) {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        unsafe { ffi::trtConfigSetFlag(self.raw, flag as i32, on) };
    }

    /// Whether a builder flag is on.
    pub fn flag(&self, flag: BuilderFlag) -> bool {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        unsafe { ffi::trtConfigGetFlag(self.raw, flag as i32) }
    }

    /// Set the builder optimization level (0..=5; higher searches more tactics).
    pub fn set_optimization_level(&self, level: i32) {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        unsafe { ffi::trtConfigSetOptimizationLevel(self.raw, level) };
    }

    /// The builder optimization level.
    pub fn optimization_level(&self) -> i32 {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        unsafe { ffi::trtConfigGetOptimizationLevel(self.raw) }
    }

    /// Raw `IBuilderConfig*`.
    pub fn as_raw(&self) -> sys::trtIBuilderConfig_t {
        self.raw
    }
}

impl Drop for BuilderConfig<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.raw` came from the matching create call, is destroyed exactly once here, and nothing can still borrow it (children borrow `&self`).
        unsafe { ffi::trtConfigDestroy(self.raw) };
    }
}

/// An `INetworkDefinition`.
///
/// Layers and tensors borrow the network. Constant weights are kept alive here
/// until the network is dropped, because TensorRT stores the pointer and reads
/// the bytes at build time.
#[derive(Debug)]
pub struct Network<'b> {
    raw: sys::trtINetworkDefinition_t,
    weights: RefCell<Vec<Box<[u8]>>>,
    _builder: PhantomData<&'b Builder>,
}

impl Network<'_> {
    fn tensor<'a>(&'a self, raw: sys::trtITensor_t, op: &'static str) -> Result<Tensor<'a>> {
        Ok(Tensor {
            raw: non_null(raw, op)?,
            _net: PhantomData,
        })
    }

    fn layer<'a>(&'a self, raw: sys::trtILayer_t, op: &'static str) -> Result<Layer<'a>> {
        Ok(Layer {
            raw: non_null(raw, op)?,
            _net: PhantomData,
        })
    }

    /// Add a network input.
    pub fn add_input<'a>(&'a self, name: &str, dtype: DataType, dims: &Dims) -> Result<Tensor<'a>> {
        let name = nul_terminated(name)?;
        let dims = dims.to_raw();
        // SAFETY: the handle is live (owned by `self`); each pointer argument is a NUL-terminated `CString` or a local that outlives the call.
        let raw = unsafe { ffi::trtNetworkAddInput(self.raw, name.as_ptr(), dtype as i32, &dims) };
        self.tensor(raw, "addInput")
    }

    /// Mark `tensor` as a network output.
    pub fn mark_output<'a>(&'a self, tensor: &Tensor<'a>) -> Result<()> {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        unsafe { ffi::trtNetworkMarkOutput(self.raw, tensor.raw) };
        Ok(())
    }

    /// Number of network inputs.
    pub fn num_inputs(&self) -> i32 {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        unsafe { ffi::trtNetworkGetNbInputs(self.raw) }
    }

    /// Number of network outputs.
    pub fn num_outputs(&self) -> i32 {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        unsafe { ffi::trtNetworkGetNbOutputs(self.raw) }
    }

    /// Number of layers.
    pub fn num_layers(&self) -> i32 {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        unsafe { ffi::trtNetworkGetNbLayers(self.raw) }
    }

    /// `a <op> b`, elementwise with broadcasting.
    pub fn add_element_wise<'a>(
        &'a self,
        a: &Tensor<'a>,
        b: &Tensor<'a>,
        op: ElementWiseOp,
    ) -> Result<Layer<'a>> {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        let raw = unsafe { ffi::trtNetworkAddElementWise(self.raw, a.raw, b.raw, op as i32) };
        self.layer(raw, "addElementWise")
    }

    /// An activation. `alpha` / `beta` apply to the kinds that take them.
    pub fn add_activation<'a>(
        &'a self,
        input: &Tensor<'a>,
        kind: Activation,
        alpha: Option<f32>,
        beta: Option<f32>,
    ) -> Result<Layer<'a>> {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        let raw = unsafe { ffi::trtNetworkAddActivation(self.raw, input.raw, kind as i32) };
        let layer = self.layer(raw, "addActivation")?;
        if let Some(alpha) = alpha {
            // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
            unsafe { ffi::trtActivationSetAlpha(layer.raw, alpha) };
        }
        if let Some(beta) = beta {
            // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
            unsafe { ffi::trtActivationSetBeta(layer.raw, beta) };
        }
        Ok(layer)
    }

    /// A unary elementwise op.
    pub fn add_unary<'a>(&'a self, input: &Tensor<'a>, op: UnaryOp) -> Result<Layer<'a>> {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        let raw = unsafe { ffi::trtNetworkAddUnary(self.raw, input.raw, op as i32) };
        self.layer(raw, "addUnary")
    }

    /// Batched matrix multiply of the last two axes.
    pub fn add_matrix_multiply<'a>(
        &'a self,
        a: &Tensor<'a>,
        op_a: MatrixOp,
        b: &Tensor<'a>,
        op_b: MatrixOp,
    ) -> Result<Layer<'a>> {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        let raw = unsafe {
            ffi::trtNetworkAddMatrixMultiply(self.raw, a.raw, op_a as i32, b.raw, op_b as i32)
        };
        self.layer(raw, "addMatrixMultiply")
    }

    /// A constant tensor of `dims` holding `values` (`f32`).
    pub fn add_constant_f32<'a>(&'a self, dims: &Dims, values: &[f32]) -> Result<Layer<'a>> {
        let bytes: Box<[u8]> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        self.add_constant_bytes(dims, DataType::Float, bytes, values.len())
    }

    /// A constant tensor of `dims` holding `values` (`i32`).
    pub fn add_constant_i32<'a>(&'a self, dims: &Dims, values: &[i32]) -> Result<Layer<'a>> {
        let bytes: Box<[u8]> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        self.add_constant_bytes(dims, DataType::Int32, bytes, values.len())
    }

    fn add_constant_bytes<'a>(
        &'a self,
        dims: &Dims,
        dtype: DataType,
        bytes: Box<[u8]>,
        count: usize,
    ) -> Result<Layer<'a>> {
        let elements: i64 = dims.as_slice().iter().product();
        if elements != count as i64 {
            return Err(Error::Call {
                op: "addConstant: value count does not match the dimensions",
            });
        }
        let ptr = bytes.as_ptr().cast();
        // The heap block does not move when the box moves into the Vec.
        self.weights.borrow_mut().push(bytes);
        let dims = dims.to_raw();
        let raw =
            // SAFETY: `ptr` points into the boxed buffer pushed onto `self.weights` just above, which lives until the network is dropped; `dims` is a local that outlives the call.
            unsafe { ffi::trtNetworkAddConstant(self.raw, &dims, dtype as i32, ptr, count as i64) };
        self.layer(raw, "addConstant")
    }

    /// A shuffle layer; configure it with [`Layer::set_reshape_dims`] and the
    /// transpose setters.
    pub fn add_shuffle<'a>(&'a self, input: &Tensor<'a>) -> Result<Layer<'a>> {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        let raw = unsafe { ffi::trtNetworkAddShuffle(self.raw, input.raw) };
        self.layer(raw, "addShuffle")
    }

    /// Reduce over the axes whose bits are set in `axes`.
    pub fn add_reduce<'a>(
        &'a self,
        input: &Tensor<'a>,
        op: ReduceOp,
        axes: u32,
        keep_dims: bool,
    ) -> Result<Layer<'a>> {
        let raw =
            // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
            unsafe { ffi::trtNetworkAddReduce(self.raw, input.raw, op as i32, axes, keep_dims) };
        self.layer(raw, "addReduce")
    }

    /// Softmax over the axes whose bits are set in `axes`.
    pub fn add_softmax<'a>(&'a self, input: &Tensor<'a>, axes: u32) -> Result<Layer<'a>> {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        let raw = unsafe { ffi::trtNetworkAddSoftMax(self.raw, input.raw) };
        let layer = self.layer(raw, "addSoftMax")?;
        unsafe { ffi::trtSoftMaxSetAxes(layer.raw, axes) };
        Ok(layer)
    }

    /// Concatenate `inputs` (at most 64) along an axis chosen with
    /// [`Layer::set_concat_axis`] (the default is axis 0).
    pub fn add_concatenation<'a>(&'a self, inputs: &[&Tensor<'a>]) -> Result<Layer<'a>> {
        if inputs.is_empty() || inputs.len() > 64 {
            return Err(Error::Call {
                op: "addConcatenation: need 1..=64 inputs",
            });
        }
        let raws: Vec<sys::trtITensor_t> = inputs.iter().map(|t| t.raw).collect();
        let raw =
            // SAFETY: `raws` holds `raws.len()` live tensor handles of this network and is valid for the call.
            unsafe { ffi::trtNetworkAddConcatenation(self.raw, raws.as_ptr(), raws.len() as i32) };
        self.layer(raw, "addConcatenation")
    }

    /// Convert to `dtype`.
    pub fn add_cast<'a>(&'a self, input: &Tensor<'a>, dtype: DataType) -> Result<Layer<'a>> {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        let raw = unsafe { ffi::trtNetworkAddCast(self.raw, input.raw, dtype as i32) };
        self.layer(raw, "addCast")
    }

    /// An identity layer.
    pub fn add_identity<'a>(&'a self, input: &Tensor<'a>) -> Result<Layer<'a>> {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        let raw = unsafe { ffi::trtNetworkAddIdentity(self.raw, input.raw) };
        self.layer(raw, "addIdentity")
    }

    /// Raw `INetworkDefinition*`.
    pub fn as_raw(&self) -> sys::trtINetworkDefinition_t {
        self.raw
    }
}

impl Drop for Network<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.raw` came from the matching create call, is destroyed exactly once here, and nothing can still borrow it (children borrow `&self`).
        unsafe { ffi::trtNetworkDestroy(self.raw) };
    }
}

/// A tensor inside a [`Network`].
#[derive(Debug, Copy, Clone)]
pub struct Tensor<'n> {
    raw: sys::trtITensor_t,
    _net: PhantomData<&'n ()>,
}

impl Tensor<'_> {
    /// Name the tensor (network inputs and outputs are bound by name at run time).
    pub fn set_name(&self, name: &str) -> Result<()> {
        let name = nul_terminated(name)?;
        // SAFETY: the handle is live (owned by `self`); each pointer argument is a NUL-terminated `CString` or a local that outlives the call.
        unsafe { ffi::trtTensorSetName(self.raw, name.as_ptr()) };
        Ok(())
    }

    /// The tensor's name.
    pub fn name(&self) -> String {
        // SAFETY: TensorRT returns a NUL-terminated string that stays valid while the tensor lives (a null pointer is handled above).
        let p = unsafe { ffi::trtTensorGetName(self.raw) };
        if p.is_null() {
            return String::new();
        }
        // SAFETY: TensorRT returns a NUL-terminated string that stays valid while the tensor lives (a null pointer is handled above).
        unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned()
    }

    /// The tensor's dimensions (`-1` marks a dynamic axis).
    pub fn dimensions(&self) -> Dims {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        Dims::from_raw(unsafe { ffi::trtTensorGetDimensions(self.raw) })
    }

    /// The tensor's element type, or `None` if TensorRT reports a type this
    /// crate does not know.
    pub fn data_type(&self) -> Option<DataType> {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        Some(match unsafe { ffi::trtTensorGetType(self.raw) } {
            0 => DataType::Float,
            1 => DataType::Half,
            2 => DataType::Int8,
            3 => DataType::Int32,
            4 => DataType::Bool,
            5 => DataType::Uint8,
            6 => DataType::Fp8,
            7 => DataType::BFloat16,
            8 => DataType::Int64,
            9 => DataType::Int4,
            10 => DataType::Fp4,
            _ => return None,
        })
    }

    /// Raw `ITensor*`.
    pub fn as_raw(&self) -> sys::trtITensor_t {
        self.raw
    }
}

/// A layer inside a [`Network`].
#[derive(Debug, Copy, Clone)]
pub struct Layer<'n> {
    raw: sys::trtILayer_t,
    _net: PhantomData<&'n ()>,
}

impl<'n> Layer<'n> {
    /// Number of output tensors.
    pub fn num_outputs(&self) -> i32 {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        unsafe { ffi::trtLayerGetNbOutputs(self.raw) }
    }

    /// Output tensor `index`.
    pub fn output(&self, index: i32) -> Result<Tensor<'n>> {
        // SAFETY: `self.raw` and every tensor / layer / profile handle passed are live, non-null handles of the same network or builder; the borrow checker ties their lifetimes to `self`.
        let raw = unsafe { ffi::trtLayerGetOutput(self.raw, index) };
        Ok(Tensor {
            raw: non_null(raw, "ILayer::getOutput")?,
            _net: PhantomData,
        })
    }

    /// Name the layer (shows up in TensorRT's logs and profiles).
    pub fn set_name(&self, name: &str) -> Result<()> {
        let name = nul_terminated(name)?;
        // SAFETY: the handle is live (owned by `self`); each pointer argument is a NUL-terminated `CString` or a local that outlives the call.
        unsafe { ffi::trtLayerSetName(self.raw, name.as_ptr()) };
        Ok(())
    }

    /// For a concatenation layer: the axis to concatenate along.
    pub fn set_concat_axis(&self, axis: i32) {
        // SAFETY: the handle is live (owned by `self`); each pointer argument is a NUL-terminated `CString` or a local that outlives the call.
        unsafe { ffi::trtConcatenationSetAxis(self.raw, axis) };
    }

    /// For a shuffle layer: reshape to `dims` (`-1` infers one axis).
    pub fn set_reshape_dims(&self, dims: &Dims) {
        let dims = dims.to_raw();
        // SAFETY: the handle is live (owned by `self`); each pointer argument is a NUL-terminated `CString` or a local that outlives the call.
        unsafe { ffi::trtShuffleSetReshapeDims(self.raw, &dims) };
    }

    /// For a shuffle layer: permute axes before the reshape. `order` lists the
    /// source axis for each output axis; missing entries default to identity.
    pub fn set_first_transpose(&self, order: &[i32]) {
        let order = permutation(order);
        // SAFETY: the handle is live (owned by `self`); each pointer argument is a NUL-terminated `CString` or a local that outlives the call.
        unsafe { ffi::trtShuffleSetFirstTranspose(self.raw, order.as_ptr()) };
    }

    /// For a shuffle layer: permute axes after the reshape.
    pub fn set_second_transpose(&self, order: &[i32]) {
        let order = permutation(order);
        // SAFETY: the handle is live (owned by `self`); each pointer argument is a NUL-terminated `CString` or a local that outlives the call.
        unsafe { ffi::trtShuffleSetSecondTranspose(self.raw, order.as_ptr()) };
    }

    /// Raw `ILayer*`.
    pub fn as_raw(&self) -> sys::trtILayer_t {
        self.raw
    }
}

fn permutation(order: &[i32]) -> [i32; sys::TRT_MAX_DIMS] {
    let mut out = [0i32; sys::TRT_MAX_DIMS];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = order.get(i).copied().unwrap_or(i as i32);
    }
    out
}
