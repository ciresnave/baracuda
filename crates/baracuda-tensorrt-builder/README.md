# baracuda-tensorrt-builder

Safe Rust API for the NVIDIA TensorRT **builder**: define a network layer by layer, configure the build, and get a serialized engine plan that [`baracuda-tensorrt`](../baracuda-tensorrt) loads and runs.

```rust,ignore
let logger = Arc::new(Logger::stderr(Severity::Warning)?);
let builder = Builder::new(logger)?;
let network = builder.create_network(NetworkFlags::STRONGLY_TYPED)?;
let a = network.add_input("a", DataType::Float, &Dims::new(&[8]))?;
let b = network.add_input("b", DataType::Float, &Dims::new(&[8]))?;
let sum = network.add_element_wise(&a, &b, ElementWiseOp::Sum)?.output(0)?;
sum.set_name("sum")?;
network.mark_output(&sum)?;
let plan = builder.build_serialized(&network, &builder.create_config()?)?;
```

The API needs the `shim` feature, and so the TensorRT SDK headers at build time (see [`../baracuda-tensorrt/NOTICE.md`](../baracuda-tensorrt/NOTICE.md)); TensorRT itself is loaded at run time, never linked. Without `shim` the crate is empty apart from `shim_built()`.

Covered so far: builder, config (memory-pool limits, flags, optimization level), optimization profiles, the ONNX parser (`OnnxParser`, which needs `nvonnxparser_10.dll` / `libnvonnxparser.so.10` next to TensorRT), and networks with elementwise, activation, unary, matrix multiply, constant, shuffle, reduce, softmax, concatenation, cast and identity layers. Still to come: INT8 calibration, the plugin registry, convolution and pooling.

Dual-licensed under MIT or Apache-2.0.
