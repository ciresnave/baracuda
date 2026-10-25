//! The ONNX parser: hand-encoded ONNX models are parsed into a network, built,
//! and run on the GPU against a CPU reference.
//!
//! ```text
//! cargo gpu-test -p baracuda-tensorrt-builder --features shim,device-tests
//! ```

#![cfg(all(feature = "shim", feature = "device-tests"))]

use std::sync::Arc;

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, require_optional};
use baracuda_tensorrt::{Logger, Runtime, Severity};
use baracuda_tensorrt_builder::{Builder, MemoryPool, NetworkFlags, OnnxParser};

fn logger() -> Option<Arc<Logger>> {
    Logger::stderr(Severity::Error).ok().map(Arc::new)
}

// ---- a tiny protobuf writer: just enough for ONNX ModelProto --------------

fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn bytes_field(tag: u64, payload: &[u8], out: &mut Vec<u8>) {
    varint((tag << 3) | 2, out);
    varint(payload.len() as u64, out);
    out.extend_from_slice(payload);
}

fn varint_field(tag: u64, value: u64, out: &mut Vec<u8>) {
    varint(tag << 3, out);
    varint(value, out);
}

/// `ValueInfoProto` for a float tensor of the given static shape.
fn value_info(name: &str, shape: &[u64]) -> Vec<u8> {
    let mut dims = Vec::new();
    for &d in shape {
        let mut dim = Vec::new();
        varint_field(1, d, &mut dim); // Dimension.dim_value
        bytes_field(1, &dim, &mut dims); // TensorShapeProto.dim
    }
    let mut tensor = Vec::new();
    varint_field(1, 1, &mut tensor); // elem_type = FLOAT
    bytes_field(2, &dims, &mut tensor); // shape
    let mut type_proto = Vec::new();
    bytes_field(1, &tensor, &mut type_proto); // TypeProto.tensor_type
    let mut info = Vec::new();
    bytes_field(1, name.as_bytes(), &mut info); // name
    bytes_field(2, &type_proto, &mut info); // type
    info
}

/// An ONNX model applying `ops` in order to an input `x` (shape `shape`),
/// producing `y`.
fn model(ops: &[&str], shape: &[u64]) -> Vec<u8> {
    let mut graph = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        let input = if i == 0 {
            "x".to_owned()
        } else {
            format!("t{}", i - 1)
        };
        let output = if i + 1 == ops.len() {
            "y".to_owned()
        } else {
            format!("t{i}")
        };
        let mut node = Vec::new();
        bytes_field(1, input.as_bytes(), &mut node);
        bytes_field(2, output.as_bytes(), &mut node);
        bytes_field(4, op.as_bytes(), &mut node);
        bytes_field(1, &node, &mut graph); // GraphProto.node
    }
    bytes_field(2, b"g", &mut graph); // GraphProto.name
    bytes_field(11, &value_info("x", shape), &mut graph);
    bytes_field(12, &value_info("y", shape), &mut graph);

    let mut opset = Vec::new();
    bytes_field(1, b"", &mut opset);
    varint_field(2, 13, &mut opset);

    let mut model = Vec::new();
    varint_field(1, 8, &mut model); // ir_version
    bytes_field(7, &graph, &mut model); // graph
    bytes_field(8, &opset, &mut model); // opset_import
    model
}

#[test]
fn an_onnx_model_parses_builds_and_runs() {
    let logger = require_optional!(logger(), "C-ABI shim (Logger::new)");
    let builder = Builder::new(Arc::clone(&logger)).expect("builder");
    let network = builder
        .create_network(NetworkFlags::STRONGLY_TYPED)
        .expect("network");
    let parser = OnnxParser::new(&network, Arc::clone(&logger)).expect("parser");
    // relu(-x): both signs matter.
    parser
        .parse(&model(&["Neg", "Relu"], &[2, 3]), None)
        .expect("parse");
    assert_eq!((network.num_inputs(), network.num_outputs()), (1, 1));
    assert_eq!(network.input(0).unwrap().name(), "x");
    assert_eq!(network.output(0).unwrap().name(), "y");
    assert_eq!(network.input(0).unwrap().dimensions().as_slice(), &[2, 3]);

    let config = builder.create_config().expect("config");
    config.set_memory_pool_limit(MemoryPool::Workspace, 64 << 20);
    let plan = builder.build_serialized(&network, &config).expect("build");

    baracuda_driver::init().expect("cuInit");
    let device = Device::get(0).expect("device");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let runtime = Runtime::with_logger(logger).expect("runtime");
    let engine = runtime
        .deserialize_engine(plan.as_slice().expect("plan"))
        .expect("deserialize");
    let exec = engine.create_execution_context().expect("exec");

    let input = [-3.0f32, -1.0, 0.0, 0.5, 2.0, 7.0];
    let x = DeviceBuffer::from_slice(&ctx, &input).expect("upload");
    let y = DeviceBuffer::<f32>::zeros(&ctx, 6).expect("output");
    // SAFETY: `x` and `y` are live device allocations of six floats that outlive the enqueue and the stream synchronize that follows.
    unsafe {
        exec.set_tensor_address("x", x.as_raw().0 as usize as *mut _)
            .expect("bind x");
        exec.set_tensor_address("y", y.as_raw().0 as usize as *mut _)
            .expect("bind y");
        exec.enqueue_v3(&stream).expect("enqueue");
    }
    stream.synchronize().expect("sync");
    let mut got = [0f32; 6];
    y.copy_to_host(&mut got).expect("download");
    let want: Vec<f32> = input.iter().map(|v| (-v).max(0.0)).collect();
    assert_eq!(got.as_slice(), want.as_slice());
    assert!(want.contains(&0.0) && want.iter().any(|v| *v > 0.0));
}

#[test]
fn malformed_bytes_report_errors_but_a_valid_model_still_parses() {
    let logger = require_optional!(logger(), "C-ABI shim (Logger::new)");
    let builder = Builder::new(Arc::clone(&logger)).expect("builder");

    let bad_net = builder
        .create_network(NetworkFlags::STRONGLY_TYPED)
        .expect("network");
    let bad = OnnxParser::new(&bad_net, Arc::clone(&logger)).expect("parser");
    let err = bad
        .parse(&[0xff, 0xff, 0xff, 0xff, 0x01, 0x02], None)
        .expect_err("garbage is not an ONNX model");
    assert!(!err.0.is_empty(), "the parser reported no error");
    assert!(err.0.iter().any(|e| e.code != 0), "{err:?}");
    assert!(
        bad.errors().is_empty(),
        "errors are cleared after being returned"
    );

    // Control: the same builder and logger parse a valid model.
    let ok_net = builder
        .create_network(NetworkFlags::STRONGLY_TYPED)
        .expect("network");
    let ok = OnnxParser::new(&ok_net, logger).expect("parser");
    ok.parse(&model(&["Relu"], &[4]), None)
        .expect("valid model");
    assert_eq!(ok_net.num_layers(), 1);
}

#[test]
fn an_unsupported_operator_names_the_operator() {
    let logger = require_optional!(logger(), "C-ABI shim (Logger::new)");
    let builder = Builder::new(Arc::clone(&logger)).expect("builder");
    let network = builder
        .create_network(NetworkFlags::STRONGLY_TYPED)
        .expect("network");
    let parser = OnnxParser::new(&network, logger).expect("parser");

    // Positive control for the question being asked.
    assert!(parser.supports_operator("Relu").unwrap());
    assert!(!parser.supports_operator("NotAnOnnxOperator").unwrap());

    let err = parser
        .parse(&model(&["NotAnOnnxOperator"], &[4]), None)
        .expect_err("unknown operator");
    assert!(
        err.0.iter().any(|e| e.node_operator == "NotAnOnnxOperator"),
        "no error named the operator: {err:?}"
    );
}
