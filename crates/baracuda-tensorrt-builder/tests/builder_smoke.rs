//! Builds small networks with the builder API, deserializes the plan with the
//! runtime crate, runs it on the GPU, and compares with a CPU reference.
//!
//! Gated behind `device-tests` and needing the shim:
//!
//! ```text
//! cargo gpu-test -p baracuda-tensorrt-builder --features shim,device-tests
//! ```

#![cfg(all(feature = "shim", feature = "device-tests"))]

use std::sync::Arc;

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, require_optional};
use baracuda_tensorrt::{DataType, Dims, Logger, Runtime, Severity};
use baracuda_tensorrt_builder::{
    Activation, Builder, BuilderConfig, BuilderFlag, ElementWiseOp, MatrixOp, MemoryPool, Network,
    NetworkFlags, ReduceOp, UnaryOp,
};

fn logger() -> Option<Arc<Logger>> {
    Logger::stderr(Severity::Warning).ok().map(Arc::new)
}

/// Build with `define`, run on the device with `inputs`, and return output `out`.
fn build_and_run(
    logger: Arc<Logger>,
    define: impl for<'n> Fn(&'n Network<'_>) -> baracuda_tensorrt::Result<()>,
    inputs: &[(&str, &[f32])],
    out: &str,
    out_len: usize,
) -> Vec<f32> {
    let builder = Builder::new(Arc::clone(&logger)).expect("builder");
    let network = builder
        .create_network(NetworkFlags::STRONGLY_TYPED)
        .expect("network");
    define(&network).expect("define network");
    let config = builder.create_config().expect("config");
    config.set_memory_pool_limit(MemoryPool::Workspace, 64 << 20);
    let plan = builder.build_serialized(&network, &config).expect("build");

    baracuda_driver::init().expect("cuInit");
    let device = Device::get(0).expect("device");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let runtime = Runtime::with_logger(logger).expect("runtime");
    let engine = runtime
        .deserialize_engine(plan.as_slice().expect("plan bytes"))
        .expect("deserialize");
    let exec = engine.create_execution_context().expect("exec context");

    let mut bufs = Vec::new();
    for (name, data) in inputs {
        let buf = DeviceBuffer::from_slice(&ctx, data).expect("upload");
        unsafe { exec.set_tensor_address(name, buf.as_raw().0 as usize as *mut _) }
            .expect("bind in");
        bufs.push(buf);
    }
    let out_buf = DeviceBuffer::<f32>::zeros(&ctx, out_len).expect("out buffer");
    unsafe { exec.set_tensor_address(out, out_buf.as_raw().0 as usize as *mut _) }
        .expect("bind out");
    unsafe { exec.enqueue_v3(&stream) }.expect("enqueue");
    stream.synchronize().expect("sync");
    let mut host = vec![0f32; out_len];
    out_buf.copy_to_host(&mut host).expect("download");
    host
}

fn assert_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "element {i}: got {g}, want {w} (tol {tol})"
        );
    }
}

#[test]
fn element_wise_sum_matches_cpu() {
    let logger = require_optional!(logger(), "C-ABI shim (Logger::new)");
    let a: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..8).map(|i| 10.0 * i as f32 + 0.5).collect();
    let got = build_and_run(
        logger,
        |net| {
            let d = Dims::new(&[8]);
            let ta = net.add_input("a", DataType::Float, &d)?;
            let tb = net.add_input("b", DataType::Float, &d)?;
            let out = net
                .add_element_wise(&ta, &tb, ElementWiseOp::Sum)?
                .output(0)?;
            out.set_name("out")?;
            net.mark_output(&out)
        },
        &[("a", &a), ("b", &b)],
        "out",
        8,
    );
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_close(&got, &want, 0.0);
}

#[test]
fn matmul_then_relu_matches_cpu() {
    let logger = require_optional!(logger(), "C-ABI shim (Logger::new)");
    // A is 2x3, B is 3x2; some products are negative so ReLU matters.
    let a = [1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0];
    let b = [1.0f32, 0.5, -1.0, 2.0, 0.0, -3.0];
    let got = build_and_run(
        logger,
        |net| {
            let ta = net.add_input("a", DataType::Float, &Dims::new(&[2, 3]))?;
            let tb = net.add_input("b", DataType::Float, &Dims::new(&[3, 2]))?;
            let mm = net
                .add_matrix_multiply(&ta, MatrixOp::None, &tb, MatrixOp::None)?
                .output(0)?;
            let out = net
                .add_activation(&mm, Activation::Relu, None, None)?
                .output(0)?;
            out.set_name("out")?;
            net.mark_output(&out)
        },
        &[("a", &a), ("b", &b)],
        "out",
        4,
    );
    let mut want = [0f32; 4];
    for i in 0..2 {
        for j in 0..2 {
            let s: f32 = (0..3).map(|k| a[i * 3 + k] * b[k * 2 + j]).sum();
            want[i * 2 + j] = s.max(0.0);
        }
    }
    assert!(want.iter().any(|v| *v == 0.0) && want.iter().any(|v| *v > 0.0));
    assert_close(&got, &want, 1e-4);
}

#[test]
fn constant_scale_and_unary_match_cpu() {
    let logger = require_optional!(logger(), "C-ABI shim (Logger::new)");
    let x = [0.25f32, 1.0, 4.0, 9.0];
    let got = build_and_run(
        logger,
        |net| {
            let tx = net.add_input("x", DataType::Float, &Dims::new(&[4]))?;
            let two = net
                .add_constant_f32(&Dims::new(&[4]), &[2.0, 2.0, 2.0, 2.0])?
                .output(0)?;
            let scaled = net
                .add_element_wise(&tx, &two, ElementWiseOp::Prod)?
                .output(0)?;
            let out = net.add_unary(&scaled, UnaryOp::Sqrt)?.output(0)?;
            out.set_name("out")?;
            net.mark_output(&out)
        },
        &[("x", &x)],
        "out",
        4,
    );
    let want: Vec<f32> = x.iter().map(|v| (2.0 * v).sqrt()).collect();
    assert_close(&got, &want, 1e-5);
}

#[test]
fn softmax_rows_sum_to_one_and_match_cpu() {
    let logger = require_optional!(logger(), "C-ABI shim (Logger::new)");
    let x = [1.0f32, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0];
    let got = build_and_run(
        logger,
        |net| {
            let tx = net.add_input("x", DataType::Float, &Dims::new(&[2, 4]))?;
            let out = net.add_softmax(&tx, 1 << 1)?.output(0)?;
            out.set_name("out")?;
            net.mark_output(&out)
        },
        &[("x", &x)],
        "out",
        8,
    );
    let mut want = [0f32; 8];
    for r in 0..2 {
        let row = &x[r * 4..r * 4 + 4];
        let denom: f32 = row.iter().map(|v| v.exp()).sum();
        for (c, v) in row.iter().enumerate() {
            want[r * 4 + c] = v.exp() / denom;
        }
    }
    assert_close(&got, &want, 1e-5);
}

#[test]
fn shuffle_transpose_and_reduce_match_cpu() {
    let logger = require_optional!(logger(), "C-ABI shim (Logger::new)");
    let x: Vec<f32> = (0..6).map(|i| i as f32).collect(); // 2x3
    let got = build_and_run(
        logger,
        |net| {
            let tx = net.add_input("x", DataType::Float, &Dims::new(&[2, 3]))?;
            let shuf = net.add_shuffle(&tx)?;
            shuf.set_first_transpose(&[1, 0]);
            let t = shuf.output(0)?; // 3x2
            let out = net
                .add_reduce(&t, ReduceOp::Sum, 1 << 1, false)?
                .output(0)?; // sum over axis 1 -> [3]
            out.set_name("out")?;
            net.mark_output(&out)
        },
        &[("x", &x)],
        "out",
        3,
    );
    // Transposed (3x2) rows are the columns of x; summing each gives column sums.
    let want = [0.0 + 3.0, 1.0 + 4.0, 2.0 + 5.0];
    assert_close(&got, &want, 1e-5);
}

#[test]
fn concatenation_joins_along_the_chosen_axis() {
    let logger = require_optional!(logger(), "C-ABI shim (Logger::new)");
    let a = [1.0f32, 2.0];
    let b = [3.0f32, 4.0, 5.0];
    let got = build_and_run(
        logger,
        |net| {
            let ta = net.add_input("a", DataType::Float, &Dims::new(&[2]))?;
            let tb = net.add_input("b", DataType::Float, &Dims::new(&[3]))?;
            let cat = net.add_concatenation(&[&ta, &tb])?;
            cat.set_concat_axis(0);
            let out = cat.output(0)?;
            out.set_name("out")?;
            net.mark_output(&out)
        },
        &[("a", &a), ("b", &b)],
        "out",
        5,
    );
    assert_close(&got, &[1.0, 2.0, 3.0, 4.0, 5.0], 0.0);
}

#[test]
fn config_round_trips_and_network_is_inspectable() {
    let logger = require_optional!(logger(), "C-ABI shim (Logger::new)");
    let builder = Builder::new(logger).expect("builder");
    let config: BuilderConfig<'_> = builder.create_config().expect("config");
    config.set_memory_pool_limit(MemoryPool::Workspace, 32 << 20);
    assert_eq!(config.memory_pool_limit(MemoryPool::Workspace), 32 << 20);
    config.set_optimization_level(2);
    assert_eq!(config.optimization_level(), 2);
    assert!(!config.flag(BuilderFlag::Debug));
    config.set_flag(BuilderFlag::Debug, true);
    assert!(config.flag(BuilderFlag::Debug));

    let net = builder
        .create_network(NetworkFlags::STRONGLY_TYPED)
        .expect("network");
    let t = net
        .add_input("x", DataType::Float, &Dims::new(&[2, 5]))
        .expect("input");
    assert_eq!(t.name(), "x");
    assert_eq!(t.dimensions().as_slice(), &[2, 5]);
    assert_eq!(t.data_type(), Some(DataType::Float));
    assert_eq!(
        (net.num_inputs(), net.num_outputs(), net.num_layers()),
        (1, 0, 0)
    );
}

#[test]
fn a_network_without_outputs_fails_to_build() {
    let logger = require_optional!(logger(), "C-ABI shim (Logger::new)");
    let builder = Builder::new(logger).expect("builder");
    let config = builder.create_config().expect("config");
    // Control: the same builder and config build a network that has an output.
    let ok = builder
        .create_network(NetworkFlags::STRONGLY_TYPED)
        .expect("network");
    let x = ok
        .add_input("x", DataType::Float, &Dims::new(&[4]))
        .expect("input");
    let y = ok
        .add_identity(&x)
        .expect("identity")
        .output(0)
        .expect("out");
    y.set_name("y").expect("name");
    ok.mark_output(&y).expect("mark");
    assert!(
        builder.build_serialized(&ok, &config).is_ok(),
        "control build"
    );

    let bad = builder
        .create_network(NetworkFlags::STRONGLY_TYPED)
        .expect("network");
    bad.add_input("x", DataType::Float, &Dims::new(&[4]))
        .expect("input");
    assert!(
        builder.build_serialized(&bad, &config).is_err(),
        "a network with no outputs must not build"
    );
}
