//! Optimization profiles: build an engine with a dynamic batch axis, then run
//! it at several shapes and across several profiles on the GPU.
//!
//! ```text
//! cargo gpu-test -p baracuda-tensorrt-builder --features shim,device-tests
//! ```

#![cfg(all(feature = "shim", feature = "device-tests"))]

use std::sync::Arc;

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, require_optional};
use baracuda_tensorrt::{DataType, Dims, ExecutionContext, HostMemory, Logger, Runtime, Severity};
use baracuda_tensorrt_builder::{
    Activation, Builder, MemoryPool, NetworkFlags, OptProfileSelector,
};

const COLS: i64 = 4;

fn logger() -> Option<Arc<Logger>> {
    Logger::stderr(Severity::Error).ok().map(Arc::new)
}

fn shape(batch: i64) -> Dims {
    Dims::new(&[batch, COLS])
}

/// Build `y = relu(x)` for `x: [-1, COLS]` with one profile per `(min, opt, max)`.
fn build_relu(logger: Arc<Logger>, profiles: &[(i64, i64, i64)]) -> HostMemory {
    let builder = Builder::new(logger).expect("builder");
    let network = builder
        .create_network(NetworkFlags::STRONGLY_TYPED)
        .expect("network");
    let x = network
        .add_input("x", DataType::Float, &shape(-1))
        .expect("input");
    let y = network
        .add_activation(&x, Activation::Relu, None, None)
        .expect("relu")
        .output(0)
        .expect("output");
    y.set_name("y").expect("name");
    network.mark_output(&y).expect("mark");

    let config = builder.create_config().expect("config");
    config.set_memory_pool_limit(MemoryPool::Workspace, 64 << 20);
    for &(min, opt, max) in profiles {
        let profile = builder.create_optimization_profile().expect("profile");
        profile
            .set_dimensions("x", OptProfileSelector::Min, &shape(min))
            .expect("min");
        profile
            .set_dimensions("x", OptProfileSelector::Opt, &shape(opt))
            .expect("opt");
        profile
            .set_dimensions("x", OptProfileSelector::Max, &shape(max))
            .expect("max");
        config
            .add_optimization_profile(&profile)
            .expect("add profile");
    }
    builder.build_serialized(&network, &config).expect("build")
}

struct Device0 {
    ctx: Context,
    stream: Stream,
}

fn device() -> Device0 {
    baracuda_driver::init().expect("cuInit");
    let device = Device::get(0).expect("device");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    Device0 { ctx, stream }
}

/// Run `relu` for `batch` rows on `exec`; `(output, expected)`.
fn run(dev: &Device0, exec: &ExecutionContext<'_>, batch: i64) -> (Vec<f32>, Vec<f32>) {
    try_run(dev, exec, batch).expect("enqueue")
}

/// Like [`run`], but returns the enqueue error instead of panicking.
fn try_run(
    dev: &Device0,
    exec: &ExecutionContext<'_>,
    batch: i64,
) -> baracuda_tensorrt::Result<(Vec<f32>, Vec<f32>)> {
    let n = (batch * COLS) as usize;
    let input: Vec<f32> = (0..n)
        .map(|i| {
            if i % 3 == 0 {
                -(i as f32) - 1.0
            } else {
                i as f32
            }
        })
        .collect();
    exec.set_input_shape("x", shape(batch))
        .expect("input shape");
    let x = DeviceBuffer::from_slice(&dev.ctx, &input).expect("upload");
    let y = DeviceBuffer::<f32>::zeros(&dev.ctx, n).expect("output");
    unsafe {
        exec.set_tensor_address("x", x.as_raw().0 as usize as *mut _)
            .expect("bind x");
        exec.set_tensor_address("y", y.as_raw().0 as usize as *mut _)
            .expect("bind y");
        exec.enqueue_v3(&dev.stream)?;
    }
    dev.stream.synchronize().expect("sync");
    let mut out = vec![0f32; n];
    y.copy_to_host(&mut out).expect("download");
    let want = input.iter().map(|v| v.max(0.0)).collect();
    Ok((out, want))
}

#[test]
fn one_engine_runs_at_two_batch_sizes() {
    let logger = require_optional!(logger(), "C-ABI shim (Logger::new)");
    let plan = build_relu(Arc::clone(&logger), &[(1, 2, 8)]);
    let dev = device();
    let runtime = Runtime::with_logger(logger).expect("runtime");
    let engine = runtime
        .deserialize_engine(plan.as_slice().expect("plan"))
        .expect("deserialize");

    assert_eq!(engine.num_optimization_profiles().unwrap(), 1);
    for (selector, batch) in [
        (OptProfileSelector::Min, 1),
        (OptProfileSelector::Opt, 2),
        (OptProfileSelector::Max, 8),
    ] {
        assert_eq!(
            engine.profile_shape("x", 0, selector).unwrap().as_slice(),
            &[batch, COLS],
            "{selector:?}"
        );
    }

    let exec = engine.create_execution_context().expect("context");
    for batch in [2, 5, 8, 1] {
        let (got, want) = run(&dev, &exec, batch);
        assert_eq!(got, want, "batch {batch}");
        assert!(want.contains(&0.0) && want.iter().any(|v| *v > 0.0));
    }
}

#[test]
fn a_shape_outside_the_profile_is_rejected() {
    let logger = require_optional!(logger(), "C-ABI shim (Logger::new)");
    let plan = build_relu(Arc::clone(&logger), &[(1, 2, 8)]);
    let runtime = Runtime::with_logger(logger).expect("runtime");
    let engine = runtime
        .deserialize_engine(plan.as_slice().expect("plan"))
        .expect("deserialize");
    let exec = engine.create_execution_context().expect("context");
    // Control: the profile's own maximum is accepted by the same context.
    exec.set_input_shape("x", shape(8)).expect("max is allowed");
    assert!(exec.set_input_shape("x", shape(9)).is_err(), "9 > max");
    assert!(exec.set_input_shape("x", shape(0)).is_err(), "0 < min");
}

#[test]
fn contexts_switch_between_two_profiles() {
    let logger = require_optional!(logger(), "C-ABI shim (Logger::new)");
    let plan = build_relu(Arc::clone(&logger), &[(1, 2, 4), (5, 6, 8)]);
    let dev = device();
    let runtime = Runtime::with_logger(logger).expect("runtime");
    let engine = runtime
        .deserialize_engine(plan.as_slice().expect("plan"))
        .expect("deserialize");
    assert_eq!(engine.num_optimization_profiles().unwrap(), 2);
    assert_eq!(
        engine
            .profile_shape("x", 1, OptProfileSelector::Max)
            .unwrap()
            .as_slice(),
        &[8, COLS]
    );

    let exec = engine.create_execution_context().expect("context");
    assert_eq!(exec.optimization_profile(), 0);
    // Control: batch 3 is inside profile 0 and runs.
    let (got, want) = run(&dev, &exec, 3);
    assert_eq!(got, want);
    // Batch 6 is outside profile 0: TensorRT accepts the shape (profile 1 has
    // it) but refuses to run it until the context selects profile 1.
    assert!(try_run(&dev, &exec, 6).is_err());
    // It runs once the context selects profile 1.
    exec.set_optimization_profile_async(1, &dev.stream)
        .expect("select profile 1");
    dev.stream.synchronize().expect("sync");
    assert_eq!(exec.optimization_profile(), 1);
    let (got, want) = run(&dev, &exec, 6);
    assert_eq!(got, want);
    assert!(
        exec.set_optimization_profile_async(2, &dev.stream).is_err(),
        "there is no profile 2"
    );
}

#[test]
fn an_inconsistent_profile_is_rejected_at_build_time() {
    let logger = require_optional!(logger(), "C-ABI shim (Logger::new)");
    let builder = Builder::new(logger).expect("builder");
    let config = builder.create_config().expect("config");

    let good = builder.create_optimization_profile().expect("profile");
    for (selector, batch) in [
        (OptProfileSelector::Min, 1),
        (OptProfileSelector::Opt, 2),
        (OptProfileSelector::Max, 4),
    ] {
        good.set_dimensions("x", selector, &shape(batch))
            .expect("dims");
    }
    assert!(good.is_valid());
    assert_eq!(
        good.dimensions("x", OptProfileSelector::Max)
            .unwrap()
            .as_slice(),
        &[4, COLS]
    );
    assert_eq!(
        config
            .add_optimization_profile(&good)
            .expect("valid profile"),
        0
    );
    assert_eq!(config.num_optimization_profiles(), 1);

    // min > max is not a valid profile.
    let bad = builder.create_optimization_profile().expect("profile");
    for (selector, batch) in [
        (OptProfileSelector::Min, 8),
        (OptProfileSelector::Opt, 4),
        (OptProfileSelector::Max, 1),
    ] {
        bad.set_dimensions("x", selector, &shape(batch))
            .expect("dims");
    }
    assert!(!bad.is_valid());
    assert!(config.add_optimization_profile(&bad).is_err());
    assert_eq!(config.num_optimization_profiles(), 1);
}
