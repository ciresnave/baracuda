//! GPU-gated integration tests for CUDA Graphs + async allocations.
//!
//! Run with: `cargo test -p baracuda-driver --test graph_smoke -- --ignored`

use baracuda_driver::{CaptureMode, Context, Device, DeviceBuffer, Module, Stream};

const VECTOR_ADD_PTX: &str = include_str!("kernels/vector_add.ptx");

fn setup() -> baracuda_driver::Result<(Context, Stream)> {
    baracuda_driver::init()?;
    let device = Device::get(0)?;
    let ctx = Context::new(&device)?;
    let stream = Stream::new(&ctx)?;
    Ok((ctx, stream))
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn stream_capture_and_replay_vector_add() {
    let (ctx, stream) = setup().expect("setup");
    let _ = &ctx;
    let n: u32 = 1 << 14;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 2.0).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let d_a = DeviceBuffer::from_slice(&ctx, &a).unwrap();
    let d_b = DeviceBuffer::from_slice(&ctx, &b).unwrap();
    let d_c: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n as usize).unwrap();

    let module = Module::load_ptx(&ctx, VECTOR_ADD_PTX).unwrap();
    let kernel = module.get_function("vector_add").unwrap();

    let block: u32 = 256;
    let grid: u32 = n.div_ceil(block);

    // Capture the kernel launch into a graph.
    let a_ptr = d_a.as_raw();
    let b_ptr = d_b.as_raw();
    let c_ptr = d_c.as_raw();
    let graph = stream
        .capture(CaptureMode::ThreadLocal, |s| {
            // SAFETY: arg types/order match the PTX signature.
            unsafe {
                kernel
                    .launch()
                    .grid((grid, 1, 1))
                    .block((block, 1, 1))
                    .stream(s)
                    .arg(&a_ptr)
                    .arg(&b_ptr)
                    .arg(&c_ptr)
                    .arg(&n)
                    .launch()
            }
        })
        .expect("capture");

    assert!(graph.node_count().unwrap() >= 1, "graph should have nodes");

    let exec = graph.instantiate().expect("cuGraphInstantiateWithFlags");

    // Launch the captured graph twice — the kernel fully overwrites d_c
    // each run, so both replays must produce the same expected output.
    for run in 0..2 {
        exec.launch(&stream).expect("cuGraphLaunch");
        stream.synchronize().unwrap();

        let mut got = vec![0.0f32; n as usize];
        d_c.copy_to_host(&mut got).unwrap();
        assert_eq!(expected, got, "mismatch on graph replay #{run}");
    }
    drop(ctx);
}

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA 11.2+"]
fn async_alloc_roundtrip() {
    let (ctx, stream) = setup().expect("setup");
    let n = 1024usize;
    let host: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();

    // Async allocate on `stream`, async-copy data, async-free.
    let buf = DeviceBuffer::<f32>::new_async(&ctx, n, &stream).unwrap();
    buf.copy_from_host_async(&host, &stream).unwrap();

    let mut back = vec![0.0f32; n];
    buf.copy_to_host_async(&mut back, &stream).unwrap();

    // Explicit async free (consumes buf so the sync Drop won't also run).
    buf.free_async(&stream).expect("cuMemFreeAsync");
    stream.synchronize().unwrap();

    assert_eq!(host, back);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn empty_graph_instantiates() {
    let (_ctx, stream) = setup().expect("setup");
    // Capture with no operations -> empty graph. Should still instantiate
    // and launch cleanly.
    let graph = stream
        .capture(CaptureMode::ThreadLocal, |_| Ok(()))
        .expect("capture empty");
    let exec = graph.instantiate().expect("instantiate");
    exec.launch(&stream).expect("launch empty");
    stream.synchronize().unwrap();
}
