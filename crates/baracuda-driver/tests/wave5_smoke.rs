//! GPU-gated integration tests for Wave-5 Driver-API additions:
//! explicit graph node construction via `cuGraphAddKernelNode_v2`,
//! `cuGraphAddEmptyNode`, `cuGraphAddMemsetNode`, and `cuGraphClone`.

use core::ffi::c_void;

use baracuda_driver::{Context, Device, DeviceBuffer, Graph, Module, Stream};

const VECTOR_ADD_PTX: &str = include_str!("kernels/vector_add.ptx");

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn explicit_graph_empty_memset_kernel() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let n: u32 = 1 << 12;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 3.0).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let d_a = DeviceBuffer::from_slice(&ctx, &a).unwrap();
    let d_b = DeviceBuffer::from_slice(&ctx, &b).unwrap();
    let d_c: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n as usize).unwrap();

    let module = Module::load_ptx(&ctx, VECTOR_ADD_PTX).unwrap();
    let kernel = module.get_function("vector_add").unwrap();

    // Build the graph by hand: empty root -> memset(d_c, 0xFFFFFFFF) -> vector_add.
    let graph = Graph::new(&ctx).unwrap();
    let root = graph.add_empty_node(&[]).unwrap();

    // Memset writes a sentinel so we can tell it ran (and was ordered before
    // the kernel). vector_add fully overwrites d_c, so the final output
    // should still be a+b — the memset is purely a dependency-ordering test.
    let memset = graph
        .add_memset_u32_node(&[root], d_c.as_raw(), 0xDEAD_BEEF, n as usize)
        .unwrap();

    // Build the kernel-arg void** table. Pointers and n must stay alive
    // until after cuGraphInstantiate copies them into the exec.
    let a_ptr = d_a.as_raw();
    let b_ptr = d_b.as_raw();
    let c_ptr = d_c.as_raw();
    let n_local = n;
    let mut args: [*mut c_void; 4] = [
        &a_ptr as *const _ as *mut c_void,
        &b_ptr as *const _ as *mut c_void,
        &c_ptr as *const _ as *mut c_void,
        &n_local as *const _ as *mut c_void,
    ];

    // SAFETY: kernel signature is (float*, float*, float*, uint).
    let _kernel_node = unsafe {
        graph
            .add_kernel_node(
                &[memset],
                &kernel,
                (n.div_ceil(256), 1, 1),
                (256u32, 1, 1),
                0,
                &mut args,
            )
            .unwrap()
    };

    // 3 nodes: empty + memset + kernel.
    assert_eq!(graph.node_count().unwrap(), 3);

    let exec = graph.instantiate().unwrap();
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    let mut got = vec![0.0f32; n as usize];
    d_c.copy_to_host(&mut got).unwrap();
    assert_eq!(expected, got, "explicit graph produced wrong result");
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn cloned_graph_runs_independently() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let n = 256usize;
    let d_c: DeviceBuffer<u32> = DeviceBuffer::new(&ctx, n).unwrap();

    let graph = Graph::new(&ctx).unwrap();
    let _ = graph
        .add_memset_u32_node(&[], d_c.as_raw(), 0xA5A5_A5A5, n)
        .unwrap();

    // Clone, instantiate both, run both. Both should land the same pattern
    // into d_c. Order isn't observable here — we're only verifying the
    // clone is a functioning executable graph.
    let clone = graph.clone_graph().unwrap();
    assert_eq!(clone.node_count().unwrap(), 1);

    let exec_a = graph.instantiate().unwrap();
    let exec_b = clone.instantiate().unwrap();

    exec_a.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0u32; n];
    d_c.copy_to_host(&mut got).unwrap();
    assert!(got.iter().all(|&v| v == 0xA5A5_A5A5));

    // Overwrite with zeroes, then run the clone — should re-apply the
    // pattern.
    let zeros = vec![0u32; n];
    d_c.copy_from_host(&zeros).unwrap();
    exec_b.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    d_c.copy_to_host(&mut got).unwrap();
    assert!(got.iter().all(|&v| v == 0xA5A5_A5A5));
}
