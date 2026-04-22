//! GPU-gated integration tests for Wave-12 Driver-API additions:
//! full graph node builders + param edit APIs.

use core::ffi::c_void;
use core::mem::size_of;
use std::sync::atomic::{AtomicU32, Ordering};

use baracuda_cuda_sys::types::{CUgraphNodeType, CUmemorytype, CUDA_MEMCPY3D};
use baracuda_driver::{Context, Device, DeviceBuffer, Graph, Module, Stream};

const VECTOR_ADD_PTX: &str = include_str!("kernels/vector_add.ptx");

static HOST_NODE_CALLS: AtomicU32 = AtomicU32::new(0);

unsafe extern "C" fn bump_counter(_user_data: *mut c_void) {
    HOST_NODE_CALLS.fetch_add(1, Ordering::SeqCst);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn graph_memcpy_and_host_nodes_run_in_order() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let n = 512usize;
    let src_device: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(7)).collect();
    let d_src = DeviceBuffer::<u32>::from_slice(&ctx, &src_device).unwrap();
    let d_dst: DeviceBuffer<u32> = DeviceBuffer::new(&ctx, n).unwrap();

    // Build: root (empty) -> memcpy DtoD(d_src -> d_dst) -> host node that bumps a counter.
    let graph = Graph::new(&ctx).unwrap();
    let root = graph.add_empty_node(&[]).unwrap();

    let mut p = CUDA_MEMCPY3D::default();
    p.src_memory_type = CUmemorytype::DEVICE;
    p.src_device = d_src.as_raw();
    p.src_pitch = n * size_of::<u32>();
    p.src_height = 1;
    p.dst_memory_type = CUmemorytype::DEVICE;
    p.dst_device = d_dst.as_raw();
    p.dst_pitch = n * size_of::<u32>();
    p.dst_height = 1;
    p.width_in_bytes = n * size_of::<u32>();
    p.height = 1;
    p.depth = 1;
    let memcpy = graph.add_memcpy_node(&[root], &p).unwrap();

    let host_node = unsafe {
        graph
            .add_host_node(&[memcpy], bump_counter, core::ptr::null_mut())
            .unwrap()
    };

    // Verify topology queries.
    assert_eq!(graph.node_count().unwrap(), 3);
    let (from, to) = graph.edges().unwrap();
    assert_eq!(from.len(), 2);
    assert_eq!(to.len(), 2);

    assert_eq!(root.node_type().unwrap(), CUgraphNodeType::EMPTY);
    assert_eq!(memcpy.node_type().unwrap(), CUgraphNodeType::MEMCPY);
    assert_eq!(host_node.node_type().unwrap(), CUgraphNodeType::HOST);
    assert_eq!(memcpy.dependencies().unwrap().len(), 1);
    assert_eq!(host_node.dependencies().unwrap().len(), 1);

    // Instantiate + launch twice — the host node should bump the counter twice.
    HOST_NODE_CALLS.store(0, Ordering::SeqCst);
    let exec = graph.instantiate().unwrap();
    exec.launch(&stream).unwrap();
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(HOST_NODE_CALLS.load(Ordering::SeqCst), 2);

    let mut back = vec![0u32; n];
    d_dst.copy_to_host(&mut back).unwrap();
    assert_eq!(src_device, back);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn graph_exec_live_edit_kernel_params() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let n: u32 = 1024;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 2.0).collect();
    let d_a = DeviceBuffer::from_slice(&ctx, &a).unwrap();
    let d_b = DeviceBuffer::from_slice(&ctx, &b).unwrap();
    let d_c1: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n as usize).unwrap();
    let d_c2: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n as usize).unwrap();

    let module = Module::load_ptx(&ctx, VECTOR_ADD_PTX).unwrap();
    let kernel = module.get_function("vector_add").unwrap();

    // Build graph with kernel targeting d_c1.
    let graph = Graph::new(&ctx).unwrap();
    let a_ptr = d_a.as_raw();
    let b_ptr = d_b.as_raw();
    let mut c_ptr = d_c1.as_raw();
    let n_val = n;
    let mut args: [*mut c_void; 4] = [
        &a_ptr as *const _ as *mut c_void,
        &b_ptr as *const _ as *mut c_void,
        &c_ptr as *const _ as *mut c_void,
        &n_val as *const _ as *mut c_void,
    ];
    let knode = unsafe {
        graph
            .add_kernel_node(
                &[],
                &kernel,
                (n.div_ceil(256), 1, 1),
                (256u32, 1, 1),
                0,
                &mut args,
            )
            .unwrap()
    };

    let exec = graph.instantiate().unwrap();
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    // Now live-edit to target d_c2 and re-launch.
    c_ptr = d_c2.as_raw();
    let mut args2: [*mut c_void; 4] = [
        &a_ptr as *const _ as *mut c_void,
        &b_ptr as *const _ as *mut c_void,
        &c_ptr as *const _ as *mut c_void,
        &n_val as *const _ as *mut c_void,
    ];
    let mut new_params = knode.kernel_params().unwrap();
    new_params.kernel_params = args2.as_mut_ptr();
    unsafe { exec.set_kernel_node_params(knode, &new_params).unwrap() };
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    // Both outputs should match a+b.
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let mut got1 = vec![0.0f32; n as usize];
    let mut got2 = vec![0.0f32; n as usize];
    d_c1.copy_to_host(&mut got1).unwrap();
    d_c2.copy_to_host(&mut got2).unwrap();
    assert_eq!(expected, got1, "first launch wrote to d_c1");
    assert_eq!(expected, got2, "live-edited launch wrote to d_c2");
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn graph_memset_node_param_edit() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let n = 256usize;
    let d: DeviceBuffer<u32> = DeviceBuffer::new(&ctx, n).unwrap();

    let graph = Graph::new(&ctx).unwrap();
    let node = graph
        .add_memset_u32_node(&[], d.as_raw(), 0x1111_1111, n)
        .unwrap();

    // Fetch, mutate, push back — change the value pattern.
    let mut p = node.memset_params().unwrap();
    assert_eq!(p.value, 0x1111_1111);
    p.value = 0x2222_2222;
    node.set_memset_params(&p).unwrap();

    let exec = graph.instantiate().unwrap();
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    let mut got = vec![0u32; n];
    d.copy_to_host(&mut got).unwrap();
    assert!(got.iter().all(|&v| v == 0x2222_2222));
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn graph_child_and_event_nodes() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    // Parent graph contains a child graph (memset inside), then an event
    // record node.
    let child = Graph::new(&ctx).unwrap();
    let n = 64usize;
    let d: DeviceBuffer<u32> = DeviceBuffer::new(&ctx, n).unwrap();
    let _ = child
        .add_memset_u32_node(&[], d.as_raw(), 0xFEED_FACE, n)
        .unwrap();

    let parent = Graph::new(&ctx).unwrap();
    let child_node = parent.add_child_graph_node(&[], &child).unwrap();
    let event = baracuda_driver::Event::new(&ctx).unwrap();
    let _evt_node = parent.add_event_record_node(&[child_node], &event).unwrap();

    assert_eq!(child_node.node_type().unwrap(), CUgraphNodeType::GRAPH);
    assert_eq!(parent.node_count().unwrap(), 2);

    let exec = parent.instantiate().unwrap();
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    event.synchronize().unwrap();

    let mut got = vec![0u32; n];
    d.copy_to_host(&mut got).unwrap();
    assert!(got.iter().all(|&v| v == 0xFEED_FACE));
}
