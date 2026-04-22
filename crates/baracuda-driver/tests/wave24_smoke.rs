//! GPU-gated integration tests for Wave-24 Driver-API additions:
//! graph memory nodes + `cuGraphExecUpdate_v2` + device graph-mem attrs.

use core::ffi::c_void;

use baracuda_cuda_sys::types::{CUgraphExecUpdateResult, CUgraphMem_attribute, CUgraphNodeType};
use baracuda_driver::graph::{self, UpdateResult};
use baracuda_driver::{Context, Device, DeviceBuffer, Graph, Module, Stream};

const VECTOR_ADD_PTX: &str = include_str!("kernels/vector_add.ptx");

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn mem_alloc_and_free_nodes_round_trip() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let g = Graph::new(&ctx).unwrap();
    let bytes = 1024 * core::mem::size_of::<u32>();
    let (alloc_node, _dptr) = g.add_mem_alloc_node(&[], &device, bytes).unwrap();
    // Must free before instantiate can succeed, otherwise the graph
    // leaks every run.
    let free_node = g.add_mem_free_node(&[alloc_node], _dptr).unwrap();

    assert_eq!(alloc_node.node_type().unwrap(), CUgraphNodeType::MEM_ALLOC);
    assert_eq!(free_node.node_type().unwrap(), CUgraphNodeType::MEM_FREE);
    assert_eq!(free_node.mem_free_ptr().unwrap().0, _dptr.0);
    assert_eq!(alloc_node.mem_alloc_params().unwrap().bytesize, bytes);

    let exec = g.instantiate().unwrap();
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    // Used memory should be back near 0 after the free-node runs.
    let used =
        graph::device_graph_mem_attribute(&device, CUgraphMem_attribute::USED_MEM_CURRENT).unwrap();
    eprintln!("graph mem used after free-node: {used}");

    // Trim shouldn't error.
    graph::device_graph_mem_trim(&device).unwrap();
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn graph_exec_update_accepts_topology_invariant_changes() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let n: u32 = 512;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let d_a = DeviceBuffer::from_slice(&ctx, &a).unwrap();
    let d_b = DeviceBuffer::from_slice(&ctx, &b).unwrap();
    let d_c1: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n as usize).unwrap();
    let d_c2: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n as usize).unwrap();

    let module = Module::load_ptx(&ctx, VECTOR_ADD_PTX).unwrap();
    let kernel = module.get_function("vector_add").unwrap();

    let build_graph = |out: baracuda_cuda_sys::CUdeviceptr,
                       a_ptr: baracuda_cuda_sys::CUdeviceptr,
                       b_ptr: baracuda_cuda_sys::CUdeviceptr| {
        let g = Graph::new(&ctx).unwrap();
        let mut args: [*mut c_void; 4] = [
            &a_ptr as *const _ as *mut c_void,
            &b_ptr as *const _ as *mut c_void,
            &out as *const _ as *mut c_void,
            &n as *const _ as *mut c_void,
        ];
        let _k = unsafe {
            g.add_kernel_node(
                &[],
                &kernel,
                (n.div_ceil(256), 1, 1),
                (256u32, 1, 1),
                0,
                &mut args,
            )
            .unwrap()
        };
        g
    };

    // Build + instantiate graph targeting d_c1.
    let g1 = build_graph(d_c1.as_raw(), d_a.as_raw(), d_b.as_raw());
    let exec = g1.instantiate().unwrap();

    // Build a fresh *topology-identical* template targeting d_c2, then
    // update in place.
    let g2 = build_graph(d_c2.as_raw(), d_a.as_raw(), d_b.as_raw());
    let result: UpdateResult = exec.update(&g2).unwrap();
    assert!(
        result.is_success() || result.result == CUgraphExecUpdateResult::SUCCESS,
        "update failed: {:?}",
        result
    );

    // Relaunch: output should now land in d_c2, not d_c1.
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0.0f32; n as usize];
    d_c2.copy_to_host(&mut got).unwrap();
    assert_eq!(expected, got, "update path should re-target output buffer");
}
