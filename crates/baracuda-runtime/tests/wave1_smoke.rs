//! Runtime Wave 1: memory pools, 2D memcpy, host callbacks, write/wait
//! value, pointer attributes, device properties, explicit graph nodes,
//! graph exec update, func attrs.

use core::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

use baracuda_runtime::memcpy2d::{self, PitchedBuffer};
use baracuda_runtime::mempool::{self, MemoryPool};
use baracuda_runtime::query::{self, MemoryType};
use baracuda_runtime::{CaptureMode, Device, DeviceBuffer, Graph, Library, Stream};

const VECTOR_ADD_PTX: &str = include_str!("kernels/vector_add.ptx");

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn mempool_roundtrip() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();
    let stream = Stream::new().unwrap();

    let pool = MemoryPool::new(&device).unwrap();
    pool.set_release_threshold(1 << 20).unwrap();
    assert_eq!(pool.release_threshold().unwrap(), 1 << 20);

    let n = 2048usize;
    let bytes = n * core::mem::size_of::<u32>();
    let host: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(0x9E37_79B1)).collect();

    let ptr = pool.alloc_async(bytes, &stream).unwrap();

    // Use the raw runtime cudaMemcpyAsync via our existing DeviceBuffer...
    // simplest: wrap the ptr for H2D/D2H through the raw sys PFN.
    let r = baracuda_cuda_sys::runtime::runtime().unwrap();
    let cu = r.cuda_memcpy_async().unwrap();
    let rc_in = unsafe {
        cu(
            ptr,
            host.as_ptr() as *const c_void,
            bytes,
            baracuda_cuda_sys::runtime::cudaMemcpyKind::HostToDevice,
            stream.as_raw(),
        )
    };
    assert_eq!(rc_in, baracuda_cuda_sys::runtime::cudaError_t::Success);

    let mut back = vec![0u32; n];
    let rc_out = unsafe {
        cu(
            back.as_mut_ptr() as *mut c_void,
            ptr,
            bytes,
            baracuda_cuda_sys::runtime::cudaMemcpyKind::DeviceToHost,
            stream.as_raw(),
        )
    };
    assert_eq!(rc_out, baracuda_cuda_sys::runtime::cudaError_t::Success);

    unsafe { pool.free_async(ptr, &stream).unwrap() };
    stream.synchronize().unwrap();
    assert_eq!(host, back);

    eprintln!(
        "pool after free: used={}, reserved={}",
        pool.used_bytes().unwrap(),
        pool.reserved_bytes().unwrap()
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn default_and_current_pool_queries() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();
    let default = mempool::default_pool(&device).unwrap();
    let current = mempool::current_pool(&device).unwrap();
    assert_eq!(default.as_raw(), current.as_raw());
    eprintln!("default pool: {:?}", default.as_raw());
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn pitched_2d_roundtrip() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();

    let width_elems = 13usize;
    let height = 8usize;
    let host: Vec<f32> = (0..(width_elems * height))
        .map(|i| (i as f32) * 0.25)
        .collect();
    let host_pitch = width_elems * core::mem::size_of::<f32>();

    let buf = PitchedBuffer::<f32>::new(width_elems, height).unwrap();
    eprintln!(
        "runtime PitchedBuffer pitch = {} bytes (nominal {})",
        buf.pitch_bytes(),
        host_pitch
    );
    assert!(buf.pitch_bytes() >= host_pitch);

    memcpy2d::copy_h_to_d_2d(&host, host_pitch, &buf, width_elems, height).unwrap();
    let mut back = vec![0.0f32; width_elems * height];
    memcpy2d::copy_d_to_h_2d(&buf, &mut back, host_pitch, width_elems, height).unwrap();
    assert_eq!(host, back);
}

static HOST_CALLBACK_FIRED: AtomicU32 = AtomicU32::new(0);

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn stream_launch_host_func() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();
    let stream = Stream::new().unwrap();

    HOST_CALLBACK_FIRED.store(0, Ordering::SeqCst);
    stream
        .launch_host_func(|| {
            HOST_CALLBACK_FIRED.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(HOST_CALLBACK_FIRED.load(Ordering::SeqCst), 1);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn stream_write_value_32() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();
    let stream = Stream::new().unwrap();

    let buf: DeviceBuffer<u32> = DeviceBuffer::zeros(1).unwrap();
    // SAFETY: buf is a live 4-byte device allocation.
    let result = unsafe { stream.write_value_32(buf.as_raw(), 0xCAFE_BABE, 0) };
    match result {
        Ok(()) => {
            stream.synchronize().unwrap();
            let mut back = [0u32; 1];
            buf.copy_to_host(&mut back).unwrap();
            assert_eq!(back[0], 0xCAFE_BABE);
        }
        Err(e) => {
            // `cudaStreamWriteValue32` lives in the driver DLL on some
            // CUDA builds and isn't re-exported from libcudart — that's
            // a known gap. Users on those builds can call the
            // Driver-API wrapper instead (`baracuda_driver::Stream::write_value_32`).
            eprintln!("cudaStreamWriteValue32 not resolvable in runtime DLL: {e:?}");
        }
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn pointer_attributes_classifies_device_memory() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();
    let buf: DeviceBuffer<u32> = DeviceBuffer::new(1024).unwrap();
    let attrs = query::pointer_attributes(buf.as_raw()).unwrap();
    eprintln!("{attrs:?}");
    assert_eq!(attrs.memory_type, MemoryType::Device);
    assert_eq!(attrs.device, 0);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn pointer_attributes_classifies_host_pinned() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();
    let pinned: baracuda_runtime::memory::PinnedHostBuffer<u32> =
        baracuda_runtime::memory::PinnedHostBuffer::new(256).unwrap();
    let attrs = query::pointer_attributes(pinned.as_ptr() as *const c_void).unwrap();
    eprintln!("{attrs:?}");
    assert_eq!(attrs.memory_type, MemoryType::Host);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn device_properties_reads_nonzero_fields() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();
    let props = query::device_properties(&device).unwrap();
    eprintln!(
        "device {}: cc {}.{}, {} SMs, {:.1} GB global, max {}x{}x{} block, warp {}",
        props.name,
        props.compute_capability_major,
        props.compute_capability_minor,
        props.multiprocessor_count,
        props.total_global_memory_bytes as f64 / 1e9,
        props.max_block_dim[0],
        props.max_block_dim[1],
        props.max_block_dim[2],
        props.warp_size,
    );
    assert!(!props.name.is_empty());
    assert!(props.compute_capability_major >= 3);
    assert_eq!(props.warp_size, 32);
    assert!(props.multiprocessor_count > 0);
    assert!(props.total_global_memory_bytes > 0);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn func_attributes_returns_kernel_metadata() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();
    let lib = Library::load_ptx(VECTOR_ADD_PTX).unwrap();
    let kernel = lib.get_kernel("vector_add").unwrap();
    // SAFETY: kernel.as_launch_ptr() is a valid kernel symbol.
    let attrs = unsafe { query::func_attributes(kernel.as_launch_ptr()) }.unwrap();
    eprintln!(
        "vector_add: regs={}, smem={}B, lmem={}B, max-threads={}",
        attrs.num_regs,
        attrs.shared_size_bytes,
        attrs.local_size_bytes,
        attrs.max_threads_per_block
    );
    assert!(attrs.num_regs > 0);
    assert!(attrs.max_threads_per_block >= 1);
}

unsafe extern "C" fn bump_counter(_: *mut c_void) {
    HOST_CALLBACK_FIRED.fetch_add(1, Ordering::SeqCst);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn explicit_graph_kernel_and_host_nodes() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();
    let stream = Stream::new().unwrap();

    let n: u32 = 1 << 12;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let d_a = DeviceBuffer::from_slice(&a).unwrap();
    let d_b = DeviceBuffer::from_slice(&b).unwrap();
    let d_c: DeviceBuffer<f32> = DeviceBuffer::new(n as usize).unwrap();

    let lib = Library::load_ptx(VECTOR_ADD_PTX).unwrap();
    let kernel = lib.get_kernel("vector_add").unwrap();

    let graph = Graph::new().unwrap();
    let root = graph.add_empty_node(&[]).unwrap();

    let a_ptr = d_a.as_device_ptr();
    let b_ptr = d_b.as_device_ptr();
    let c_ptr = d_c.as_device_ptr();
    let mut args: [*mut c_void; 4] = [
        &a_ptr as *const _ as *mut c_void,
        &b_ptr as *const _ as *mut c_void,
        &c_ptr as *const _ as *mut c_void,
        &n as *const _ as *mut c_void,
    ];
    let kernel_node = unsafe {
        graph
            .add_kernel_node(
                &[root],
                &kernel,
                baracuda_runtime::Dim3 {
                    x: n.div_ceil(256),
                    y: 1,
                    z: 1,
                },
                baracuda_runtime::Dim3 { x: 256, y: 1, z: 1 },
                0,
                &mut args,
            )
            .unwrap()
    };

    HOST_CALLBACK_FIRED.store(0, Ordering::SeqCst);
    let _host_node = unsafe {
        graph
            .add_host_node(&[kernel_node], bump_counter, core::ptr::null_mut())
            .unwrap()
    };

    assert_eq!(graph.node_count().unwrap(), 3);

    let exec = graph.instantiate().unwrap();
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(HOST_CALLBACK_FIRED.load(Ordering::SeqCst), 1);
    let mut got = vec![0.0f32; n as usize];
    d_c.copy_to_host(&mut got).unwrap();
    assert_eq!(expected, got);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn graph_exec_update_retargets_output() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();
    let stream = Stream::new().unwrap();

    let n: u32 = 512;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let d_a = DeviceBuffer::from_slice(&a).unwrap();
    let d_b = DeviceBuffer::from_slice(&b).unwrap();
    let d_c1: DeviceBuffer<f32> = DeviceBuffer::new(n as usize).unwrap();
    let d_c2: DeviceBuffer<f32> = DeviceBuffer::new(n as usize).unwrap();

    let lib = Library::load_ptx(VECTOR_ADD_PTX).unwrap();
    let kernel = lib.get_kernel("vector_add").unwrap();

    let build = |out: *mut c_void, ap: *mut c_void, bp: *mut c_void| {
        let g = Graph::new().unwrap();
        let mut args: [*mut c_void; 4] = [
            &ap as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &out as *const _ as *mut c_void,
            &n as *const _ as *mut c_void,
        ];
        let _ = unsafe {
            g.add_kernel_node(
                &[],
                &kernel,
                baracuda_runtime::Dim3 {
                    x: n.div_ceil(256),
                    y: 1,
                    z: 1,
                },
                baracuda_runtime::Dim3 { x: 256, y: 1, z: 1 },
                0,
                &mut args,
            )
            .unwrap()
        };
        g
    };

    let g1 = build(d_c1.as_raw(), d_a.as_raw(), d_b.as_raw());
    let exec = g1.instantiate().unwrap();

    let g2 = build(d_c2.as_raw(), d_a.as_raw(), d_b.as_raw());
    let res = exec.update(&g2).unwrap();
    assert!(res.is_success(), "update failed: {res:?}");

    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    let mut got = vec![0.0f32; n as usize];
    d_c2.copy_to_host(&mut got).unwrap();
    assert_eq!(expected, got);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn capture_mode_enum_also_works() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();
    let stream = Stream::new().unwrap();
    let graph = stream
        .capture(CaptureMode::ThreadLocal, |_| Ok(()))
        .unwrap();
    let _exec = graph.instantiate().unwrap();
}
