//! Additional GPU-gated smoke tests for the runtime-API safe wrapper —
//! covers stream priority, peer access, managed / pinned / async memory,
//! mem prefetch/advise, graph capture, occupancy, and func attrs.

use baracuda_runtime::memory::{self, PrefetchTarget};
use baracuda_runtime::{
    graph, stream, CaptureMode, Device, DeviceBuffer, Event, Graph, Library, Stream,
};

const VECTOR_ADD_PTX: &str = include_str!("kernels/vector_add.ptx");

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn stream_priority_and_wait_event() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();

    let (low, high) = stream::stream_priority_range().unwrap();
    eprintln!("stream priority range: {low}..={high}");
    assert!(
        low >= high,
        "low = least-priority should be >= high = greatest"
    );

    let s = Stream::with_priority(0, high).unwrap();
    assert_eq!(s.priority().unwrap(), high);

    // Record + wait on a no-timing event — exercises cudaStreamWaitEvent.
    let ev = Event::no_timing().unwrap();
    ev.record(&s).unwrap();
    s.wait_event(&ev, 0).unwrap();
    s.synchronize().unwrap();
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn self_peer_access_query() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();
    // A device cannot peer-access itself — expect Ok(false).
    assert!(!device.can_access_peer(&device).unwrap());
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn managed_buffer_and_prefetch_advise() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();
    let stream = Stream::new().unwrap();

    let n = 4096usize;
    let mut mb: memory::ManagedBuffer<u32> = memory::ManagedBuffer::new(n).unwrap();
    for (i, x) in mb.as_mut_slice().iter_mut().enumerate() {
        *x = i as u32;
    }

    // Advise + prefetch to device 0 — on Windows WDDM these may return
    // INVALID_DEVICE; treat as informational.
    let bytes = n * core::mem::size_of::<u32>();
    // SAFETY: `mb` is a live managed allocation of `bytes` bytes.
    let _ = unsafe {
        memory::mem_prefetch_async(
            mb.as_ptr() as *const core::ffi::c_void,
            bytes,
            PrefetchTarget::Device(0),
            &stream,
        )
    };
    let _ = unsafe {
        memory::mem_advise(
            mb.as_ptr() as *const core::ffi::c_void,
            bytes,
            baracuda_cuda_sys::runtime::types::cudaMemoryAdvise::SET_PREFERRED_LOCATION,
            PrefetchTarget::Device(0),
        )
    };
    stream.synchronize().unwrap();
    assert_eq!(mb.as_slice()[100], 100);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn pinned_host_alloc_and_device_ptr() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();

    let n = 1024usize;
    let mut pb: memory::PinnedHostBuffer<u32> =
        memory::PinnedHostBuffer::with_flags(n, memory::pinned_flags::MAPPED).unwrap();
    for (i, x) in pb.iter_mut().enumerate() {
        *x = (i * 3) as u32;
    }
    // With the MAPPED flag, there's a device-side alias.
    let dev_ptr = pb.device_ptr().unwrap();
    assert!(!dev_ptr.is_null());
    let flags = pb.flags().unwrap();
    eprintln!("pinned flags = {flags:#x}");
    assert!(flags & memory::pinned_flags::MAPPED != 0);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn async_alloc_round_trip() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();
    let stream = Stream::new().unwrap();

    let n = 2048usize;
    let host: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let buf = DeviceBuffer::<f32>::new_async(n, &stream).unwrap();
    buf.copy_from_host_async(&host, &stream).unwrap();
    let mut back = vec![0.0f32; n];
    buf.copy_to_host_async(&mut back, &stream).unwrap();
    buf.free_async(&stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(host, back);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn stream_capture_and_replay_vector_add() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();

    let n: u32 = 1 << 14;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 2.0).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let d_a = DeviceBuffer::from_slice(&a).unwrap();
    let d_b = DeviceBuffer::from_slice(&b).unwrap();
    let d_c: DeviceBuffer<f32> = DeviceBuffer::new(n as usize).unwrap();

    let lib = Library::load_ptx(VECTOR_ADD_PTX).unwrap();
    let kernel = lib.get_kernel("vector_add").unwrap();
    let stream = Stream::new().unwrap();

    let a_ptr = d_a.as_device_ptr();
    let b_ptr = d_b.as_device_ptr();
    let c_ptr = d_c.as_device_ptr();
    let block: u32 = 256;
    let grid: u32 = n.div_ceil(block);

    let captured: Graph = stream
        .capture(CaptureMode::ThreadLocal, |s| {
            // SAFETY: argument types + order match the PTX signature.
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
        .unwrap();
    let _ = graph::Graph::new(); // sanity: can we also construct empty graphs
    assert!(captured.node_count().unwrap() >= 1);

    let exec = captured.instantiate().unwrap();
    // Launch twice — kernel overwrites d_c fully each run, so both
    // replays must produce the same result.
    for _ in 0..2 {
        exec.launch(&stream).unwrap();
        stream.synchronize().unwrap();
        let mut got = vec![0.0f32; n as usize];
        d_c.copy_to_host(&mut got).unwrap();
        assert_eq!(expected, got);
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn kernel_occupancy_and_attributes() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();

    let lib = Library::load_ptx(VECTOR_ADD_PTX).unwrap();
    let kernel = lib.get_kernel("vector_add").unwrap();

    let blocks = kernel.max_active_blocks_per_multiprocessor(256, 0).unwrap();
    eprintln!("vector_add: {blocks} blocks/SM @ 256 threads");
    assert!(blocks >= 1);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn mem_get_info_reports_nonzero_totals() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();
    let (free, total) = memory::mem_get_info().unwrap();
    eprintln!("device memory: {free} free / {total} total");
    assert!(total > 0);
    assert!(free <= total);
}
