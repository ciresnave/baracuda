//! GPU-gated integration tests for Wave-1 Driver-API additions:
//! occupancy, unified memory, context queries + limits, peer query,
//! pointer attributes, stream priority, stream host callback,
//! event record-with-flags, primary-context state.

use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

use baracuda_cuda_sys::types::{CUfunc_cache, CUlimit, CUstream_flags};
use baracuda_driver::{
    mem_get_info, occupancy, pointer, Context, Device, DeviceBuffer, ManagedAttach, ManagedBuffer,
    MemAdvise, Module, Stream,
};

const PTX: &str = include_str!("kernels/vector_add.ptx");

fn setup() -> baracuda_driver::Result<(Device, Context, Stream)> {
    baracuda_driver::init()?;
    let device = Device::get(0)?;
    let ctx = Context::new(&device)?;
    let stream = Stream::new(&ctx)?;
    Ok((device, ctx, stream))
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn occupancy_max_active_blocks_and_potential_size() {
    let (_dev, ctx, _stream) = setup().expect("setup");
    let module = Module::load_ptx(&ctx, PTX).unwrap();
    let kernel = module.get_function("vector_add").unwrap();

    let blocks_per_sm = occupancy::max_active_blocks_per_multiprocessor(&kernel, 256, 0).unwrap();
    eprintln!("blocks per SM @ blockDim=256: {blocks_per_sm}");
    assert!(blocks_per_sm >= 1, "no blocks fit per SM? {blocks_per_sm}");

    let (min_grid, optimal_block) = occupancy::max_potential_block_size(&kernel, 0, 0).unwrap();
    eprintln!("optimal block size: {optimal_block}, covers {min_grid} blocks to saturate");
    assert!((32..=1024).contains(&optimal_block));
    assert!(min_grid >= 1);

    let smem = occupancy::available_dynamic_smem_per_block(&kernel, blocks_per_sm, 256).unwrap();
    eprintln!("dynamic smem budget per block @ {blocks_per_sm}×256: {smem} bytes");
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn mem_get_info_reports_sensible_sizes() {
    let (_dev, _ctx, _stream) = setup().expect("setup");
    let (free, total) = mem_get_info().expect("cuMemGetInfo");
    eprintln!(
        "mem: {:.2} GiB free / {:.2} GiB total",
        free as f64 / (1024.0 * 1024.0 * 1024.0),
        total as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    assert!(total > 0);
    assert!(free <= total);
    assert!(total > 512 * 1024 * 1024, "tiny total mem? {total}");
}

#[test]
#[ignore = "requires an NVIDIA GPU with managed-memory support"]
fn managed_buffer_round_trip() {
    let (dev, ctx, stream) = setup().expect("setup");

    let n = 1024;
    let mut mb = ManagedBuffer::<f32>::new_with_flags(&ctx, n, ManagedAttach::Global).unwrap();
    // SAFETY: no kernel in flight; we're the only accessor.
    unsafe {
        let slice = mb.as_host_slice_mut();
        for (i, v) in slice.iter_mut().enumerate() {
            *v = i as f32 * 2.0;
        }
    }

    // `SetAccessedBy` is optional on Windows/WDDM — the driver may return
    // `INVALID_DEVICE` when concurrent-managed-access isn't supported.
    // Ignore the error; the actual migration still works.
    let _ = mb.advise(MemAdvise::SetAccessedBy, &dev);
    // Prefetching can likewise fail on WDDM with no-ops; ignore and go on.
    let _ = mb.prefetch_async(&dev, &stream);
    stream.synchronize().unwrap();

    // SAFETY: no kernel is accessing this buffer.
    unsafe {
        let slice = mb.as_host_slice();
        for (i, v) in slice.iter().enumerate() {
            assert_eq!(*v, i as f32 * 2.0, "managed mismatch at {i}");
        }
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn pointer_attribute_queries() {
    let (dev, ctx, _stream) = setup().expect("setup");

    // Regular device allocation.
    let buf: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, 256).unwrap();
    let ptr = buf.as_raw();
    let ty = pointer::memory_type(ptr).expect("memory_type");
    assert_eq!(ty, pointer::MemoryType::Device);
    assert!(!pointer::is_managed(ptr).unwrap());
    assert_eq!(pointer::device_ordinal(ptr).unwrap(), dev.ordinal());

    // Managed allocation. On Windows/WDDM the driver may report managed
    // memory as `Device` rather than `Unified`; only the `is_managed` flag
    // is portable. Accept both so the test passes on WDDM + native Linux.
    let mb = ManagedBuffer::<u32>::new(&ctx, 256).unwrap();
    let ty = pointer::memory_type(mb.as_raw()).unwrap();
    assert!(
        matches!(
            ty,
            pointer::MemoryType::Unified | pointer::MemoryType::Device
        ),
        "managed pointer reported unexpected type: {ty:?}"
    );
    assert!(pointer::is_managed(mb.as_raw()).unwrap());
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn context_limits_and_cache_config() {
    let (_dev, ctx, _stream) = setup().expect("setup");
    ctx.set_current().unwrap();

    // Read current values.
    let api = ctx.api_version().unwrap();
    let flags = Context::current_flags().unwrap();
    let cur_dev = Context::current_device().unwrap();
    eprintln!(
        "api={api} flags={flags:#x} current_device_ordinal={}",
        cur_dev.ordinal()
    );
    assert_eq!(cur_dev.ordinal(), 0);

    let stack_before = Context::get_limit(CUlimit::STACK_SIZE).unwrap();
    let printf_before = Context::get_limit(CUlimit::PRINTF_FIFO_SIZE).unwrap();
    eprintln!("stack size before: {stack_before}, printf fifo: {printf_before}");
    assert!(stack_before > 0);

    // Try setting the printf FIFO size. Some limits are not writable on every
    // platform — ignore the error silently and just check the setter call path.
    let _ = Context::set_limit(CUlimit::PRINTF_FIFO_SIZE, printf_before * 2);

    let cache_before = Context::cache_config().unwrap();
    // Try setting PreferL1 (may be a no-op on modern GPUs but should return success).
    Context::set_cache_config(CUfunc_cache::PREFER_L1).unwrap();
    let cache_after = Context::cache_config().unwrap();
    eprintln!("cache config before/after: {cache_before} / {cache_after}");

    let (least, greatest) = Context::stream_priority_range().unwrap();
    eprintln!("stream priority range: {least}..={greatest}");
    // On NVIDIA GPUs this is `(least, greatest)` with greatest <= least
    // (because lower number = higher priority).
    assert!(greatest <= least);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn device_self_peer_query() {
    let (dev, _ctx, _stream) = setup().expect("setup");
    // A device cannot access itself as a peer (makes no sense).
    let can = dev.can_access_peer(&dev).unwrap();
    assert!(!can, "device reported peer-access to itself: {can}");
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn stream_priority_and_flags() {
    let (_dev, ctx, _stream) = setup().expect("setup");
    let (least, greatest) = Context::stream_priority_range().unwrap();

    let hi = Stream::with_priority(&ctx, CUstream_flags::NON_BLOCKING, greatest).unwrap();
    let lo = Stream::with_priority(&ctx, CUstream_flags::DEFAULT, least).unwrap();

    assert_eq!(hi.priority().unwrap(), greatest);
    assert_eq!(lo.priority().unwrap(), least);
    assert_eq!(hi.flags().unwrap(), CUstream_flags::NON_BLOCKING);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn stream_host_callback_runs() {
    let (_dev, _ctx, stream) = setup().expect("setup");
    let counter = Arc::new(AtomicU32::new(0));
    let c2 = counter.clone();
    stream
        .launch_host_func(move || {
            c2.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn event_record_with_flags() {
    let (_dev, ctx, stream) = setup().expect("setup");
    let e = baracuda_driver::Event::new(&ctx).unwrap();
    e.record_with_flags(&stream, 0).expect("record_with_flags");
    stream.synchronize().unwrap();
    assert!(e.is_complete().unwrap());
}
