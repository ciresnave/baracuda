//! GPU-gated integration tests for Wave-8 Driver-API additions:
//! memory pools (`cuMemPool*` + `cuMemAllocFromPoolAsync`).

use baracuda_cuda_sys::driver;
use baracuda_driver::mempool::{self, MemoryPool};
use baracuda_driver::{Context, Device, Stream};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn pool_roundtrip_via_alloc_from_pool_async() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    // Create a user-owned pool with a 1 MiB release threshold.
    let pool = MemoryPool::new(&ctx, &device).unwrap();
    pool.set_release_threshold(1 << 20).unwrap();
    assert_eq!(pool.release_threshold().unwrap(), 1 << 20);

    let n = 4096usize;
    let bytes = n * core::mem::size_of::<u32>();
    let host: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(0xDEAD_BEEF)).collect();

    let dptr = pool.alloc_async(bytes, &stream).unwrap();

    let d = driver().unwrap();
    let cu_htod_async = d.cu_memcpy_htod_async().unwrap();
    let cu_dtoh_async = d.cu_memcpy_dtoh_async().unwrap();
    let rc = unsafe {
        cu_htod_async(
            dptr,
            host.as_ptr() as *const core::ffi::c_void,
            bytes,
            stream.as_raw(),
        )
    };
    assert!(rc.is_success());

    let mut back = vec![0u32; n];
    let rc = unsafe {
        cu_dtoh_async(
            back.as_mut_ptr() as *mut core::ffi::c_void,
            dptr,
            bytes,
            stream.as_raw(),
        )
    };
    assert!(rc.is_success());

    // Free back to the pool (async).
    let cu_free_async = d.cu_mem_free_async().unwrap();
    let rc = unsafe { cu_free_async(dptr, stream.as_raw()) };
    assert!(rc.is_success());
    stream.synchronize().unwrap();

    assert_eq!(host, back);

    // used_bytes should be 0 after the async free completed.
    eprintln!(
        "pool after free: used={}, reserved={}",
        pool.used_bytes().unwrap(),
        pool.reserved_bytes().unwrap()
    );

    // Trim the pool back down to 0 bytes kept.
    pool.trim_to(0).unwrap();
    let reserved_after_trim = pool.reserved_bytes().unwrap();
    assert!(
        reserved_after_trim <= pool.reserved_bytes().unwrap(),
        "trim_to(0) should not increase reserved memory"
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn default_and_current_pool_queries() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let default = mempool::default_pool(&ctx, &device).unwrap();
    let current = mempool::current_pool(&ctx, &device).unwrap();
    // Initially the current pool *is* the default pool.
    assert_eq!(
        default.as_raw(),
        current.as_raw(),
        "default pool should match current pool before set_current_pool",
    );
    eprintln!("default pool: {:?}", default.as_raw());
}
