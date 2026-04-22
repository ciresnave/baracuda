//! GPU-gated integration test for Wave-7 Driver-API additions:
//! the Virtual Memory Management (VMM) API round-trip.

use baracuda_cuda_sys::driver;
use baracuda_driver::vmm::{self, AccessFlags, AddressRange, MappedRange, PhysicalMemory};
use baracuda_driver::{Context, Device};

#[test]
#[ignore = "requires an NVIDIA GPU + VMM support"]
fn vmm_roundtrip_via_memcpy() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    // Probe granularity — VMM allocations must be multiples of this.
    let gran_min = vmm::allocation_granularity(&device, false).unwrap();
    let gran_rec = vmm::allocation_granularity(&device, true).unwrap();
    eprintln!("VMM granularity: min={gran_min}B, recommended={gran_rec}B");
    assert!(gran_min > 0);

    // Round our desired size up to the granularity. Aim for ~1 MB worth of
    // u32s but let granularity win.
    let desired = 1024 * 1024usize;
    let size_bytes = desired.div_ceil(gran_min) * gran_min;
    let num_u32 = size_bytes / core::mem::size_of::<u32>();
    eprintln!("reserving {size_bytes}B for {num_u32} u32s");

    // 1) Reserve VA space.
    let range = AddressRange::reserve(&ctx, size_bytes, 0).unwrap();
    // 2) Create physical backing.
    let physical = PhysicalMemory::create(&ctx, &device, size_bytes).unwrap();
    // 3) Map.
    let mapped = MappedRange::new(&range, &physical, 0).unwrap();
    // 4) Grant RW access.
    mapped.set_access(&device, AccessFlags::ReadWrite).unwrap();

    // Use it like a regular device pointer: HtoD, DtoH round trip.
    let host: Vec<u32> = (0..num_u32 as u32)
        .map(|i| i.wrapping_mul(0x9E37_79B1))
        .collect();

    let d = driver().unwrap();
    let cu_htod = d.cu_memcpy_htod().unwrap();
    let cu_dtoh = d.cu_memcpy_dtoh().unwrap();
    let rc_in = unsafe {
        cu_htod(
            mapped.as_raw(),
            host.as_ptr() as *const core::ffi::c_void,
            size_bytes,
        )
    };
    assert!(rc_in.is_success(), "HtoD failed: {rc_in:?}");

    let mut back = vec![0u32; num_u32];
    let rc_out = unsafe {
        cu_dtoh(
            back.as_mut_ptr() as *mut core::ffi::c_void,
            mapped.as_raw(),
            size_bytes,
        )
    };
    assert!(rc_out.is_success(), "DtoH failed: {rc_out:?}");
    ctx.synchronize().unwrap();

    assert_eq!(host, back);

    // Explicit drop order: mapped first (cuMemUnmap), then physical
    // (cuMemRelease), then range (cuMemAddressFree). Rust drops in
    // reverse-declaration order so this is what happens automatically;
    // we drop explicitly to make the test's intent clear.
    drop(mapped);
    drop(physical);
    drop(range);
    drop(ctx);
}
