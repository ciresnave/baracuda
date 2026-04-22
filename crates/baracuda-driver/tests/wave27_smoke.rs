//! GPU-gated integration tests for Wave-27 Driver-API additions:
//! v2 prefetch/advise + VMM reverse lookups.

use baracuda_cuda_sys::types::CUmem_advise;
use baracuda_driver::memory::{
    self, allocation_properties_from_handle, mem_advise_v2, mem_prefetch_v2,
    retain_allocation_handle, PrefetchTarget,
};
use baracuda_driver::vmm::{AccessFlags, AddressRange, MappedRange, PhysicalMemory};
use baracuda_driver::{vmm, Context, Device, ManagedBuffer, Stream};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn managed_prefetch_and_advise_v2() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let n = 4096usize;
    let buf = ManagedBuffer::<u32>::new(&ctx, n).unwrap();

    // Prefetch to the device, then advise a preferred location of device.
    // Either may be unsupported on WDDM / consumer Windows — treat that
    // as informational.
    let bytes = n * core::mem::size_of::<u32>();
    match mem_prefetch_v2(buf.as_raw(), bytes, PrefetchTarget::Device(0), &stream) {
        Ok(()) => eprintln!("prefetch_v2(device) OK"),
        Err(e) => eprintln!("prefetch_v2(device) unsupported: {e:?}"),
    }
    match mem_advise_v2(
        buf.as_raw(),
        bytes,
        CUmem_advise::SET_PREFERRED_LOCATION,
        PrefetchTarget::Device(0),
    ) {
        Ok(()) => eprintln!("advise_v2(SET_PREFERRED_LOCATION, device) OK"),
        Err(e) => eprintln!("advise_v2 unsupported: {e:?}"),
    }

    stream.synchronize().unwrap();

    // A no-op use silences unused imports on branches that early-return.
    let _ = memory::mem_get_info;
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn vmm_retain_handle_round_trip() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let gran = vmm::allocation_granularity(&device, false).unwrap();
    let size = gran;

    let range = AddressRange::reserve(&ctx, size, 0).unwrap();
    let physical = PhysicalMemory::create(&ctx, &device, size).unwrap();
    let mapped = MappedRange::new(&range, &physical, 0).unwrap();
    mapped.set_access(&device, AccessFlags::ReadWrite).unwrap();

    // Retain the underlying handle by address.
    let retained = retain_allocation_handle(mapped.as_raw()).unwrap();
    assert_eq!(
        retained,
        physical.as_raw(),
        "retain_allocation_handle should return the same handle cuMemCreate gave us"
    );

    // Query properties — allocation type should be PINNED (what VMM uses).
    let props = allocation_properties_from_handle(retained).unwrap();
    assert_eq!(
        props.type_,
        baracuda_cuda_sys::types::CUmemAllocationType::PINNED
    );

    // Drop our extra ref so the handle's internal refcount drops by 1;
    // `physical`'s Drop handles the original ref.
    let d = baracuda_cuda_sys::driver().unwrap();
    let release = d.cu_mem_release().unwrap();
    let rc = unsafe { release(retained) };
    assert!(rc.is_success());
}
