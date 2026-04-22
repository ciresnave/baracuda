//! Runtime Wave 3: batch mem ops, conditional graph handles,
//! driver-entry-point bridge, occupancy-with-flags + available-smem.

use baracuda_cuda_sys::types::{CUdeviceptr, CUstreamBatchMemOpParams, CUstreamWriteValue_flags};

use baracuda_runtime::driver_entry::{self, DriverEntryPoint};
use baracuda_runtime::{Device, DeviceBuffer, Graph, Library, Stream};

const VECTOR_ADD_PTX: &str = include_str!("kernels/vector_add.ptx");

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn stream_batch_mem_op_write_and_readback() {
    Device::from_ordinal(0).set_current().unwrap();
    let stream = Stream::new().unwrap();
    let buf: DeviceBuffer<u32> = DeviceBuffer::new(2).unwrap();

    // Two WRITE_VALUE_32 entries, one per slot.
    let slot0 = CUdeviceptr(buf.as_device_ptr());
    let slot1 = CUdeviceptr(buf.as_device_ptr() + 4);
    let mut params = [
        CUstreamBatchMemOpParams::write_value_32(
            slot0,
            0xDEAD_BEEF,
            CUstreamWriteValue_flags::DEFAULT,
        ),
        CUstreamBatchMemOpParams::write_value_32(
            slot1,
            0xCAFE_F00D,
            CUstreamWriteValue_flags::DEFAULT,
        ),
    ];

    // cudaStreamBatchMemOp is frequently not exported from the
    // Windows cudart DLL (the op is driver-only on WDDM). Skip cleanly
    // if the symbol is missing.
    match unsafe { stream.batch_mem_op(&mut params, 0) } {
        Ok(()) => {}
        Err(baracuda_runtime::Error::Loader(_)) => {
            eprintln!("cudaStreamBatchMemOp unavailable on this build — skipping");
            return;
        }
        Err(e) => panic!("batch_mem_op: {e:?}"),
    }
    stream.synchronize().unwrap();

    let mut out = [0u32; 2];
    buf.copy_to_host(&mut out).unwrap();
    assert_eq!(out, [0xDEAD_BEEF, 0xCAFE_F00D]);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn occupancy_with_flags_and_available_smem() {
    Device::from_ordinal(0).set_current().unwrap();
    let lib = Library::load_ptx(VECTOR_ADD_PTX).unwrap();
    let kernel = lib.get_kernel("vector_add").unwrap();

    // flags = 0 (default) should match the non-flagged call on most devices.
    let a = kernel.max_active_blocks_per_multiprocessor(128, 0).unwrap();
    let b = kernel
        .max_active_blocks_per_multiprocessor_with_flags(128, 0, 0)
        .unwrap();
    eprintln!("occupancy: plain={a}, with_flags={b}");
    assert!(a > 0);
    assert!(b > 0);

    // Available dynamic smem for a single 128-thread block running per SM.
    let dyn_smem = kernel.available_dynamic_smem_per_block(1, 128).unwrap();
    eprintln!("available dynamic smem per block (1 × 128): {dyn_smem} B");
    assert!(dyn_smem > 0);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn driver_entry_point_resolves_known_symbol() {
    Device::from_ordinal(0).set_current().unwrap();
    let DriverEntryPoint { fn_ptr, status } =
        driver_entry::driver_entry_point("cuDriverGetVersion", 0).unwrap();
    eprintln!("cuDriverGetVersion via runtime: status={status}, ptr={fn_ptr:?}");
    assert_eq!(status, 0);
    assert!(!fn_ptr.is_null());
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn conditional_handle_requires_12_3() {
    Device::from_ordinal(0).set_current().unwrap();
    let graph = Graph::new().unwrap();
    match graph.conditional_handle_create(0, 0) {
        Ok(h) => {
            eprintln!("conditional handle: {h:#x}");
            assert_ne!(h, 0);
        }
        Err(baracuda_runtime::Error::FeatureNotSupported { api, since }) => {
            eprintln!("{api} not supported on this driver (needs {since})");
        }
        Err(e) => panic!("conditional_handle_create: {e:?}"),
    }
}

#[test]
fn runtime_wave3_symbols_resolve() {
    if let Ok(r) = baracuda_cuda_sys::runtime() {
        let _ = r.cuda_stream_batch_mem_op();
        let _ = r.cuda_graph_add_node();
        let _ = r.cuda_graph_conditional_handle_create();
        let _ = r.cuda_get_driver_entry_point();
        let _ = r.cuda_occupancy_max_active_blocks_per_multiprocessor_with_flags();
        let _ = r.cuda_occupancy_available_dynamic_smem_per_block();
    }
}
