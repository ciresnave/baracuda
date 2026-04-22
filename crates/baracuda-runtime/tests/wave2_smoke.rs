//! Runtime Wave 2: arrays + tex/surf sys, user objects, cooperative
//! launch, stream attach/attrs, IPC, device-flag config.

use core::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

use baracuda_runtime::ipc;
use baracuda_runtime::user_object::UserObject;
use baracuda_runtime::{get_device_flags, memory, Device, DeviceBuffer, Event, Graph, Stream};

static DESTRUCTOR_RAN: AtomicU32 = AtomicU32::new(0);

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn user_object_destructor_fires() {
    Device::from_ordinal(0).set_current().unwrap();
    DESTRUCTOR_RAN.store(0, Ordering::SeqCst);

    let uo = UserObject::new(
        || {
            DESTRUCTOR_RAN.fetch_add(1, Ordering::SeqCst);
        },
        1,
    )
    .unwrap();

    let graph = Graph::new().unwrap();
    graph.retain_user_object(&uo, 1, 0).unwrap();
    uo.release(1).unwrap();

    assert_eq!(DESTRUCTOR_RAN.load(Ordering::SeqCst), 0);

    drop(graph);
    assert_eq!(DESTRUCTOR_RAN.load(Ordering::SeqCst), 1);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn stream_attach_mem_and_copy_attrs() {
    Device::from_ordinal(0).set_current().unwrap();
    let s1 = Stream::new().unwrap();
    let s2 = Stream::new().unwrap();

    // Managed allocation — attach to s1 in SINGLE mode (flag=4).
    let n = 1024usize;
    let mb: memory::ManagedBuffer<u32> = memory::ManagedBuffer::new(n).unwrap();
    // SAFETY: managed-memory alloc.
    unsafe {
        s1.attach_mem_async(mb.as_ptr() as *mut c_void, n * 4, 4)
            .unwrap();
    }
    s1.synchronize().unwrap();

    // Copy attrs from s1 to s2 (no-op with defaults, exercises the path).
    s2.copy_attributes_from(&s1).unwrap();
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn device_flags_round_trip() {
    Device::from_ordinal(0).set_current().unwrap();
    // Calling set_device_flags after context creation typically returns
    // cudaErrorSetOnActiveProcess. We exercise the safe wrapper and
    // tolerate either outcome.
    match baracuda_runtime::set_device_flags(0) {
        Ok(()) => eprintln!("set_device_flags accepted"),
        Err(e) => eprintln!("set_device_flags post-context: {e:?}"),
    }
    let flags = get_device_flags().unwrap();
    eprintln!("device flags = {flags:#x}");
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn ipc_symbols_resolve_gracefully() {
    Device::from_ordinal(0).set_current().unwrap();

    // IPC is Linux-primary; Windows returns NOT_SUPPORTED for events
    // and sometimes OK for mem. We just exercise the paths.
    let event = Event::no_timing().unwrap();
    match ipc::event_get_handle(&event) {
        Ok(h) => eprintln!("IPC event handle, first bytes: {:?}", &h.reserved[..8]),
        Err(e) => eprintln!("cudaIpcGetEventHandle: {e:?}"),
    }

    let buf: DeviceBuffer<u32> = DeviceBuffer::new(1024).unwrap();
    match ipc::mem_get_handle(buf.as_raw()) {
        Ok(h) => eprintln!("IPC mem handle, first bytes: {:?}", &h.reserved[..8]),
        Err(e) => eprintln!("cudaIpcGetMemHandle: {e:?}"),
    }
}

#[test]
fn runtime_wave2_symbols_resolve() {
    if let Ok(r) = baracuda_cuda_sys::runtime() {
        // Just verify no panic on lazy-resolution.
        let _ = r.cuda_malloc_array();
        let _ = r.cuda_free_array();
        let _ = r.cuda_memcpy_2d_to_array();
        let _ = r.cuda_memcpy_2d_from_array();
        let _ = r.cuda_create_texture_object();
        let _ = r.cuda_destroy_texture_object();
        let _ = r.cuda_create_surface_object();
        let _ = r.cuda_destroy_surface_object();
        let _ = r.cuda_user_object_create();
        let _ = r.cuda_user_object_retain();
        let _ = r.cuda_user_object_release();
        let _ = r.cuda_graph_retain_user_object();
        let _ = r.cuda_graph_release_user_object();
        let _ = r.cuda_launch_cooperative_kernel();
        let _ = r.cuda_stream_attach_mem_async();
        let _ = r.cuda_stream_get_attribute();
        let _ = r.cuda_stream_set_attribute();
        let _ = r.cuda_stream_copy_attributes();
        let _ = r.cuda_ipc_get_event_handle();
        let _ = r.cuda_ipc_open_event_handle();
        let _ = r.cuda_ipc_get_mem_handle();
        let _ = r.cuda_ipc_open_mem_handle();
        let _ = r.cuda_ipc_close_mem_handle();
        let _ = r.cuda_set_device_flags();
        let _ = r.cuda_get_device_flags();
    }
}
