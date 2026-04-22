//! GPU-gated integration test for Wave-20 Driver-API additions:
//! IPC handle export. Only validates that symbols resolve; IPC is
//! Linux-only (Windows returns NOT_SUPPORTED).

use baracuda_driver::ipc;
use baracuda_driver::{Context, Device, DeviceBuffer, Event};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn ipc_get_event_and_mem_handles_resolve() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let event = Event::new(&ctx).unwrap();
    match ipc::event_get_handle(&event) {
        Ok(h) => eprintln!("event IPC handle: first bytes {:?}", &h.reserved[..8]),
        Err(e) => eprintln!("cuIpcGetEventHandle unsupported on this platform: {e:?}"),
    }

    let buf: DeviceBuffer<u32> = DeviceBuffer::new(&ctx, 1024).unwrap();
    match ipc::mem_get_handle(buf.as_raw()) {
        Ok(h) => eprintln!("mem IPC handle: first bytes {:?}", &h.reserved[..8]),
        Err(e) => eprintln!("cuIpcGetMemHandle unsupported on this platform: {e:?}"),
    }
}
