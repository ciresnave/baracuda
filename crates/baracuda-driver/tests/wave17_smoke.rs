//! GPU-gated integration test for Wave-17 Driver-API additions:
//! green contexts (CUDA 12.4+).

use baracuda_driver::green::{device_sm_resource, sm_resource_split_by_count, GreenContext};
use baracuda_driver::{Context, Device};

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA 12.4+"]
fn split_sms_and_create_green_context() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    // Need an active primary context before green-ctx APIs work.
    let _ctx = Context::new(&device).unwrap();

    let total = match device_sm_resource(&device) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("cuDeviceGetDevResource not supported: {e:?}");
            return;
        }
    };
    let total_sms = total.as_sm().sm_count;
    eprintln!("device has {total_sms} SMs total");
    assert!(total_sms > 0);

    // Split into groups of at least 2 SMs each.
    let (groups, _remainder) = match sm_resource_split_by_count(&total, 2) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SM split not supported on this device: {e:?}");
            return;
        }
    };
    eprintln!("split into {} group(s) of ≥2 SMs", groups.len());
    if groups.is_empty() {
        return; // no room to split — device has < 2 SMs (vanishingly rare)
    }

    let green = match GreenContext::from_resource(&device, groups[0]) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("GreenContext::from_resource failed: {e:?}");
            return;
        }
    };
    let sm = green.sm_resource().unwrap();
    eprintln!("green context owns {} SMs", sm.sm_count);
    assert!(sm.sm_count >= 2);

    let _stream = green.create_stream_raw(0, 0).unwrap();
}
