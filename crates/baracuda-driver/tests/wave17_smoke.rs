//! GPU-gated integration test for Wave-17 Driver-API additions:
//! green contexts (CUDA 12.4+).

use baracuda_driver::green::{GreenContext, device_sm_resource, sm_resource_split_by_count};
use baracuda_driver::{Context, Device, require};

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA 12.4+"]
fn split_sms_and_create_green_context() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    // Need an active primary context before green-ctx APIs work.
    let _ctx = Context::new(&device).unwrap();

    let total = require!(
        device_sm_resource(&device),
        "cuDeviceGetDevResource — green-context SM query (CUDA 12.4+)"
    );
    let total_sms = total.as_sm().sm_count;
    eprintln!("device has {total_sms} SMs total");
    assert!(total_sms > 0);

    // Split into groups of at least 2 SMs each.
    let (groups, _remainder) = require!(
        sm_resource_split_by_count(&total, 2),
        "cuDevSmResourceSplitByCount — SM-resource split (CUDA 12.4+)"
    );
    eprintln!("split into {} group(s) of ≥2 SMs", groups.len());
    // On any real GPU (≥2 SMs) the split yields at least one group; an empty
    // result is an anomaly, not a skip — require! it so the box fails loud.
    require!(
        (!groups.is_empty()).then_some(()),
        "SM-resource split produced ≥1 group (device has ≥2 SMs)"
    );

    let green = require!(
        GreenContext::from_resource(&device, groups[0]),
        "cuGreenCtxCreate — green context from SM resource (CUDA 12.4+)"
    );
    let sm = green.sm_resource().unwrap();
    eprintln!("green context owns {} SMs", sm.sm_count);
    assert!(sm.sm_count >= 2);

    // cuGreenCtxStreamCreate requires CU_STREAM_NON_BLOCKING (0x01); flags=0
    // (CU_STREAM_DEFAULT) is rejected with CUDA_ERROR_INVALID_VALUE. (Pre-existing
    // test bug surfaced on-device once the silent-skip migration let this test
    // run to completion — orthogonal to the require! change above.)
    let _stream = green.create_stream_raw(0x01, 0).unwrap();
}
