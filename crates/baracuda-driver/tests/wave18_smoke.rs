//! GPU-gated integration test for Wave-18 Driver-API additions:
//! multicast objects. End-to-end multicast requires NVSwitch fabric, so
//! we only verify the symbols resolve and the granularity query returns.

#![cfg(feature = "device-tests")]

use baracuda_driver::multicast;
use baracuda_driver::multicast::MulticastObject;
use baracuda_driver::{Context, Device};

#[test]
fn multicast_granularity_and_create() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let _ctx = Context::new(&device).unwrap();

    // Granularity query on a 1-device group is fine on any system — it's
    // a pure sizing calculation that the driver serves from static data.
    match multicast::multicast_granularity(1, 1 << 21, false) {
        Ok(g) => {
            eprintln!("multicast min granularity (1 device, 2 MiB) = {g}");
            // A granularity is a byte count the driver rounds sizes to: nonzero and a
            // power of two.
            assert!(
                g > 0 && g.is_power_of_two(),
                "bad multicast granularity {g}"
            );
        }
        Err(e) => {
            eprintln!("multicast_granularity not supported: {e:?}");
            return;
        }
    }

    // Creating the object itself fails on single-GPU without NVSwitch;
    // we accept either outcome.
    match MulticastObject::new(1, 1 << 21) {
        Ok(mc) => {
            eprintln!("multicast object created: {:#x}", mc.as_raw());
            assert_ne!(mc.as_raw(), 0, "created multicast object has a null handle");
        }
        Err(e) => {
            eprintln!("cuMulticastCreate rejected as expected on non-NVSwitch: {e:?}");
        }
    }
}
