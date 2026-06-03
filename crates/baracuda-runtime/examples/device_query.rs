//! Enumerate CUDA devices via the Runtime API, print compute capability
//! and total global memory for each.
//!
//! Mirrors `nvidia-smi -L` (minus the rich properties), demonstrating
//! the Runtime-API view: a device is just an ordinal, the primary
//! context is implicit.
//!
//! Run with:
//!
//! ```text
//! cargo run --example device_query -p baracuda-runtime
//! ```

use baracuda_runtime::{
    device_synchronize, driver_version, query::device_properties, runtime_version, Device,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("CUDA driver version: {}", driver_version()?);
    println!("CUDA runtime version: {}", runtime_version()?);

    let devices = Device::all()?;
    println!("found {} CUDA device(s)", devices.len());

    for dev in &devices {
        // Switch current-thread device before querying properties:
        // `device_properties` reads via `cudaGetDeviceProperties`, which
        // is keyed off the ordinal we pass — but other Runtime calls
        // (memory queries, kernel launches) operate on the current
        // device, so it's good practice to bind first.
        dev.set_current()?;
        let props = device_properties(dev)?;
        let (cc_major, cc_minor) = dev.compute_capability()?;
        let sm_count = dev.multiprocessor_count()?;
        let warp = dev.warp_size()?;

        let gib = (props.total_global_memory_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
        println!(
            "  [{}] {:30}  cc {}.{}  {} SMs  warp {}  {:.2} GiB",
            dev.ordinal(),
            props.name,
            cc_major,
            cc_minor,
            sm_count,
            warp,
            gib,
        );
    }

    // Make sure all pending work on every device (none here) is done.
    device_synchronize()?;
    println!("OK");
    Ok(())
}
