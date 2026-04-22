//! GPU-gated NVML smoke test. Requires the NVIDIA driver (not the CUDA toolkit).

use baracuda_nvml::Nvml;

#[test]
#[ignore = "requires an NVIDIA driver"]
fn query_device_health() {
    let nvml = Nvml::init().expect("nvmlInit");
    let driver = nvml.driver_version().expect("driver version");
    eprintln!("driver: {driver}");
    assert!(!driver.is_empty());

    let devices = nvml.devices().expect("devices");
    assert!(!devices.is_empty(), "no NVML devices visible");

    for (i, dev) in devices.iter().enumerate() {
        let name = dev.name().expect("name");
        let mem = dev.memory_info().expect("memory_info");
        let temp = dev.temperature().unwrap_or(0);
        let power = dev.power_usage_watts().unwrap_or(0.0);
        let util = dev.utilization().unwrap_or_default();
        eprintln!(
            "[{i}] {name}: {temp}°C, {power:.1}W, util gpu={}% mem={}%, {} MiB / {} MiB",
            util.gpu,
            util.memory,
            mem.used / (1024 * 1024),
            mem.total / (1024 * 1024),
        );
        assert!(!name.is_empty());
        assert!(mem.total > 0, "device should report nonzero total memory");
    }
}
