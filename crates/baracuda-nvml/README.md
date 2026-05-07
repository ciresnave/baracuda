# baracuda-nvml

Safe Rust wrappers for **NVIDIA NVML** — driver-bundled GPU monitoring
and management. Same library `nvidia-smi` uses; covers everything from
basic device enumeration to fine-grained ECC and NVLink topology
queries.

NVML ships with the NVIDIA driver, so it's available wherever a driver
is installed (no separate CUDA toolkit needed).

```rust,no_run
use baracuda_nvml::Nvml;

# fn demo() -> Result<(), Box<dyn std::error::Error>> {
let nvml = Nvml::init()?;
for i in 0..nvml.device_count()? {
    let dev = nvml.device(i)?;
    println!("GPU {i}: {} ({} MiB free / {} MiB total)",
             dev.name()?,
             dev.memory_info()?.free / (1024 * 1024),
             dev.memory_info()?.total / (1024 * 1024));
}
# Ok(()) }
```

## Coverage

Comprehensive:

- **Enumeration & identity**: device count, by index / UUID / PCI bus
  ID; name, serial, UUID, minor number.
- **Memory**: free / total / used.
- **Telemetry**: temperature, power draw, fan speed, utilization
  (compute + memory + encoder + decoder).
- **Clocks**: current / max / applications / default; applications-clock
  set.
- **Power**: power draw, power limit get / set / range, power-management
  mode.
- **P-states**: power and performance.
- **Temperature thresholds.**
- **ECC**: per-location and total error counters; ECC mode get / set.
- **PCI**: info, link generation, link width, throughput.
- **NVLink**: state, version, capability, throughput.
- **Process enumeration**: compute + graphics processes per device.
- **Compute mode** (Default, Exclusive, Prohibited).
- **Persistence mode.**
- **Event sets**: register and wait on per-device events.
- **Field values**: arbitrary `nvmlFieldValue_t` queries.

Pairs with [`baracuda-nvml-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-nvml-sys`]: https://docs.rs/baracuda-nvml-sys
