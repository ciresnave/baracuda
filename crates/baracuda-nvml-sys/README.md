# baracuda-nvml-sys

Raw FFI bindings + dynamic loader for **NVIDIA NVML** — driver-bundled
GPU monitoring and management (the same library `nvidia-smi` uses).

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libnvidia-ml.so` / `nvml.dll`.

**Most users want [`baracuda-nvml`]** — that crate exposes typed
`Device` handles, ECC controls, P-state / clock / power-limit
operations, NVLink topology queries, and event watchers.

## What's exposed

- Device enumeration, identification (UUID / PCI / index / minor
  number), name / serial.
- Memory / temperature / power / fan / utilization queries.
- Clock info (current, max, applications, default) and applications-clock
  set.
- Power-limit get / set / range.
- P-states (power + performance).
- Temperature thresholds.
- ECC error counters (per-location + total) and ECC mode set.
- PCI info, link generation, link width, throughput.
- NVLink state, version, capability.
- Process enumeration (compute + graphics).
- Compute mode, persistence mode, GOM.
- Event sets — register / wait.
- Field values.

NVML is bundled with the NVIDIA driver, so it's available wherever a
driver is installed (no separate CUDA toolkit needed).

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-nvml`]: https://docs.rs/baracuda-nvml
