# baracuda-cupti-sys

Raw FFI bindings + dynamic loader for **NVIDIA CUPTI** — the CUDA
Profiling Tools Interface used by Nsight Systems / Compute and any
custom profiler that wants per-API-call traces, performance counters,
and kernel-level metrics.

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libcupti.so` / `cupti64_*.dll`.

**Most users want [`baracuda-cupti`]** — that crate exposes the
Activity API, Callback API (`Subscriber` RAII), Event / Metric APIs,
and the Profiler Host API in idiomatic Rust.

## What's exposed

- **Activity API**: record buffers, walker for parsing records.
- **Callback API**: subscribe / unsubscribe to driver / runtime / NVTX
  / resource callbacks.
- **Event API**: `cuptiEventGroup_*`, device / domain enumeration,
  read / read-counter functions.
- **Metric API**: id-by-name, attributes, values.
- **Profiler Host API**: initialize / de-initialize, sessions,
  configurations, passes, range push / pop, flush, counter availability.
- `cuptiGetResultString` helper.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cupti`]: https://docs.rs/baracuda-cupti
