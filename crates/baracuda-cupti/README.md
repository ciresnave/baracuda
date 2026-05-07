# baracuda-cupti

Safe Rust wrappers for **NVIDIA CUPTI** — the CUDA Profiling Tools
Interface. Used by Nsight and any custom profiler that wants per-API-call
traces, kernel-level performance counters, or SASS-level metrics.

## Coverage

All four CUPTI sub-APIs:

- **Activity API**: record buffer setup, walker for parsing every
  `CUpti_Activity*` record kind (kernel, memcpy, runtime, driver, NVTX,
  device, context, ...).
- **Callback API**: `Subscriber` RAII; subscribe / unsubscribe per
  domain (driver, runtime, NVTX, resource, synchronize, etc.).
- **Event API**: `Group` RAII for per-device / per-domain event groups,
  read counters, enumerate available events.
- **Metric API**: id-by-name lookup, attribute query, value computation
  (combine event values into the requested metric).
- **Profiler Host API**: initialize / de-initialize, sessions, configs,
  passes (begin / end), `enable` / `disable`, range push / pop, flush,
  counter availability.
- `cuptiGetResultString` helper for human-readable error messages.

## Stability

CUPTI's API is large and changes more frequently than the rest of CUDA.
This crate tracks the stable "host-side profiling" entry points; if
you're writing a SASS-level kernel profiler you may need APIs we haven't
wrapped yet — open an issue or use [`baracuda-cupti-sys`] directly for
the gap.

Pairs with [`baracuda-cupti-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cupti-sys`]: https://docs.rs/baracuda-cupti-sys
