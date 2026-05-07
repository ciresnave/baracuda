# baracuda-npp-sys

Raw FFI bindings + dynamic loader for **NVIDIA NPP** (NVIDIA Performance
Primitives) — a large catalog of image, signal, and statistical
processing functions on the GPU.

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libnppc.so` / `libnppi*.so` /
`libnpps.so` / `nppc64_*.dll` etc.

**Most users want [`baracuda-npp`]** — that crate exposes typed safe
wrappers for the workhorse subset. NPP has *thousands* of function
variants (every dtype × channel-count × in-place-or-not combination);
the safe wrapper covers what most callers actually use, with on-request
expansion for the rest.

## What's exposed

- All NPP context / stream-context types.
- Function pointer types for the wrapped subset (arithmetic, geometric
  transforms, filters, statistics, etc.) — additional families load on
  demand as the safe wrapper expands.
- Image, signal, and statistical primitives.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-npp`]: https://docs.rs/baracuda-npp
