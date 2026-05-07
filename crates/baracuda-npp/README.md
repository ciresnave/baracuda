# baracuda-npp

Safe Rust wrappers for **NVIDIA NPP** (NVIDIA Performance Primitives)
— a GPU library of image, signal, and statistical processing functions.

## Status: workhorse subset

NPP exposes *thousands* of function variants (every dtype × channel
count × in-place vs out-of-place combination). This crate covers the
workhorse subset most callers need plus the version + context helpers.
Additional families wrap on request.

Today's surface:

- `version()` query.
- Sample arithmetic primitives.
- StreamContext binding.

Pairs with [`baracuda-npp-sys`] for the raw FFI surface — if you need
an NPP function not yet wrapped here, the sys crate exposes the full
catalog and can be called directly.

Open an issue if you need a specific NPP family wrapped.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-npp-sys`]: https://docs.rs/baracuda-npp-sys
