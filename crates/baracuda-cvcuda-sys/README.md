# baracuda-cvcuda-sys

Raw FFI bindings + dynamic loader for **NVIDIA CV-CUDA** — a GPU-native
computer-vision operator catalog (resize, warp, filter, color space
conversion, morphology, edge detection, ...).

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libcvcuda.so`.

**Most users want [`baracuda-cvcuda`]** — that crate exposes typed
operator handles (`Resize`, `Warp`, `Gaussian`, `Canny`, ...) with
stream-async execution.

## Platform support

CV-CUDA is **Linux-only**. The loader fails fast with a clear error on
Windows / macOS.

## What's exposed

- Operator C ABI for the ~30 operators wrapped by baracuda-cvcuda:
  geometric (Resize, PillowResize, Warp*, Remap, Rotate, Crop,
  CopyMakeBorder), filters (Gaussian, Median, Average, Laplacian,
  Bilateral, Motion, Conv2D), morphology, edges (Canny), thresholds,
  color (ColorTwist, BrightnessContrast, GammaContrast), stats,
  composite, misc.
- Tensor / image-batch types.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cvcuda`]: https://docs.rs/baracuda-cvcuda
