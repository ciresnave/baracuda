# baracuda-cvcuda

Safe Rust wrappers for **NVIDIA CV-CUDA** — a GPU-native computer-vision
operator catalog. Designed for high-throughput image / video pipelines
where every operator runs on-GPU and stays on-GPU.

## Status: ~30 operators wrapped

Workhorse coverage across the major operator families:

- **Geometric**: Resize, PillowResize, Warp{Affine, Perspective},
  Remap, Rotate, Crop, CopyMakeBorder.
- **Filters**: Gaussian, Median, Average, Laplacian, Bilateral,
  Motion, Conv2D.
- **Morphology** (erode, dilate, etc.).
- **Edges**: Canny.
- **Thresholds.**
- **Color**: ColorTwist, BrightnessContrast, GammaContrast.
- **Statistics.**
- **Composite operators.**
- Plus a handful of misc operators.

Each operator is a typed Rust handle (`Resize`, `Warp`, ...) with a
`run` method that takes a `Stream` for async dispatch.

## Platform support

CV-CUDA is **Linux-only**. The loader fails fast with a clear error on
Windows / macOS.

Pairs with [`baracuda-cvcuda-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cvcuda-sys`]: https://docs.rs/baracuda-cvcuda-sys
