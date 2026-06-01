# baracuda-nvjpeg

Safe Rust wrappers for **NVIDIA nvJPEG** — GPU-accelerated JPEG decode
and encode. Comprehensive coverage of single + batched paths and the
modern three-phase hybrid pipeline.

> **Preferred new path: [`baracuda-nvimagecodec`].** NVIDIA's nvImageCodec
> supersedes the standalone nvJPEG and adds PNG / TIFF / JPEG2000 / WebP /
> BMP behind one batched pipeline (it uses nvJPEG internally for JPEG). This
> crate remains fully supported for back-compat and JPEG-only callers, and
> the two coexist (different shared objects). New code that needs more than
> JPEG should reach for `baracuda-nvimagecodec`.

## Coverage

- **Single-image decode**: `nvjpegDecode` with planar / interleaved
  output, JpegStream parser, DecodeParams (output format, ROI, CMYK
  handling).
- **Batched decode**: simple batched + the **three-phase hybrid**
  pipeline:
  1. Host parse (CPU-side bitstream parsing).
  2. Host transfer (D2H of phase-1 outputs + H2D of decode tables).
  3. Device decode (the actual GPU kernels).

  Decoupling the phases lets the caller overlap CPU parsing of batch
  N+1 with GPU decode of batch N.
- **Buffer pools**: `BufferPinned` for host pinned memory,
  `BufferDevice` for GPU memory; both poolable.
- **DecodeParams**: pick output format (RGB, BGR, RGBI, BGRI, YUV420,
  YUV422, YUV444), ROI cropping, CMYK input handling.
- **Encoder**: quality, chroma subsampling (`Css420` / `Css422` /
  `Css444`), optimized Huffman tables, encode from packed image or
  planar YUV, retrieve bitstream.

```rust,no_run
use baracuda_nvjpeg::{Handle, JpegState};
use baracuda_driver::{Context, Device};

# fn demo() -> Result<(), Box<dyn std::error::Error>> {
let ctx = Context::new(&Device::get(0)?)?;
let handle = Handle::new(&ctx)?;
let mut state = JpegState::new(&handle)?;

let bitstream: Vec<u8> = std::fs::read("photo.jpg")?;
let info = handle.get_image_info(&bitstream)?;
// ... decode + retrieve into device buffers ...
# Ok(()) }
```

Pairs with [`baracuda-nvjpeg-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-nvjpeg-sys`]: https://docs.rs/baracuda-nvjpeg-sys
[`baracuda-nvimagecodec`]: https://docs.rs/baracuda-nvimagecodec
