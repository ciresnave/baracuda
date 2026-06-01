# baracuda-nvimagecodec

Safe Rust wrappers for **NVIDIA nvImageCodec** — the unified GPU image codec
library. nvImageCodec supersedes the standalone nvJPEG and decodes many
container formats (JPEG, JPEG2000, TIFF, PNG, BMP, WebP, ...) behind one
batched pipeline. It's the modern path for GPU image decode in ML data
loading.

> **Coexists with [`baracuda-nvjpeg`].** That crate stays for back-compat
> callers; new code should prefer this one. nvImageCodec uses nvJPEG
> internally for the JPEG path, so linking both is fine (different `.so`).

## Coverage (v0.1)

- **Single-image decode** of JPEG / PNG / TIFF (and any other codec the
  installed library supports) into a caller-provided `DeviceBuffer<u8>`.
- **Async**: decode runs on a CUDA stream and returns a [`Future`]; the host
  blocks via `Future::wait()` (or by synchronizing the stream).
- **Metadata probe**: `CodeStream::image_info()` reports dimensions, codec
  name, sample format, and color spec without decoding.
- Interleaved 8-bit RGB output via `Image::new_interleaved_rgb8`.

### Deferred (Tier-2)

Encoder support, batch / GPU-resident batch decode, JPEG2000 / WebP / BMP /
DICOM specifics, and augmentation pipelines are follow-ups. Augmentation
belongs in a separate crate (`baracuda-cvcuda` / DALI), not here.

```rust,no_run
use baracuda_nvimagecodec::{CodeStream, Decoder, DecodeParams, Image, Instance};
use baracuda_driver::{Context, Device, DeviceBuffer};

# fn demo(jpeg: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
let ctx = Context::new(&Device::get(0)?)?;

let instance = Instance::new()?;
let decoder = Decoder::new(&instance)?;

let stream = CodeStream::from_host_mem(&instance, jpeg)?;
let info = stream.image_info()?;

let mut buf: DeviceBuffer<u8> =
    DeviceBuffer::zeros(&ctx, (info.width * info.height * 3) as usize)?;
let image = Image::new_interleaved_rgb8(&instance, &mut buf, info.width, info.height, None)?;

let future = decoder.decode(&stream, &image, &DecodeParams::default())?;
future.wait()?;
assert!(future.all_succeeded()?);
// `buf` now holds interleaved RGB8 pixels.
# Ok(()) }
```

## Lifetimes

`CodeStream::from_host_mem` **borrows** the input bytes; the returned
`CodeStream<'data>` cannot outlive the slice. An `Image<'buf>` borrows its
output buffer mutably. These borrows are enforced at compile time, so the
classic use-after-free footguns of the C API are unrepresentable here.

## Installing nvImageCodec

`libnvimgcodec` ships separately from the CUDA toolkit (part of NVIDIA's
nvImageCodec / cuCIM family). Install via pip
(`pip install nvidia-nvimgcodec-cu12`) or the standalone tarball, then make
sure `libnvimgcodec.so.0` (Linux) / `nvimgcodec_0.dll` (Windows) is on the
loader search path. The crate resolves it lazily — there is no link-time
dependency, so it builds without the library present (the GPU smoke tests
are `#[ignore]`-gated for the same reason).

Pairs with [`baracuda-nvimagecodec-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-nvjpeg`]: https://docs.rs/baracuda-nvjpeg
[`baracuda-nvimagecodec-sys`]: https://docs.rs/baracuda-nvimagecodec-sys
