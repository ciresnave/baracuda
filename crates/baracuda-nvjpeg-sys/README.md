# baracuda-nvjpeg-sys

Raw FFI bindings + dynamic loader for **NVIDIA nvJPEG** — GPU-accelerated
JPEG decode and encode.

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libnvjpeg.so` / `nvjpeg64_*.dll`.

**Most users want [`baracuda-nvjpeg`]** — that crate exposes typed
handles, single + batched decode (simple and the three-phase hybrid:
host parse → host transfer → device decode), DecodeParams (output
format, ROI, CMYK), JpegStream parser, and a full encoder API.

## What's exposed

- `nvjpegHandle_t`, `nvjpegJpegState_t`, `nvjpegJpegStream_t`,
  `nvjpegBufferPinned_t`, `nvjpegBufferDevice_t`, `nvjpegDecodeParams_t`,
  `nvjpegEncoderState_t`, `nvjpegEncoderParams_t`.
- Single-image decode: `nvjpegDecode`, planar + interleaved output.
- Batched decode: simple + three-phase hybrid pipeline.
- Encoder: quality, chroma subsampling, optimized Huffman, encode
  image / encode YUV, retrieve bitstream.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-nvjpeg`]: https://docs.rs/baracuda-nvjpeg
