# baracuda-nvimagecodec-sys

Raw FFI bindings + dynamic loader for **NVIDIA nvImageCodec** — the unified
GPU image codec library (JPEG, JPEG2000, TIFF, PNG, BMP, WebP, ...).

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading); no
link-time dependency on `libnvimgcodec.so` / `nvimgcodec_0.dll`. The library
ships separately from the CUDA toolkit (part of NVIDIA's nvImageCodec /
cuCIM / RAPIDS family) — see the safe wrapper's README for install notes.

**Most users want [`baracuda-nvimagecodec`]** — that crate exposes typed
`Instance` / `Decoder` / `CodeStream` / `Image` / `Future` handles.

## What's exposed

- Handles: `nvimgcodecInstance_t`, `nvimgcodecCodeStream_t`,
  `nvimgcodecImage_t`, `nvimgcodecDecoder_t`, `nvimgcodecFuture_t`.
- Structs (versioned `struct_type` / `struct_size` preamble, populated by
  `new()` constructors): `nvimgcodecImageInfo_t`,
  `nvimgcodecImagePlaneInfo_t`, `nvimgcodecInstanceCreateInfo_t`,
  `nvimgcodecExecutionParams_t`, `nvimgcodecDecodeParams_t`,
  `nvimgcodecProperties_t`, `nvimgcodecOrientation_t`.
- Enums: sample format / data type, chroma subsampling, color spec, image
  buffer kind, structure type, processing-status bitfield.
- High-level batched decode pipeline: instance → code stream → image →
  decoder → `nvimgcodecDecoderDecode` → future.

## ABI target

Layouts and enum discriminants target the **nvImageCodec 0.x** C ABI
(`nvimgcodec.h`). The versioned struct preamble is the library's
forward-compatibility mechanism: each struct carries its own
`struct_size`, so a newer library tolerates these definitions.

## Coexistence with `baracuda-nvjpeg-sys`

nvImageCodec supersedes the standalone nvJPEG. Both `-sys` crates can be
linked simultaneously (different shared objects); nvImageCodec uses nvJPEG
internally for the JPEG path.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-nvimagecodec`]: https://docs.rs/baracuda-nvimagecodec
