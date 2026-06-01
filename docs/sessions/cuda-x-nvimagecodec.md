# Session prompt — Add `baracuda-nvimagecodec` wrapper

Working on baracuda at `c:\Users\cires\OneDrive\Documents\projects\baracuda`.
Per the Phase 65 CUDA-X audit, nvImageCodec supersedes the standalone
nvJPEG (which baracuda already wraps). This session adds the new,
broader codec library.

## Context

[nvImageCodec](https://developer.nvidia.com/nvimgcodec) is NVIDIA's
unified GPU image codec library. It replaces / supersedes standalone
nvJPEG and adds support for:

- JPEG, JPEG2000, JPEG-XR
- TIFF
- PNG
- BMP
- WebP (decode only)
- DICOM (medical imaging, partial)

It's the modern way to do GPU image decode for ML training pipelines
(image augmentation, distributed dataloading). Replaces the
piecemeal nvJPEG + 3rd-party-codec approach.

baracuda has `baracuda-nvjpeg{,-sys}` from a prior phase. This new
crate would coexist; `baracuda-nvjpeg` stays for back-compat callers,
new callers use `baracuda-nvimagecodec`.

## Scope

**Crates to create:**

1. `crates/baracuda-nvimagecodec-sys/` — `extern "C"` FFI declarations
   over `libnvimgcodec.so`. C-ABI is reasonably clean; bindgen
   probably overkill, manual declarations work.
2. `crates/baracuda-nvimagecodec/` — safe wrapper.

## Reference patterns

`crates/baracuda-nvjpeg/` — directly analogous. Mirror the design:
typed `Decoder` / `Encoder` handles, `Image` descriptor for source
data, async streams support.

## Linking

`libnvimgcodec.so` is part of NVIDIA's RAPIDS / cuCIM family. May
need to be installed separately; use `libloading` for lazy lib
resolution (same pattern as baracuda-nccl / baracuda-nvjpeg). No
build.rs link directives.

## Tier 1 deliverables

1. Decoder for JPEG + PNG + TIFF (the three most common in ML
   pipelines).
2. Async decode interface accepting a CUDA stream.
3. Output to caller-provided `DeviceBuffer<u8>` (or RGB f32 if
   the decoder supports direct format conversion).
4. Cargo feature `nvimagecodec` on `-sys` and the safe wrapper.
5. Smoke tests using small embedded test images (1x1 PNG, 4x4 JPEG,
   etc.).

## Tier 2 deferrable

- Encoder support (JPEG, PNG)
- JPEG2000, WebP, BMP decode
- DICOM (medical imaging)
- Batch decode / GPU-resident batch operations
- Augmentation pipeline integration (zoom, crop, normalize) — that's a
  separate concern, possibly under `baracuda-cvcuda` or DALI.

## Coexistence with `baracuda-nvjpeg`

Both crates can coexist. nvImageCodec internally uses nvJPEG for the
JPEG path; if a caller links both, no conflict (different .so).

Document in the README of `baracuda-nvjpeg` that nvImageCodec is the
preferred new path.

## Out of scope

- Don't deprecate or remove `baracuda-nvjpeg`. Coexist.
- Don't add training-pipeline integration (DALI-style). Pure decode.

## Coordination

- Working directory: `c:\Users\cires\OneDrive\Documents\projects\baracuda`
- Branch: `phase70-nvimagecodec`
- No version bump, no publish.
- Commit on branch + push + stop.

## Stop conditions

- If `libnvimgcodec.so` is not available on the dev machine + can't be
  installed easily: ship the crates with `#[ignore]`-gated tests, no
  actual hardware exercise. Document install path requirement.
- If the API has unusual lifetime requirements (e.g. decoder must
  outlive the input stream): document them in the safe-wrapper
  docstring + use `PhantomData` lifetime tracking.
- If you find the crate pair already shipped: stop, report.

## Memory file

Write `project_phase70_complete.md`.
