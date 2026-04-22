# CV-CUDA guide

CV-CUDA (NVCV) is NVIDIA's GPU computer-vision primitives library.
baracuda wraps ~30 operators covering geometric, filter, morphology,
color, stats, and composite ops.

## Installation

CV-CUDA ships as Linux binaries only (as of v0.x). On Linux:
`libcvcuda.so.0` + `libnvcv_types.so.0`. Windows users need to build
from source or wait for official Windows binaries.

## Concepts

- [`Tensor`](../../crates/baracuda-cvcuda/src/lib.rs) — an NVCV tensor
  (shape + dtype + layout). Reference-counted under the hood; Rust's
  `Drop` releases the ref.
- Per-operator RAII types (`Resize`, `CvtColor`, `Gaussian`, …). Each
  exposes `new()` and an `unsafe fn submit(&self, stream, ...)`.
- The `submit` calls are `unsafe` because NVCV tensor handles are
  opaque C pointers that the wrapper doesn't bounds-check.

## Flow: ML-preproc pipeline (resize → cvt → normalize → reformat)

```rust
use baracuda_cvcuda::*;

const LAYOUT_NHWC: i32 = 0;
const LAYOUT_NCHW: i32 = 1;
const DTYPE_U8: i32 = 2;
const DTYPE_F32: i32 = 8;

// Input: [1, 1080, 1920, 3] NHWC u8 (BGR frame from a decoder)
let input  = Tensor::new(&[1, 1080, 1920, 3], DTYPE_U8,  LAYOUT_NHWC)?;
let resized= Tensor::new(&[1,  224,  224, 3], DTYPE_U8,  LAYOUT_NHWC)?;
let rgb    = Tensor::new(&[1,  224,  224, 3], DTYPE_U8,  LAYOUT_NHWC)?;
let as_f32 = Tensor::new(&[1,  224,  224, 3], DTYPE_F32, LAYOUT_NHWC)?;
let nchw   = Tensor::new(&[1, 3, 224, 224],  DTYPE_F32, LAYOUT_NCHW)?;

unsafe {
    Resize::new()?.submit(stream, &input, &resized, Interpolation::Linear)?;
    CvtColor::new()?.submit(stream, &resized, &rgb,
        NVCVColorConversionCode::BGR2RGB)?;
    ConvertTo::new()?.submit(stream, &rgb, &as_f32,
        1.0 / 255.0,   // alpha
        0.0)?;         // beta
    Reformat::new()?.submit(stream, &as_f32, &nchw)?;
}
```

## Operator catalog

### Geometric

- `Resize`, `PillowResize` — bilinear / bicubic / area / Lanczos resampling.
- `WarpAffine`, `WarpPerspective` — 2×3 / 3×3 transforms with border modes.
- `Remap` — sample by pixel-offset map (optical-flow, undistort).
- `Rotate` — around image center, angle in degrees.
- `CenterCrop`, `CustomCrop` — rectangle crops.
- `CopyMakeBorder` — constant / replicate / reflect / wrap borders.
- `Reformat` — layout conversion (NHWC ↔ NCHW).
- `PadAndStack` — stack a VarShape image-batch into one fixed-shape tensor.

### Filters

- `Gaussian` — separable Gaussian blur.
- `MedianBlur` — median blur.
- `AverageBlur`, `BoxFilter` — averaging filters (normalized / unnormalized).
- `Laplacian` — 2nd-derivative edge.
- `BilateralFilter` — edge-preserving smooth.
- `MotionBlur` — directional motion blur.
- `Conv2D` — arbitrary-kernel convolution.

### Morphology

- `Morphology` — erode / dilate / open / close with a user mask.

### Edges + stats

- `Canny` — Canny edge detector.
- `Histogram`, `HistogramEq` — histogram compute + equalization.
- `MinMaxLoc` — min/max values + locations.

### Thresholds

- `Threshold` — binary / trunc / tozero / Otsu / triangle.
- `AdaptiveThreshold` — mean-C / gaussian-C variants.

### Color

- `CvtColor` — OpenCV-style colorspace conversion (BGR ↔ RGB ↔ YUV ↔ HSV ↔ grayscale).
- `ColorTwist` — 3×4 / 4×4 color-transform matrix.
- `BrightnessContrast`, `GammaContrast` — per-channel adjustments.

### Composite + channel

- `Composite` — alpha-mask composite.
- `Stack` — stack a VarShape image batch into one tensor.
- `ChannelReorder` — arbitrary channel permutation (BGR→RGB, etc).

### Misc

- `Erase` — random / fixed-content rectangle erasure.
- `Inpaint` — mask-based inpainting.
- `AddWeighted` — `out = α·in1 + β·in2 + γ` with saturation.
- `NonMaxSuppression` — over detection boxes + scores.
- `Normalize` — ML preprocessing `(x - base) * scale + shift`.

## Submit-is-unsafe rationale

NVCV tensors and operator handles are opaque `*mut c_void`. CV-CUDA's C
ABI doesn't encode shape / dtype invariants in the tensor type, so the
wrapper can't statically verify that a tensor matches an operator's
expected layout. We mark `submit(...)` `unsafe` so callers acknowledge
they've set up shapes / dtypes / layouts correctly.
