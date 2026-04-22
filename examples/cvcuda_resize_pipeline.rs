//! CV-CUDA image preprocessing pipeline (Linux).
//!
//! Takes an input NHWC u8 BGR frame, resizes to 224×224, converts to
//! RGB, scales to f32 in [0, 1], and reformats to NCHW f32 — the
//! classic vision-model input transform.
//!
//! This example is Linux-only since CV-CUDA ships as Linux binaries.

use baracuda_cvcuda::*;
use baracuda_runtime::{Device, Stream};

const LAYOUT_NHWC: i32 = 0;
const LAYOUT_NCHW: i32 = 1;
const DTYPE_U8: i32 = 2;
const DTYPE_F32: i32 = 8;

fn main() {
    if let Err(e) = baracuda_cvcuda::probe() {
        eprintln!("CV-CUDA not available: {e:?}");
        std::process::exit(1);
    }

    Device::from_ordinal(0).set_current().unwrap();
    let stream = Stream::new().unwrap();

    println!("CV-CUDA preprocessing: 1920×1080 BGR u8 → 224×224 RGB f32 NCHW");

    // Input: 1080p BGR frame (NVCV u8 NHWC).
    let input = Tensor::new(&[1, 1080, 1920, 3], DTYPE_U8, LAYOUT_NHWC).unwrap();

    // Stage tensors.
    let resized = Tensor::new(&[1, 224, 224, 3], DTYPE_U8, LAYOUT_NHWC).unwrap();
    let rgb = Tensor::new(&[1, 224, 224, 3], DTYPE_U8, LAYOUT_NHWC).unwrap();
    let as_f32 = Tensor::new(&[1, 224, 224, 3], DTYPE_F32, LAYOUT_NHWC).unwrap();
    let nchw = Tensor::new(&[1, 3, 224, 224], DTYPE_F32, LAYOUT_NCHW).unwrap();

    // Build operators once, reuse across frames.
    let resize = Resize::new().unwrap();
    let cvt = CvtColor::new().unwrap();
    let conv = ConvertTo::new().unwrap();
    let reformat = Reformat::new().unwrap();

    let t0 = std::time::Instant::now();
    unsafe {
        resize
            .submit(stream.as_raw(), &input, &resized, Interpolation::Linear)
            .unwrap();
        cvt.submit(
            stream.as_raw(),
            &resized,
            &rgb,
            NVCVColorConversionCode::BGR2RGB,
        )
        .unwrap();
        conv.submit(stream.as_raw(), &rgb, &as_f32, 1.0 / 255.0, 0.0)
            .unwrap();
        reformat.submit(stream.as_raw(), &as_f32, &nchw).unwrap();
    }
    stream.synchronize().unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("  pipeline: {:.3} ms", ms);
    println!("OK — preprocessed 1 frame");
}
