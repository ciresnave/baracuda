//! CV-CUDA integration test — build tensors and submit a Resize.
//!
//! CV-CUDA ships as Linux binaries only (as of 2026). On Windows this
//! test skips gracefully.

use baracuda_cvcuda::{Interpolation, Resize, Tensor};
use baracuda_runtime::{Device, Stream};

const LAYOUT_NHWC: i32 = 0; // NVCVTensorLayout::NHWC
const DTYPE_U8: i32 = 2; // NVCVDataType::U8 (first unsigned 8-bit slot)

#[test]
#[ignore = "requires CV-CUDA installed + NVIDIA GPU (Linux-only)"]
fn resize_tensor_smoke() {
    if baracuda_cvcuda::probe().is_err() {
        eprintln!("CV-CUDA not installed on this host — skipping");
        return;
    }
    Device::from_ordinal(0).set_current().unwrap();
    let stream = Stream::new().unwrap();

    // NHWC tensor: batch=1, h=32, w=32, c=3, u8
    let input_shape = [1i64, 32, 32, 3];
    let output_shape = [1i64, 16, 16, 3];

    let input = Tensor::new(&input_shape, DTYPE_U8, LAYOUT_NHWC).unwrap();
    let output = Tensor::new(&output_shape, DTYPE_U8, LAYOUT_NHWC).unwrap();

    let op = Resize::new().unwrap();
    unsafe {
        op.submit(stream.as_raw(), &input, &output, Interpolation::Linear)
            .unwrap();
    }
    stream.synchronize().unwrap();
    eprintln!("CV-CUDA Resize 32×32 → 16×16 completed");
}
