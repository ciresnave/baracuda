//! Minimal cuDNN 2-D forward convolution demo.
//!
//! Builds NCHW tensor + filter + convolution descriptors, queries the
//! required workspace, runs `cudnnConvolutionForward`, then verifies the
//! output sum against a known fixture (all-ones input, all-ones filter
//! → output cell value equals the receptive-field overlap).
//!
//! Run with:
//!
//! ```text
//! cargo run --example conv2d -p baracuda-cudnn
//! ```

use baracuda_cudnn::{
    convolution_forward, convolution_forward_workspace_size, ConvMode,
    ConvolutionDescriptor, DType, FilterDescriptor, FwdAlgo, Handle, TensorDescriptor,
    TensorFormat,
};
use baracuda_driver::{Context, Device, DeviceBuffer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    baracuda_driver::init()?;
    let device = Device::get(0)?;
    println!("device: {}", device.name()?);
    let ctx = Context::new(&device)?;
    let cudnn = Handle::new()?;
    println!("cuDNN version (packed): {}", baracuda_cudnn::version()?);

    // Shapes: NCHW 1×1×8×8 input, 1 output channel, 3×3 kernel,
    // padding 1, stride 1, dilation 1 → output is the same 8×8.
    let (n, c, h, w) = (1, 1, 8, 8);
    let (k, kh, kw) = (1, 3, 3);
    let (pad_h, pad_w) = (1, 1);
    let (str_h, str_w) = (1, 1);
    let (dil_h, dil_w) = (1, 1);

    let x_desc = TensorDescriptor::new_4d(TensorFormat::Nchw, DType::F32, n, c, h, w)?;
    let w_desc = FilterDescriptor::new_4d(TensorFormat::Nchw, DType::F32, k, c, kh, kw)?;
    let conv = ConvolutionDescriptor::new_2d(
        pad_h,
        pad_w,
        str_h,
        str_w,
        dil_h,
        dil_w,
        ConvMode::CrossCorrelation,
        DType::F32,
    )?;

    // Compute output dims (should equal `h`, `w` for pad=1, k=3, stride=1).
    let (on, oc, oh, ow) = conv.output_dim_2d(&x_desc, &w_desc)?;
    println!("output dim: {on}×{oc}×{oh}×{ow}");
    let y_desc = TensorDescriptor::new_4d(TensorFormat::Nchw, DType::F32, on, oc, oh, ow)?;

    let algo = FwdAlgo::ImplicitGemm;
    let ws_bytes = convolution_forward_workspace_size(&cudnn, &x_desc, &w_desc, &conv, &y_desc, algo)?;
    println!("workspace bytes: {ws_bytes}");

    // Inputs: all ones in X and W → each output cell = number of valid
    // (in-bounds, non-padded) (kh, kw, c) taps contributing to it. For a
    // padded interior cell that's 3×3 = 9; corners get 4; edges get 6.
    let x_host = vec![1.0f32; (n * c * h * w) as usize];
    let w_host = vec![1.0f32; (k * c * kh * kw) as usize];
    let x_buf = DeviceBuffer::from_slice(&ctx, &x_host)?;
    let w_buf = DeviceBuffer::from_slice(&ctx, &w_host)?;
    let mut y_buf: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (on * oc * oh * ow) as usize)?;
    let mut ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes.max(1))?;

    convolution_forward(
        &cudnn,
        1.0,
        &x_desc,
        &x_buf,
        &w_desc,
        &w_buf,
        &conv,
        algo,
        &mut ws,
        0.0,
        &y_desc,
        &mut y_buf,
    )?;

    let mut y_host = vec![0.0f32; (on * oc * oh * ow) as usize];
    y_buf.copy_to_host(&mut y_host)?;

    // Corners contribute 4, edges 6, interior 9; sum over an 8×8 image:
    //   4 corners × 4 + (4 edges × 6 cells) × 6 + (6×6 interior) × 9
    // = 16 + 144 + 324 = 484.
    let sum: f32 = y_host.iter().sum();
    println!("output sum = {sum}  (expected 484 for all-ones 8×8 with 3×3 pad-1)");
    assert!((sum - 484.0).abs() < 1e-2, "unexpected conv2d sum {sum}");
    println!("OK");
    Ok(())
}
