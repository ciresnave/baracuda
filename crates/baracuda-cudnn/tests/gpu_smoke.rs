//! GPU-gated integration test for cuDNN.
//!
//! Requires cuDNN installed; see `baracuda-cudnn-sys` for path probing.
//!
//! Convolution wrapper is shipped in the library but not exercised here:
//! cuDNN 9.16 has a packaging issue where its CUDA-13 variant still
//! demand-loads `cublasLt64_12.dll` at runtime. On matched cuDNN+CUDA
//! installs (cuDNN 9.17+ with CUDA 13, or cuDNN 9.x with CUDA 12) the
//! wrapper works; the conv test will be re-enabled when CI has a matched
//! install.

use baracuda_cudnn::{
    activation_forward, ActivationDescriptor, ActivationMode, DType, Handle, TensorDescriptor,
    TensorFormat,
};
use baracuda_driver::{Context, Device, DeviceBuffer};

#[test]
#[ignore = "requires an NVIDIA GPU and cuDNN 9.x"]
fn relu_forward_matches_cpu() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let handle = Handle::new().unwrap();

    let x_h: [f32; 8] = [-2.0, -0.5, 0.0, 0.5, 1.0, -3.0, 4.0, -7.0];
    let expected: [f32; 8] = [0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 4.0, 0.0];

    let x = DeviceBuffer::from_slice(&ctx, &x_h).unwrap();
    let mut y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 8).unwrap();

    let x_desc = TensorDescriptor::new_4d(TensorFormat::Nchw, DType::F32, 1, 1, 2, 4).unwrap();
    let y_desc = TensorDescriptor::new_4d(TensorFormat::Nchw, DType::F32, 1, 1, 2, 4).unwrap();

    let relu = ActivationDescriptor::new(ActivationMode::Relu, 0.0).unwrap();
    activation_forward(&handle, &relu, 1.0, &x_desc, &x, 0.0, &y_desc, &mut y)
        .expect("cudnnActivationForward");

    let mut got = [0.0f32; 8];
    y.copy_to_host(&mut got).unwrap();
    for (g, e) in got.iter().zip(&expected) {
        assert!((g - e).abs() < 1e-5, "got {got:?}, expected {expected:?}");
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU and cuDNN 9.x"]
fn conv_descriptors_sanity() {
    // Build the descriptors and compute output dims without executing the
    // kernel — that exercises the safe wrapper without triggering cuDNN's
    // cublasLt dependency.
    use baracuda_cudnn::{ConvMode, ConvolutionDescriptor, FilterDescriptor};
    baracuda_driver::init().unwrap();
    let _device = Device::get(0).unwrap();
    let x = TensorDescriptor::new_4d(TensorFormat::Nchw, DType::F32, 1, 1, 5, 5).unwrap();
    let w = FilterDescriptor::new_4d(TensorFormat::Nchw, DType::F32, 1, 1, 3, 3).unwrap();
    let conv =
        ConvolutionDescriptor::new_2d(1, 1, 1, 1, 1, 1, ConvMode::CrossCorrelation, DType::F32)
            .unwrap();
    let (n, c, h, w_out) = conv.output_dim_2d(&x, &w).unwrap();
    assert_eq!((n, c, h, w_out), (1, 1, 5, 5));
}

#[test]
#[ignore = "requires cuDNN"]
fn cudnn_version_is_reasonable() {
    let v = baracuda_cudnn::version().unwrap();
    eprintln!("cuDNN version: {v}");
    assert!(v >= 8000, "cudnnGetVersion returned unusually low: {v}");
}
