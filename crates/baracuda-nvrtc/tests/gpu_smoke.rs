//! GPU-gated integration test: compile a kernel via NVRTC and execute it
//! through the Driver API.

use baracuda_driver::{Context, Device, DeviceBuffer, Module, Stream};
use baracuda_nvrtc::Program;

const CUDA_SRC: &str = r#"
extern "C" __global__ void saxpy(
    float a,
    const float* x,
    float* y,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
"#;

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn compile_and_launch_saxpy() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let (cc_major, cc_minor) = device.compute_capability().unwrap();

    // Tell NVRTC the compute capability of the GPU we're targeting.
    let arch = format!("--gpu-architecture=compute_{cc_major}{cc_minor}");

    let ptx = Program::compile(CUDA_SRC, "saxpy.cu", &[&arch]).expect("NVRTC compile");
    assert!(!ptx.is_empty(), "PTX output is empty");

    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();
    let module = Module::load_ptx(&ctx, &ptx).expect("Module::load_ptx");
    let kernel = module.get_function("saxpy").expect("get_function");

    let n: u32 = 1 << 14;
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let y0: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let a = 2.5f32;
    let expected: Vec<f32> = x.iter().zip(&y0).map(|(xi, yi)| a * xi + yi).collect();

    let d_x = DeviceBuffer::from_slice(&ctx, &x).unwrap();
    let d_y = DeviceBuffer::from_slice(&ctx, &y0).unwrap();
    let x_ptr = d_x.as_raw();
    let y_ptr = d_y.as_raw();

    let block: u32 = 256;
    let grid: u32 = n.div_ceil(block);
    unsafe {
        kernel
            .launch()
            .grid((grid, 1, 1))
            .block((block, 1, 1))
            .stream(&stream)
            .arg(&a)
            .arg(&x_ptr)
            .arg(&y_ptr)
            .arg(&n)
            .launch()
            .expect("launch");
    }
    stream.synchronize().unwrap();

    let mut got = vec![0.0f32; n as usize];
    d_y.copy_to_host(&mut got).unwrap();
    for (g, e) in got.iter().zip(&expected) {
        assert!((g - e).abs() < 1e-3, "saxpy mismatch");
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn nvrtc_version_is_reasonable() {
    let (major, minor) = baracuda_nvrtc::version().expect("nvrtcVersion");
    eprintln!("NVRTC {major}.{minor}");
    assert!(major >= 11);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn compile_error_surfaces_log() {
    let prog =
        Program::new("this is definitely not valid CUDA C++", "broken.cu").expect("create program");
    match prog.compile_raw(&["--gpu-architecture=compute_50"]) {
        Err(_) => {
            let log = prog.log().expect("program log");
            assert!(!log.is_empty(), "compile failure should leave a log");
            eprintln!("expected failure log:\n{log}");
        }
        Ok(()) => panic!("expected compile error"),
    }
}
