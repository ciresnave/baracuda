//! GPU-gated integration tests for Wave-3 Driver-API additions:
//! cuLaunchKernelEx, and cuLibrary* (context-independent module loading).

use baracuda_driver::{library::Library, Context, Device, DeviceBuffer, Module, Stream};

const VECTOR_ADD_PTX: &str = include_str!("kernels/vector_add.ptx");

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA 12.0+"]
fn launch_ex_no_attributes_matches_classic_launch() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let module = Module::load_ptx(&ctx, VECTOR_ADD_PTX).unwrap();
    let kernel = module.get_function("vector_add").unwrap();

    let n: u32 = 1 << 12;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let d_a = DeviceBuffer::from_slice(&ctx, &a).unwrap();
    let d_b = DeviceBuffer::from_slice(&ctx, &b).unwrap();
    let d_c: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n as usize).unwrap();
    let a_ptr = d_a.as_raw();
    let b_ptr = d_b.as_raw();
    let c_ptr = d_c.as_raw();

    let block: u32 = 256;
    let grid: u32 = n.div_ceil(block);

    // SAFETY: args match the PTX signature.
    unsafe {
        kernel
            .launch()
            .grid((grid, 1, 1))
            .block((block, 1, 1))
            .stream(&stream)
            .arg(&a_ptr)
            .arg(&b_ptr)
            .arg(&c_ptr)
            .arg(&n)
            .launch_ex(&mut [])
            .expect("cuLaunchKernelEx");
    }
    stream.synchronize().unwrap();

    let mut got = vec![0.0f32; n as usize];
    d_c.copy_to_host(&mut got).unwrap();
    assert_eq!(got, expected);
}

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA 12.0+"]
fn library_load_and_launch_across_implicit_context() {
    // Load the PTX through the context-independent library API, then
    // materialize a Function in the current thread's context and launch.
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    ctx.set_current().unwrap();

    let lib = Library::load_ptx(VECTOR_ADD_PTX).expect("cuLibraryLoadData");
    let kernel = lib.get_kernel("vector_add").expect("cuLibraryGetKernel");
    let func = kernel
        .function_for_current_context()
        .expect("cuKernelGetFunction");

    let n: u32 = 1 << 10;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let d_a = DeviceBuffer::from_slice(&ctx, &a).unwrap();
    let d_b = DeviceBuffer::from_slice(&ctx, &b).unwrap();
    let d_c: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n as usize).unwrap();
    let a_ptr = d_a.as_raw();
    let b_ptr = d_b.as_raw();
    let c_ptr = d_c.as_raw();

    let block: u32 = 128;
    let grid: u32 = n.div_ceil(block);

    let stream = Stream::new(&ctx).unwrap();
    // SAFETY: args match the PTX signature.
    unsafe {
        func.launch()
            .grid((grid, 1, 1))
            .block((block, 1, 1))
            .stream(&stream)
            .arg(&a_ptr)
            .arg(&b_ptr)
            .arg(&c_ptr)
            .arg(&n)
            .launch()
            .unwrap();
    }
    stream.synchronize().unwrap();

    let mut got = vec![0.0f32; n as usize];
    d_c.copy_to_host(&mut got).unwrap();
    assert_eq!(got, expected);
}
