//! GPU-gated tests for cuBLAS L1/L2 additions.

use baracuda_cublas::{asum, copy, dot, gemv, iamax, iamin, nrm2, scal, Handle, Op};
use baracuda_driver::{Context, Device, DeviceBuffer};

fn setup() -> (Context, Handle) {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let handle = Handle::new().unwrap();
    (ctx, handle)
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn sdot_basic() {
    let (ctx, handle) = setup();
    let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let y: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
    // 1*5 + 2*6 + 3*7 + 4*8 = 70
    let d_x = DeviceBuffer::from_slice(&ctx, &x).unwrap();
    let d_y = DeviceBuffer::from_slice(&ctx, &y).unwrap();
    let got = dot(&handle, 4, &d_x.as_slice(), 1, &d_y.as_slice(), 1).unwrap();
    assert!((got - 70.0).abs() < 1e-5, "got {got}");
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn sscal_in_place() {
    let (ctx, handle) = setup();
    let mut d = DeviceBuffer::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    scal(&handle, 4, 0.5f32, &mut d, 1).unwrap();
    let mut back = vec![0.0f32; 4];
    d.copy_to_host(&mut back).unwrap();
    assert_eq!(back, [0.5, 1.0, 1.5, 2.0]);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn snrm2_and_sasum() {
    let (ctx, handle) = setup();
    let d = DeviceBuffer::from_slice(&ctx, &[3.0f32, 4.0]).unwrap();
    let n = nrm2(&handle, 2, &d.as_slice(), 1).unwrap();
    assert!((n - 5.0).abs() < 1e-5);

    let s = asum(&handle, 2, &d.as_slice(), 1).unwrap();
    assert!((s - 7.0).abs() < 1e-5);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn iamax_iamin_are_1_based() {
    let (ctx, handle) = setup();
    let d = DeviceBuffer::from_slice(&ctx, &[0.1f32, -0.5, 0.3, 0.2]).unwrap();
    // Largest abs: index 2 (value -0.5) -> 1-based index 2.
    assert_eq!(iamax(&handle, 4, &d.as_slice(), 1).unwrap(), 2);
    // Smallest abs: index 1 (value 0.1) -> 1-based index 1.
    assert_eq!(iamin(&handle, 4, &d.as_slice(), 1).unwrap(), 1);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn scopy_device_to_device() {
    let (ctx, handle) = setup();
    let src = DeviceBuffer::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let mut dst: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).unwrap();
    copy(&handle, 4, &src.as_slice(), 1, &mut dst, 1).unwrap();
    let mut back = vec![0.0f32; 4];
    dst.copy_to_host(&mut back).unwrap();
    assert_eq!(back, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn sgemv_2x3_times_3vec() {
    let (ctx, handle) = setup();
    // Column-major 2×3 matrix:
    //   [[1, 2, 3],
    //    [4, 5, 6]]  -> stored [1, 4, 2, 5, 3, 6]
    let a_host: [f32; 6] = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let x_host: [f32; 3] = [1.0, 2.0, 3.0];
    // A*x = [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
    let expected = [14.0f32, 32.0];

    let a = DeviceBuffer::from_slice(&ctx, &a_host).unwrap();
    let x = DeviceBuffer::from_slice(&ctx, &x_host).unwrap();
    let mut y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 2).unwrap();

    gemv(
        &handle,
        Op::N,
        2,
        3,
        1.0f32,
        &a,
        2,
        &x.as_slice(),
        1,
        0.0f32,
        &mut y,
        1,
    )
    .unwrap();
    let mut back = vec![0.0f32; 2];
    y.copy_to_host(&mut back).unwrap();
    for (g, e) in back.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-5, "got {back:?}, expected {expected:?}");
    }
}
