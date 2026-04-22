//! GPU-gated integration test for cuSOLVER dense LU + solve.

use baracuda_cusolver::{sgetrf, sgetrs, DnHandle, Op};
use baracuda_driver::{Context, Device, DeviceBuffer};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn dense_lu_solve_3x3() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let handle = DnHandle::new().unwrap();

    // A 3×3 column-major matrix:
    //   A = [[2, 1, 1],
    //        [4, 3, 3],
    //        [8, 7, 9]]
    // Stored as [2, 4, 8, 1, 3, 7, 1, 3, 9].
    let a_h: [f32; 9] = [2.0, 4.0, 8.0, 1.0, 3.0, 7.0, 1.0, 3.0, 9.0];

    // RHS: b = A * [1, 1, 1] = [4, 10, 24] → x should be [1, 1, 1].
    let b_h: [f32; 3] = [4.0, 10.0, 24.0];

    let mut a = DeviceBuffer::from_slice(&ctx, &a_h).unwrap();
    let mut b = DeviceBuffer::from_slice(&ctx, &b_h).unwrap();
    let mut ipiv: DeviceBuffer<i32> = DeviceBuffer::new(&ctx, 3).unwrap();
    let mut info: DeviceBuffer<i32> = DeviceBuffer::new(&ctx, 1).unwrap();

    sgetrf(&handle, 3, 3, &mut a, 3, &mut ipiv, &mut info).expect("cusolverDnSgetrf");
    let mut info_host = [42i32];
    info.copy_to_host(&mut info_host).unwrap();
    assert_eq!(
        info_host[0], 0,
        "non-zero info from getrf: {}",
        info_host[0]
    );

    sgetrs(&handle, Op::N, 3, 1, &a, 3, &ipiv, &mut b, 3, &mut info).expect("cusolverDnSgetrs");
    let mut info_host = [42i32];
    info.copy_to_host(&mut info_host).unwrap();
    assert_eq!(
        info_host[0], 0,
        "non-zero info from getrs: {}",
        info_host[0]
    );

    let mut x = [0.0f32; 3];
    b.copy_to_host(&mut x).unwrap();
    for &v in &x {
        assert!(
            (v - 1.0).abs() < 1e-3,
            "LU solve gave {x:?}, expected [1, 1, 1]"
        );
    }
}
