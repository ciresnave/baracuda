//! GPU-gated integration test for cuSPARSE SpMV.

use baracuda_cusparse::{spmv, spmv_buffer_size, DnVec, Handle, Op, SpMVAlg, SpMat};
use baracuda_driver::{Context, Device, DeviceBuffer};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn csr_spmv_matches_cpu_reference() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let handle = Handle::new().unwrap();

    // A simple 4×4 CSR matrix:
    //   A = [[1, 0, 0, 2],
    //        [0, 3, 0, 0],
    //        [0, 0, 0, 4],
    //        [5, 0, 6, 0]]
    // nnz = 6
    let row_offsets_h: [i32; 5] = [0, 2, 3, 4, 6];
    let col_indices_h: [i32; 6] = [0, 3, 1, 3, 0, 2];
    let values_h: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x_h: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    // y_expected = A * x = [1+8, 6, 16, 5+18] = [9, 6, 16, 23]
    let y_expected: [f32; 4] = [9.0, 6.0, 16.0, 23.0];

    let mut row_offsets = DeviceBuffer::from_slice(&ctx, &row_offsets_h).unwrap();
    let mut col_indices = DeviceBuffer::from_slice(&ctx, &col_indices_h).unwrap();
    let mut values = DeviceBuffer::from_slice(&ctx, &values_h).unwrap();
    let mut x = DeviceBuffer::from_slice(&ctx, &x_h).unwrap();
    let mut y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).unwrap();

    let a: SpMat<'_, f32> =
        SpMat::csr(4, 4, 6, &mut row_offsets, &mut col_indices, &mut values).unwrap();
    let x_d: DnVec<'_, f32> = DnVec::new(&mut x).unwrap();
    let mut y_d: DnVec<'_, f32> = DnVec::new(&mut y).unwrap();

    let buffer_bytes = spmv_buffer_size::<f32>(
        &handle,
        Op::N,
        &1.0,
        &a,
        &x_d,
        &0.0,
        &y_d,
        SpMVAlg::Default,
    )
    .unwrap();

    let mut workspace: DeviceBuffer<u8> = DeviceBuffer::new(&ctx, buffer_bytes.max(1)).unwrap();

    spmv::<f32>(
        &handle,
        Op::N,
        &1.0,
        &a,
        &x_d,
        &0.0,
        &mut y_d,
        SpMVAlg::Default,
        &mut workspace,
    )
    .unwrap();

    // Drop the descriptors before touching the buffer again.
    drop(y_d);
    drop(x_d);
    drop(a);

    let mut got = [0.0f32; 4];
    y.copy_to_host(&mut got).unwrap();
    for (g, e) in got.iter().zip(&y_expected) {
        assert!(
            (g - e).abs() < 1e-4,
            "SpMV mismatch: got {got:?}, expected {y_expected:?}"
        );
    }
}
