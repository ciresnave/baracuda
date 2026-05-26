//! Phase 31 — direct-FFI smoke for `reduce_max_to_<dtype>_run`.

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn contig_strides(shape: &[i32]) -> Vec<i64> {
    let mut s = vec![0i64; shape.len()];
    let mut acc: i64 = 1;
    for i in (0..shape.len()).rev() {
        s[i] = acc;
        acc *= shape[i] as i64;
    }
    s
}

/// CPU reference: max over the broadcast-reverse set.
fn cpu_max_to_f32(
    src: &[f32],
    in_shape: &[i32],
    out_shape: &[i32],
) -> Vec<f32> {
    let rank = in_shape.len();
    let out_numel: usize = out_shape.iter().map(|&d| d as usize).product();
    let in_numel: usize = in_shape.iter().map(|&d| d as usize).product();
    let mut dst = vec![f32::NEG_INFINITY; out_numel];
    let out_contig = contig_strides(out_shape);

    for in_lin in 0..in_numel {
        let mut lin = in_lin;
        let mut in_coord = vec![0i32; rank];
        for d in (0..rank).rev() {
            let s = in_shape[d] as usize;
            in_coord[d] = (lin % s) as i32;
            lin /= s;
        }

        let mut out_lin: usize = 0;
        for d in 0..rank {
            let c = if out_shape[d] == 1 { 0 } else { in_coord[d] };
            out_lin += (c as usize) * (out_contig[d] as usize);
        }
        if src[in_lin] > dst[out_lin] {
            dst[out_lin] = src[in_lin];
        }
    }
    dst
}

#[test]
#[ignore]
fn ffi_reduce_max_to_f32_2d_reduce_dim0() {
    let (ctx, stream) = setup();
    let in_shape: Vec<i32> = vec![4, 5];
    let out_shape: Vec<i32> = vec![1, 5];
    let in_stride = contig_strides(&in_shape);

    // Stagger the input so the max is non-trivial.
    let host_src: Vec<f32> = (0..20).map(|i| {
        let r = (i / 5) as f32;
        let c = (i % 5) as f32;
        c - r + 0.1 * (i as f32)
    }).collect();
    let expected = cpu_max_to_f32(&host_src, &in_shape, &out_shape);

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload src");
    let mut dev_dst: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, expected.len()).expect("alloc dst");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_reduce_max_to_f32_run(
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
            in_shape.as_ptr(), in_stride.as_ptr(),
            in_shape.len() as i32,
            out_shape.as_ptr(),
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "reduce_max_to_f32 returned {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; expected.len()];
    dev_dst.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(),
            "max_to f32 @ {i}: got {g} expected {e}");
    }
}

#[test]
#[ignore]
fn ffi_reduce_max_to_f32_full_reduce_with_negatives() {
    let (ctx, stream) = setup();
    let in_shape: Vec<i32> = vec![6];
    let out_shape: Vec<i32> = vec![1];
    let in_stride = contig_strides(&in_shape);
    let host_src: Vec<f32> = vec![-7.0, -3.0, -10.0, -1.5, -8.0, -2.0];
    let expected = vec![-1.5_f32];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload src");
    let mut dev_dst: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, 1).expect("alloc dst");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_reduce_max_to_f32_run(
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
            in_shape.as_ptr(), in_stride.as_ptr(),
            in_shape.len() as i32,
            out_shape.as_ptr(),
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 1];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got[0].to_bits(), expected[0].to_bits(),
        "max_to full reduce with negatives: got {} expected {}", got[0], expected[0]);
}

#[test]
#[ignore]
fn ffi_reduce_max_to_f64_3d() {
    let (ctx, stream) = setup();
    let in_shape: Vec<i32> = vec![2, 3, 4];
    let out_shape: Vec<i32> = vec![2, 1, 4];
    let in_stride = contig_strides(&in_shape);
    let host_src: Vec<f64> = (0..24).map(|i| (i as f64) * 0.5 - 5.0).collect();

    // Compute the CPU reference using the same path as the kernel.
    let rank = in_shape.len();
    let out_numel: usize = out_shape.iter().map(|&d| d as usize).product();
    let in_numel: usize = in_shape.iter().map(|&d| d as usize).product();
    let mut expected = vec![f64::NEG_INFINITY; out_numel];
    let out_contig = contig_strides(&out_shape);
    for in_lin in 0..in_numel {
        let mut lin = in_lin;
        let mut in_coord = vec![0i32; rank];
        for d in (0..rank).rev() {
            let s = in_shape[d] as usize;
            in_coord[d] = (lin % s) as i32;
            lin /= s;
        }
        let mut out_lin: usize = 0;
        for d in 0..rank {
            let c = if out_shape[d] == 1 { 0 } else { in_coord[d] };
            out_lin += (c as usize) * (out_contig[d] as usize);
        }
        if host_src[in_lin] > expected[out_lin] {
            expected[out_lin] = host_src[in_lin];
        }
    }

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload src");
    let mut dev_dst: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, expected.len()).expect("alloc dst");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_reduce_max_to_f64_run(
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
            in_shape.as_ptr(), in_stride.as_ptr(),
            in_shape.len() as i32,
            out_shape.as_ptr(),
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; expected.len()];
    dev_dst.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(),
            "max_to f64 @ {i}: got {g} expected {e}");
    }
}
