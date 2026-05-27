//! Phase 36 (Fuel ask Gap 6a) — direct-FFI smoke for the new argsort
//! dtype fanout (`u8`, `i8`, `u32`, `i16`, `bf16`, `f16`, FP8 E4M3).
//!
//! `row_len ≤ 1024` block-bitonic, same kernel structure as the
//! existing F32/F64/I32/I64 family.

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn check_ascending<T: PartialOrd + Copy + core::fmt::Debug>(input: &[T], indices: &[i32]) {
    assert_eq!(input.len(), indices.len());
    for i in 1..indices.len() {
        let a = input[indices[i - 1] as usize];
        let b = input[indices[i] as usize];
        assert!(
            a <= b,
            "argsort ascending broken at i={i}: input[{}]={a:?} > input[{}]={b:?}",
            indices[i - 1],
            indices[i]
        );
    }
}

#[test]
#[ignore]
fn argsort_u8_basic() {
    let (ctx, stream) = setup();
    let row_len: i32 = 16;
    let input: Vec<u8> = vec![5, 3, 8, 1, 9, 2, 7, 4, 6, 0, 15, 11, 13, 10, 14, 12];
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_idx: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, input.len()).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_u8_run(
            1,
            row_len,
            0,
            dev_in.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut got = vec![0i32; input.len()];
    dev_idx.copy_to_host(&mut got).expect("dl");
    check_ascending(&input, &got);
}

#[test]
#[ignore]
fn argsort_i8_basic() {
    let (ctx, stream) = setup();
    let row_len: i32 = 8;
    let input: Vec<i8> = vec![-5, 3, -8, 1, 9, -2, 7, 0];
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_idx: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, input.len()).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_i8_run(
            1,
            row_len,
            0,
            dev_in.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut got = vec![0i32; input.len()];
    dev_idx.copy_to_host(&mut got).expect("dl");
    check_ascending(&input, &got);
}

#[test]
#[ignore]
fn argsort_u32_basic() {
    let (ctx, stream) = setup();
    let row_len: i32 = 8;
    let input: Vec<u32> = vec![5_000_000, 3, 800, 1, 0xDEAD_BEEF, 2, 7, 4];
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_idx: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, input.len()).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_u32_run(
            1,
            row_len,
            0,
            dev_in.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut got = vec![0i32; input.len()];
    dev_idx.copy_to_host(&mut got).expect("dl");
    check_ascending(&input, &got);
}

#[test]
#[ignore]
fn argsort_i16_descending() {
    let (ctx, stream) = setup();
    let row_len: i32 = 8;
    let input: Vec<i16> = vec![-5, 3, -8, 1, 9, -2, 7, 0];
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_idx: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, input.len()).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_i16_run(
            1,
            row_len,
            1, // descending
            dev_in.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut got = vec![0i32; input.len()];
    dev_idx.copy_to_host(&mut got).expect("dl");
    // Descending check.
    for i in 1..got.len() {
        let a = input[got[i - 1] as usize];
        let b = input[got[i] as usize];
        assert!(a >= b, "argsort descending broken @ {i}: {a} < {b}");
    }
}

#[test]
#[ignore]
fn argsort_f16_basic() {
    let (ctx, stream) = setup();
    let row_len: i32 = 16;
    let input: Vec<f16> = (0..row_len as usize)
        .map(|i| f16::from_f32(((i as f32) * 1.7).sin()))
        .collect();
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_idx: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, input.len()).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_f16_run(
            1,
            row_len,
            0,
            dev_in.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut got = vec![0i32; input.len()];
    dev_idx.copy_to_host(&mut got).expect("dl");
    for i in 1..got.len() {
        let a = input[got[i - 1] as usize].to_f32();
        let b = input[got[i] as usize].to_f32();
        assert!(a <= b, "argsort_f16 broken @ {i}: {a} > {b}");
    }
}

#[test]
#[ignore]
fn argsort_bf16_basic() {
    let (ctx, stream) = setup();
    let row_len: i32 = 16;
    let input: Vec<bf16> = (0..row_len as usize)
        .map(|i| bf16::from_f32(((i as f32) * 0.9 - 7.0).cos()))
        .collect();
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_idx: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, input.len()).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_bf16_run(
            1,
            row_len,
            0,
            dev_in.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut got = vec![0i32; input.len()];
    dev_idx.copy_to_host(&mut got).expect("dl");
    for i in 1..got.len() {
        let a = input[got[i - 1] as usize].to_f32();
        let b = input[got[i] as usize].to_f32();
        assert!(a <= b, "argsort_bf16 broken @ {i}: {a} > {b}");
    }
}

#[test]
#[ignore]
fn argsort_fp8e4m3_basic() {
    let (ctx, stream) = setup();
    // FP8 E4M3 raw byte encodings. Choose distinct values to avoid
    // ties: a mix of positive small / zero / negative.
    //   0x00 =  0.0
    //   0x38 =  1.0
    //   0xB8 = -1.0
    //   0x40 =  2.0
    //   0xC0 = -2.0
    //   0x30 =  0.5
    //   0x4C =  3.5
    //   0x44 =  2.5
    let row_len: i32 = 8;
    let input: Vec<u8> = vec![0x00, 0x38, 0xB8, 0x40, 0xC0, 0x30, 0x4C, 0x44];
    // Float-equivalent values for verification.
    let float_eq: Vec<f32> = vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5, 3.5, 2.5];

    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_idx: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, input.len()).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_fp8e4m3_run(
            1,
            row_len,
            0,
            dev_in.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut got = vec![0i32; input.len()];
    dev_idx.copy_to_host(&mut got).expect("dl");
    // Check ordering using float-equivalent values.
    for i in 1..got.len() {
        let a = float_eq[got[i - 1] as usize];
        let b = float_eq[got[i] as usize];
        assert!(a <= b, "argsort_fp8e4m3 broken @ {i}: {a} > {b} (got {got:?})");
    }
    // Expected ordering by float value: -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.5, 3.5
    // -> indices: 4, 2, 0, 5, 1, 3, 7, 6
    assert_eq!(got, vec![4, 2, 0, 5, 1, 3, 7, 6]);
}

#[test]
fn can_implement_rejects_large_row() {
    let s = unsafe { baracuda_kernels_sys::baracuda_kernels_argsort_u32_can_implement(2, 2048) };
    assert_ne!(s, 0, "row_len > 1024 should be rejected (status 3)");
    let s = unsafe { baracuda_kernels_sys::baracuda_kernels_argsort_u32_can_implement(2, 1024) };
    assert_eq!(s, 0);
    let s = unsafe { baracuda_kernels_sys::baracuda_kernels_argsort_fp8e4m3_can_implement(-1, 8) };
    assert_ne!(s, 0, "negative batch should be rejected");
}
