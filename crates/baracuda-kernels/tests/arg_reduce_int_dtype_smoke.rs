//! Phase 37 Gap 1b — direct-FFI smoke for integer-dtype argmin / argmax
//! (`arg_reduce_{argmax, argmin}_<int_dt>_{i32, i64}_run`).
//!
//! Spot-covers one i32 idx and one i64 idx case per integer input
//! dtype. Ties broken by first-occurrence (smallest index wins) —
//! matches FP family.
//!
//! Shape: `[3, 4]` reducing axis=1; output is `[3, 1]` (one idx per row).

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

const ROWS: i32 = 3;
const COLS: i32 = 4;

fn out_shape() -> [i32; 2] { [ROWS, 1] }
fn in_stride() -> [i64; 2] { [COLS as i64, 1] }
fn out_stride() -> [i64; 2] { [1, 1] }

type ArgReduceFn = unsafe extern "C" fn(
    i64, i32, *const i32, *const i64, *const i64, i32, i32, i64,
    *const c_void, *mut c_void, *mut c_void, usize, *mut c_void,
) -> i32;

unsafe fn run_arg_reduce(
    f: ArgReduceFn,
    stream: &Stream,
    src_ptr: *const c_void,
    dst_ptr: *mut c_void,
) {
    let out_sh = out_shape();
    let in_st = in_stride();
    let out_st = out_stride();
    let status = unsafe {
        f(
            ROWS as i64,
            2,
            out_sh.as_ptr(),
            in_st.as_ptr(),
            out_st.as_ptr(),
            1,        // reduce_axis
            COLS,     // reduce_extent
            1,        // reduce_stride_x
            src_ptr,
            dst_ptr,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "arg_reduce returned {status}");
}

// ----------------------------------------------------------------------------
// argmax / argmin × i32 idx, sampled across all 6 input dtypes.
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn ffi_argmax_u8_i32() {
    let (ctx, stream) = setup();
    let host_src: Vec<u8> = vec![
        1, 2, 3, 4,        // argmax=3
        10, 200, 30, 40,   // argmax=1
        255, 1, 255, 255,  // argmax=0 (first-occurrence tie)
    ];
    let expected: Vec<i32> = vec![3, 1, 0];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_arg_reduce(
            baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmax_u8_i32_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0i32; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "argmax u8 i32");
}

#[test]
#[ignore]
fn ffi_argmin_i8_i32() {
    let (ctx, stream) = setup();
    let host_src: Vec<i8> = vec![
        1, -1, 3, 4,       // argmin=1
        -10, -20, -30, 40, // argmin=2
        5, 5, 5, 5,        // argmin=0 (first-occurrence tie)
    ];
    let expected: Vec<i32> = vec![1, 2, 0];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_arg_reduce(
            baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmin_i8_i32_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0i32; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "argmin i8 i32");
}

#[test]
#[ignore]
fn ffi_argmax_u32_i32() {
    let (ctx, stream) = setup();
    let host_src: Vec<u32> = vec![
        1, 2, 3, 4,
        10, 200, 30, 40,
        u32::MAX, 1, u32::MAX, u32::MAX,
    ];
    let expected: Vec<i32> = vec![3, 1, 0];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_arg_reduce(
            baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmax_u32_i32_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0i32; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "argmax u32 i32");
}

#[test]
#[ignore]
fn ffi_argmin_i16_i32() {
    let (ctx, stream) = setup();
    let host_src: Vec<i16> = vec![
        1000, -1, 3000, 4000,
        -10, -20, -30, 4000,
        i16::MIN, 5, i16::MIN, i16::MIN,
    ];
    let expected: Vec<i32> = vec![1, 2, 0];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_arg_reduce(
            baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmin_i16_i32_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0i32; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "argmin i16 i32");
}

#[test]
#[ignore]
fn ffi_argmax_i32_i32() {
    let (ctx, stream) = setup();
    let host_src: Vec<i32> = vec![
        1, 2, 3, 4,
        10, 200, 30, 40,
        i32::MAX, 1, i32::MAX, i32::MAX,
    ];
    let expected: Vec<i32> = vec![3, 1, 0];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_arg_reduce(
            baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmax_i32_i32_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0i32; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "argmax i32 i32");
}

#[test]
#[ignore]
fn ffi_argmin_i64_i32() {
    let (ctx, stream) = setup();
    let host_src: Vec<i64> = vec![
        1_000_000_000_000, -1, 3, 4,
        -10, -20, -30, 4,
        i64::MIN, 5, i64::MIN, i64::MIN,
    ];
    let expected: Vec<i32> = vec![1, 2, 0];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_arg_reduce(
            baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmin_i64_i32_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0i32; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "argmin i64 i32");
}

// ----------------------------------------------------------------------------
// i64 idx output — verify the idx-dtype narrowing is wired correctly for
// at least one input dtype.
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn ffi_argmax_i32_i64() {
    let (ctx, stream) = setup();
    let host_src: Vec<i32> = vec![
        1, 2, 3, 4,
        10, 200, 30, 40,
        i32::MAX, 1, i32::MAX, i32::MAX,
    ];
    let expected: Vec<i64> = vec![3, 1, 0];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<i64> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_arg_reduce(
            baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmax_i32_i64_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0i64; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "argmax i32 i64");
}

#[test]
#[ignore]
fn ffi_argmin_u8_i64() {
    let (ctx, stream) = setup();
    let host_src: Vec<u8> = vec![
        1, 2, 3, 4,        // argmin=0
        10, 5, 100, 200,   // argmin=1
        5, 5, 5, 5,        // argmin=0 (tie)
    ];
    let expected: Vec<i64> = vec![0, 1, 0];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<i64> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_arg_reduce(
            baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmin_u8_i64_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0i64; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "argmin u8 i64");
}
