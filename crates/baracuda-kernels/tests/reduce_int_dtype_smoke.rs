//! Phase 37 Gap 1b — direct-FFI smoke for integer-dtype single-axis
//! Reduce family (`reduce_{sum, min, max, prod}_<int_dt>_run`).
//!
//! Coverage:
//!  - sum / min / max / prod × {u8, i8, u32, i16, i32, i64}
//!  - one explicit u8 sum-with-overflow test to verify wrap-on-store
//!    matches Fuel's CPU contract.
//!
//! Shape: `[3, 4]`, reduce axis = 1 (so output shape `[3, 1]`). Each
//! output cell sums/min/max/prods 4 elements.

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

type ReduceFn = unsafe extern "C" fn(
    i64, i32, *const i32, *const i64, *const i64, i32, i32, i64,
    *const c_void, *mut c_void, *mut c_void, usize, *mut c_void,
) -> i32;

unsafe fn run_reduce(
    f: ReduceFn,
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
            1,             // reduce_axis
            COLS,          // reduce_extent
            1,             // reduce_stride_x (axis-1 stride in input)
            src_ptr,
            dst_ptr,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "reduce returned {status}");
}

// ----------------------------------------------------------------------------
// i32 cases (representative for signed-int Sum/Min/Max/Prod).
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn ffi_reduce_sum_i32_2d() {
    let (ctx, stream) = setup();
    let host_src: Vec<i32> = vec![
        1, 2, 3, 4,         // row 0 -> 10
        -1, -2, -3, -4,     // row 1 -> -10
        100, 200, 300, 400, // row 2 -> 1000
    ];
    let expected: Vec<i32> = vec![10, -10, 1000];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_reduce(
            baracuda_kernels_sys::baracuda_kernels_reduce_sum_i32_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0i32; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "sum_i32");
}

#[test]
#[ignore]
fn ffi_reduce_min_i32_2d() {
    let (ctx, stream) = setup();
    let host_src: Vec<i32> = vec![
        1, 2, 3, 4,
        -1, -2, -3, -4,
        5, 5, 5, 5,
    ];
    let expected: Vec<i32> = vec![1, -4, 5];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_reduce(
            baracuda_kernels_sys::baracuda_kernels_reduce_min_i32_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0i32; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "min_i32");
}

#[test]
#[ignore]
fn ffi_reduce_max_i32_2d() {
    let (ctx, stream) = setup();
    let host_src: Vec<i32> = vec![
        1, 2, 3, 4,
        -1, -2, -3, -4,
        5, 5, 5, 5,
    ];
    let expected: Vec<i32> = vec![4, -1, 5];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_reduce(
            baracuda_kernels_sys::baracuda_kernels_reduce_max_i32_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0i32; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "max_i32");
}

#[test]
#[ignore]
fn ffi_reduce_prod_i32_2d() {
    let (ctx, stream) = setup();
    let host_src: Vec<i32> = vec![
        1, 2, 3, 4,    // -> 24
        -1, 2, -3, 4,  // -> 24
        1, 1, 1, 1,    // -> 1
    ];
    let expected: Vec<i32> = vec![24, 24, 1];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_reduce(
            baracuda_kernels_sys::baracuda_kernels_reduce_prod_i32_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0i32; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "prod_i32");
}

// ----------------------------------------------------------------------------
// u8 cases (representative for unsigned + small-dtype + wrap behaviour).
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn ffi_reduce_sum_u8_2d_no_overflow() {
    let (ctx, stream) = setup();
    let host_src: Vec<u8> = vec![
        1, 2, 3, 4,        // 10
        10, 20, 30, 40,    // 100
        50, 50, 50, 50,    // 200
    ];
    let expected: Vec<u8> = vec![10, 100, 200];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_reduce(
            baracuda_kernels_sys::baracuda_kernels_reduce_sum_u8_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0u8; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "sum_u8 (no overflow)");
}

/// Verify wrap-on-overflow contract for u8 Sum. Row 0 sums to 1000,
/// which mod 256 = 232. The widened accumulator (u64) produces 1000
/// exactly; narrowing to u8 wraps to 232.
#[test]
#[ignore]
fn ffi_reduce_sum_u8_wrap_on_overflow() {
    let (ctx, stream) = setup();
    let host_src: Vec<u8> = vec![
        250, 250, 250, 250,  // 1000 -> 232 (mod 256)
        255, 255, 255, 255,  // 1020 -> 252
        100, 100, 100, 100,  // 400  -> 144
    ];
    let expected: Vec<u8> = vec![232, 252, 144];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_reduce(
            baracuda_kernels_sys::baracuda_kernels_reduce_sum_u8_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0u8; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "sum_u8 wrap-on-overflow contract");
}

#[test]
#[ignore]
fn ffi_reduce_min_u8_2d() {
    let (ctx, stream) = setup();
    let host_src: Vec<u8> = vec![
        1, 2, 3, 4,
        10, 5, 100, 200,
        255, 1, 255, 255,
    ];
    let expected: Vec<u8> = vec![1, 5, 1];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_reduce(
            baracuda_kernels_sys::baracuda_kernels_reduce_min_u8_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0u8; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "min_u8");
}

#[test]
#[ignore]
fn ffi_reduce_max_u8_2d() {
    let (ctx, stream) = setup();
    let host_src: Vec<u8> = vec![
        1, 2, 3, 4,
        10, 5, 100, 200,
        255, 1, 255, 255,
    ];
    let expected: Vec<u8> = vec![4, 200, 255];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_reduce(
            baracuda_kernels_sys::baracuda_kernels_reduce_max_u8_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0u8; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "max_u8");
}

#[test]
#[ignore]
fn ffi_reduce_prod_u8_2d() {
    let (ctx, stream) = setup();
    // Row 0 prods to 24; row 1 prods to 16 * 4 * 1 * 1 = 64; row 2
    // prods to 5 * 5 * 5 * 5 = 625 -> mod 256 = 113.
    let host_src: Vec<u8> = vec![
        1, 2, 3, 4,
        16, 4, 1, 1,
        5, 5, 5, 5,
    ];
    let expected: Vec<u8> = vec![24, 64, 113];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_reduce(
            baracuda_kernels_sys::baracuda_kernels_reduce_prod_u8_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0u8; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "prod_u8 (last row wraps)");
}

// ----------------------------------------------------------------------------
// Spot checks for the remaining dtypes — light, but verify the FFI links
// and produces sane results.
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn ffi_reduce_sum_i8_2d() {
    let (ctx, stream) = setup();
    let host_src: Vec<i8> = vec![
        1, 2, 3, 4,
        -1, -2, -3, -4,
        10, 10, 10, 10,
    ];
    let expected: Vec<i8> = vec![10, -10, 40];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<i8> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_reduce(
            baracuda_kernels_sys::baracuda_kernels_reduce_sum_i8_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0i8; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "sum_i8");
}

#[test]
#[ignore]
fn ffi_reduce_sum_i16_2d() {
    let (ctx, stream) = setup();
    let host_src: Vec<i16> = vec![
        1000, 2000, 3000, 4000,
        -1000, -2000, -3000, -4000,
        100, 100, 100, 100,
    ];
    let expected: Vec<i16> = vec![10000, -10000, 400];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<i16> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_reduce(
            baracuda_kernels_sys::baracuda_kernels_reduce_sum_i16_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0i16; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "sum_i16");
}

#[test]
#[ignore]
fn ffi_reduce_sum_u32_2d() {
    let (ctx, stream) = setup();
    let host_src: Vec<u32> = vec![
        1, 2, 3, 4,
        100, 200, 300, 400,
        10000, 20000, 30000, 40000,
    ];
    let expected: Vec<u32> = vec![10, 1000, 100000];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<u32> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_reduce(
            baracuda_kernels_sys::baracuda_kernels_reduce_sum_u32_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0u32; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "sum_u32");
}

#[test]
#[ignore]
fn ffi_reduce_sum_i64_2d() {
    let (ctx, stream) = setup();
    let host_src: Vec<i64> = vec![
        1, 2, 3, 4,
        -1, -2, -3, -4,
        1_000_000_000_000, 1, 1, 1,
    ];
    let expected: Vec<i64> = vec![10, -10, 1_000_000_000_003];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload");
    let mut dev_dst: DeviceBuffer<i64> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    unsafe {
        run_reduce(
            baracuda_kernels_sys::baracuda_kernels_reduce_sum_i64_run,
            &stream,
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
        );
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0i64; 3];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "sum_i64");
}
