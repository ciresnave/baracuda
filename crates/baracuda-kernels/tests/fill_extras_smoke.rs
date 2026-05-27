//! Phase 36 (Fuel ask Gap 4) — direct-FFI smoke for the new fill
//! dtypes (`u32`, `i16`, FP8 E4M3) + a representative strided case.

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn fill_u32_contig() {
    let (ctx, stream) = setup();
    let numel = 1024usize;
    let mut dev_y: DeviceBuffer<u32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let value: u32 = 0xDEAD_BEEF;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fill_u32_run(
            numel as i64,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            value,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0u32; numel];
    dev_y.copy_to_host(&mut got).expect("dl");
    for (i, g) in got.iter().enumerate() {
        assert_eq!(*g, value, "fill_u32 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn fill_i16_contig() {
    let (ctx, stream) = setup();
    let numel = 1024usize;
    let mut dev_y: DeviceBuffer<i16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let value: i16 = -12345;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fill_i16_run(
            numel as i64,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            value,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0i16; numel];
    dev_y.copy_to_host(&mut got).expect("dl");
    for (i, g) in got.iter().enumerate() {
        assert_eq!(*g, value, "fill_i16 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn fill_fp8e4m3_contig() {
    let (ctx, stream) = setup();
    let numel = 1024usize;
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    // FP8 E4M3 encoding for 1.0 = 0x38 (sign=0, exp=0111, mantissa=000).
    let value: u8 = 0x38;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fill_fp8e4m3_run(
            numel as i64,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            value,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0u8; numel];
    dev_y.copy_to_host(&mut got).expect("dl");
    for (i, g) in got.iter().enumerate() {
        assert_eq!(*g, value, "fill_fp8e4m3 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn fill_f32_strided_skip_every_other_row() {
    // Logical shape [4, 8], stride_y = [16, 1]. The kernel writes
    // every other "row" of a [4, 16] backing buffer — the first 8
    // cells of each 16-cell row receive the value; the trailing 8
    // are left at 0.
    let (ctx, stream) = setup();
    let shape: [i32; 2] = [4, 8];
    let stride_y: [i64; 2] = [16, 1];
    let numel = (shape[0] * shape[1]) as usize;
    let backing = (shape[0] as usize) * 16;
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, backing).expect("alloc");

    let value: f32 = 7.5;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fill_f32_strided_run(
            numel as i64,
            2,
            shape.as_ptr(),
            stride_y.as_ptr(),
            value,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; backing];
    dev_y.copy_to_host(&mut got).expect("dl");
    for row in 0..(shape[0] as usize) {
        for col in 0..16 {
            let expected = if col < (shape[1] as usize) {
                value
            } else {
                0.0
            };
            let i = row * 16 + col;
            assert_eq!(got[i], expected, "row={row} col={col} got={} expected={expected}", got[i]);
        }
    }
}

#[test]
fn can_implement_basic() {
    let s = unsafe { baracuda_kernels_sys::baracuda_kernels_fill_u32_can_implement(8, core::ptr::null()) };
    assert_eq!(s, 0);
    let s = unsafe { baracuda_kernels_sys::baracuda_kernels_fill_i16_can_implement(-1, core::ptr::null()) };
    assert_ne!(s, 0, "should reject negative numel");
    let s = unsafe { baracuda_kernels_sys::baracuda_kernels_fill_fp8e4m3_can_implement(0, core::ptr::null()) };
    assert_eq!(s, 0);
    let s = unsafe { baracuda_kernels_sys::baracuda_kernels_fill_f32_strided_can_implement(8, 2) };
    assert_eq!(s, 0);
    let s = unsafe { baracuda_kernels_sys::baracuda_kernels_fill_f32_strided_can_implement(8, 99) };
    assert_ne!(s, 0, "should reject rank > MAX_RANK");
}
