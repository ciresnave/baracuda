//! Phase 40 (Fuel 6c.4 Gap 6b spillover) — integer value-dtype fanout
//! for indexing ops.
//!
//! Direct-FFI smoke for:
//!   * `scatter_<int>_run` — pure-assign, last-writer-wins
//!   * `index_add_<int>_run` — atomicAdd-Σ (native-atomic-int dtypes only)
//!
//! `gather` / `index_select` integer dtypes are not exercised in this
//! file (read-only ops; kernel structure identical to fp counterparts
//! already covered by `gather_smoke.rs` / `index_select_smoke.rs`).

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
fn scatter_u8_basic() {
    let (ctx, stream) = setup();
    // dst[idx[i]] = updates[i] along scatter_dim=0
    // Shape: out=[5], updates=[3], index=[3] with values {2, 4, 0}
    let updates: Vec<u8> = vec![10, 20, 30];
    let index: Vec<i32> = vec![2, 4, 0];
    let mut out: Vec<u8> = vec![0; 5];

    let dev_upd = DeviceBuffer::from_slice(&ctx, &updates).expect("up upd");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &index).expect("up idx");
    let mut dev_out = DeviceBuffer::from_slice(&ctx, &out).expect("up out");

    let upd_shape: [i32; 1] = [3];
    let stride_upd: [i64; 1] = [1];
    let stride_index: [i64; 1] = [1];
    let stride_out: [i64; 1] = [1];

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_scatter_u8_run(
            3,
            1,
            0,
            5,
            upd_shape.as_ptr(),
            stride_upd.as_ptr(),
            stride_index.as_ptr(),
            stride_out.as_ptr(),
            dev_upd.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice().as_raw().0 as *const c_void,
            dev_out.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    dev_out.copy_to_host(&mut out).expect("dl");
    // Expected: out[2]=10, out[4]=20, out[0]=30, rest=0.
    assert_eq!(out, vec![30, 0, 10, 0, 20]);
}

#[test]
#[ignore]
fn scatter_i16_basic() {
    let (ctx, stream) = setup();
    let updates: Vec<i16> = vec![-100, 200, -300];
    let index: Vec<i32> = vec![1, 0, 3];
    let mut out: Vec<i16> = vec![0; 4];

    let dev_upd = DeviceBuffer::from_slice(&ctx, &updates).expect("up upd");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &index).expect("up idx");
    let mut dev_out = DeviceBuffer::from_slice(&ctx, &out).expect("up out");

    let upd_shape: [i32; 1] = [3];
    let stride_upd: [i64; 1] = [1];
    let stride_index: [i64; 1] = [1];
    let stride_out: [i64; 1] = [1];

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_scatter_i16_run(
            3, 1, 0, 4,
            upd_shape.as_ptr(),
            stride_upd.as_ptr(),
            stride_index.as_ptr(),
            stride_out.as_ptr(),
            dev_upd.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice().as_raw().0 as *const c_void,
            dev_out.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    dev_out.copy_to_host(&mut out).expect("dl");
    // out[1]=-100, out[0]=200, out[3]=-300, out[2]=0
    assert_eq!(out, vec![200, -100, 0, -300]);
}

#[test]
#[ignore]
fn scatter_i64_basic() {
    let (ctx, stream) = setup();
    let updates: Vec<i64> = vec![1_000_000_000_000, -2_000_000_000_000];
    let index: Vec<i32> = vec![3, 1];
    let mut out: Vec<i64> = vec![0; 5];

    let dev_upd = DeviceBuffer::from_slice(&ctx, &updates).expect("up upd");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &index).expect("up idx");
    let mut dev_out = DeviceBuffer::from_slice(&ctx, &out).expect("up out");

    let upd_shape: [i32; 1] = [2];
    let stride_upd: [i64; 1] = [1];
    let stride_index: [i64; 1] = [1];
    let stride_out: [i64; 1] = [1];

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_scatter_i64_run(
            2, 1, 0, 5,
            upd_shape.as_ptr(),
            stride_upd.as_ptr(),
            stride_index.as_ptr(),
            stride_out.as_ptr(),
            dev_upd.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice().as_raw().0 as *const c_void,
            dev_out.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    dev_out.copy_to_host(&mut out).expect("dl");
    assert_eq!(
        out,
        vec![0, -2_000_000_000_000, 0, 1_000_000_000_000, 0]
    );
}

#[test]
#[ignore]
fn index_add_i32_basic() {
    let (ctx, stream) = setup();
    // dst[idx[i]] += src[i] along add_dim=0; pre-populate dst with 100.
    // src shape=[3], idx=[3], dst=[5].
    let src: Vec<i32> = vec![10, 20, 30];
    let idx: Vec<i32> = vec![1, 1, 4]; // duplicate index at 1 -> 10+20=30 accumulated
    let mut dst: Vec<i32> = vec![100; 5];

    let dev_src = DeviceBuffer::from_slice(&ctx, &src).expect("up src");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &idx).expect("up idx");
    let mut dev_dst = DeviceBuffer::from_slice(&ctx, &dst).expect("up dst");

    let src_shape: [i32; 1] = [3];
    let stride_src: [i64; 1] = [1];
    let stride_dst: [i64; 1] = [1];

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_index_add_i32_run(
            3, 1, 0, 5,
            src_shape.as_ptr(),
            stride_src.as_ptr(),
            stride_dst.as_ptr(),
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    dev_dst.copy_to_host(&mut dst).expect("dl");
    // dst[0]=100, dst[1]=100+10+20=130, dst[2]=100, dst[3]=100, dst[4]=100+30=130
    assert_eq!(dst, vec![100, 130, 100, 100, 130]);
}

#[test]
#[ignore]
fn index_add_u32_basic() {
    let (ctx, stream) = setup();
    let src: Vec<u32> = vec![5, 7, 11];
    let idx: Vec<i32> = vec![0, 2, 0]; // duplicate at 0 -> 5+11=16
    let mut dst: Vec<u32> = vec![1; 4];

    let dev_src = DeviceBuffer::from_slice(&ctx, &src).expect("up");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &idx).expect("up");
    let mut dev_dst = DeviceBuffer::from_slice(&ctx, &dst).expect("up");

    let src_shape: [i32; 1] = [3];
    let stride_src: [i64; 1] = [1];
    let stride_dst: [i64; 1] = [1];

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_index_add_u32_run(
            3, 1, 0, 4,
            src_shape.as_ptr(),
            stride_src.as_ptr(),
            stride_dst.as_ptr(),
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    dev_dst.copy_to_host(&mut dst).expect("dl");
    // dst[0]=1+5+11=17, dst[1]=1, dst[2]=1+7=8, dst[3]=1
    assert_eq!(dst, vec![17, 1, 8, 1]);
}

#[test]
#[ignore]
fn index_add_i64_basic() {
    let (ctx, stream) = setup();
    let src: Vec<i64> = vec![1_000_000, -500_000];
    let idx: Vec<i32> = vec![2, 2];
    let mut dst: Vec<i64> = vec![0; 4];

    let dev_src = DeviceBuffer::from_slice(&ctx, &src).expect("up");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &idx).expect("up");
    let mut dev_dst = DeviceBuffer::from_slice(&ctx, &dst).expect("up");

    let src_shape: [i32; 1] = [2];
    let stride_src: [i64; 1] = [1];
    let stride_dst: [i64; 1] = [1];

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_index_add_i64_run(
            2, 1, 0, 4,
            src_shape.as_ptr(),
            stride_src.as_ptr(),
            stride_dst.as_ptr(),
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    dev_dst.copy_to_host(&mut dst).expect("dl");
    // dst[2] = 0 + 1_000_000 + (-500_000) = 500_000; rest=0
    assert_eq!(dst, vec![0, 0, 500_000, 0]);
}

#[test]
#[ignore]
fn index_select_u32_basic() {
    let (ctx, stream) = setup();
    // out[..., j, ...] = src[..., idx[j], ...] along select_dim=0.
    let src: Vec<u32> = vec![100, 200, 300, 400, 500];
    let idx: Vec<i32> = vec![3, 0, 4];
    let mut out: Vec<u32> = vec![0; 3];

    let dev_src = DeviceBuffer::from_slice(&ctx, &src).expect("up");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &idx).expect("up");
    let mut dev_out = DeviceBuffer::from_slice(&ctx, &out).expect("up");

    let out_shape: [i32; 1] = [3];
    let stride_src: [i64; 1] = [1];
    let stride_out: [i64; 1] = [1];

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_index_select_u32_run(
            3, 1, 0, 5,
            out_shape.as_ptr(),
            stride_src.as_ptr(),
            stride_out.as_ptr(),
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice().as_raw().0 as *const c_void,
            dev_out.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    dev_out.copy_to_host(&mut out).expect("dl");
    assert_eq!(out, vec![400, 100, 500]);
}

#[test]
#[ignore]
fn gather_u8_basic() {
    let (ctx, stream) = setup();
    // out[i] = src[index[i]] along gather_dim=0 (matching shapes).
    let src: Vec<u8> = vec![10, 20, 30, 40, 50];
    let index: Vec<i32> = vec![4, 2, 0];
    let mut out: Vec<u8> = vec![0; 3];

    let dev_src = DeviceBuffer::from_slice(&ctx, &src).expect("up");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &index).expect("up");
    let mut dev_out = DeviceBuffer::from_slice(&ctx, &out).expect("up");

    let out_shape: [i32; 1] = [3];
    let stride_src: [i64; 1] = [1];
    let stride_index: [i64; 1] = [1];
    let stride_out: [i64; 1] = [1];

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_gather_u8_run(
            3, 1, 0, 5,
            out_shape.as_ptr(),
            stride_src.as_ptr(),
            stride_index.as_ptr(),
            stride_out.as_ptr(),
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice().as_raw().0 as *const c_void,
            dev_out.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    dev_out.copy_to_host(&mut out).expect("dl");
    assert_eq!(out, vec![50, 30, 10]);
}
