//! Phase 72 direct-FFI smoke tests for the strided sibling exports
//! (`_strided_run` / `_strided_can_implement`) on the 7 op families that
//! lacked them: rms_norm, layer_norm, softmax, log_softmax, flip, roll,
//! permute. FW only (the BW siblings share the same launcher as the
//! existing BW which is already strided-correct; FW coverage proves
//! linker resolution + stride honoring).
//!
//! Each test:
//!   1. Constructs a non-contig input view (transposed-of-larger / stride-2).
//!   2. Calls the `_strided_run` FFI symbol directly (bypasses the safe
//!      wrapper, which still calls the non-strided `_run` name).
//!   3. Validates the result matches a CPU reference computed against the
//!      same non-contig view.
//!
//! `#[ignore]` per the GPU smoke suite convention. Run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test strided_siblings_ffi_smoke -- --ignored`.

#![cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels_sys as sys;
use core::ffi::c_void;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// RmsNorm FW strided sibling, f32. Non-contig input: 2D buffer `[rows,
/// 2*cols]` with last-axis stride 2 over an outer-padded layout. RMSNorm
/// applies along the last axis (norm_axes_mask = 1 << (rank-1)).
#[test]
#[ignore]
fn rms_norm_f32_strided_smoke() {
    let (ctx, stream) = setup();
    let rows = 4i32;
    let cols = 64i32;
    let outer_stride = 2i64 * cols as i64;
    let buf_len = (rows as i64 * outer_stride) as usize;
    let eps = 1e-5f32;

    // Big buffer; only the even columns are the "live" view.
    let host_x: Vec<f32> = (0..buf_len).map(|i| ((i % 17) as f32) * 0.125 - 1.0).collect();

    // Reference: per-row RMS over the live cols only.
    let mut expected = vec![0f32; (rows as usize) * (cols as usize)];
    for r in 0..rows as usize {
        let mut sumsq = 0.0f32;
        for c in 0..cols as usize {
            let x = host_x[r * outer_stride as usize + c * 2];
            sumsq += x * x;
        }
        let rms = (sumsq / cols as f32 + eps).sqrt();
        for c in 0..cols as usize {
            let x = host_x[r * outer_stride as usize + c * 2];
            expected[r * cols as usize + c] = x / rms;
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (rows as usize) * (cols as usize)).expect("alloc y");

    let shape = [rows, cols];
    let stride_x = [outer_stride, 2i64];
    let stride_y = [cols as i64, 1i64];
    let stride_rms = [1i64, 0i64];
    let numel = rows as i64 * cols as i64;
    let mask: i32 = 1 << (shape.len() - 1) as i32;

    let status = unsafe {
        sys::baracuda_kernels_rms_norm_f32_strided_run(
            eps,
            numel,
            shape.len() as i32,
            shape.as_ptr(),
            stride_x.as_ptr(),
            stride_y.as_ptr(),
            stride_rms.as_ptr(),
            mask,
            cols,
            dev_x.as_slice().as_raw().0 as *const c_void,
            core::ptr::null(),
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "rms_norm_f32_strided_run returned {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (rows as usize) * (cols as usize)];
    dev_y.copy_to_host(&mut got).expect("download");

    let tol = 1e-5f32;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= tol.max(e.abs() * tol),
            "rms_norm_f32_strided @ {i}: got {g} expected {e}"
        );
    }
}

/// LayerNorm FW strided sibling, f32. Same stride-2 layout as rms_norm.
#[test]
#[ignore]
fn layer_norm_f32_strided_smoke() {
    let (ctx, stream) = setup();
    let rows = 4i32;
    let cols = 64i32;
    let outer_stride = 2i64 * cols as i64;
    let buf_len = (rows as i64 * outer_stride) as usize;
    let eps = 1e-5f32;

    let host_x: Vec<f32> = (0..buf_len).map(|i| ((i % 13) as f32) * 0.0625 - 0.5).collect();

    let mut expected = vec![0f32; (rows as usize) * (cols as usize)];
    for r in 0..rows as usize {
        let mut sum = 0.0f32;
        for c in 0..cols as usize {
            sum += host_x[r * outer_stride as usize + c * 2];
        }
        let mean = sum / cols as f32;
        let mut var = 0.0f32;
        for c in 0..cols as usize {
            let d = host_x[r * outer_stride as usize + c * 2] - mean;
            var += d * d;
        }
        let inv_std = 1.0 / (var / cols as f32 + eps).sqrt();
        for c in 0..cols as usize {
            let x = host_x[r * outer_stride as usize + c * 2];
            expected[r * cols as usize + c] = (x - mean) * inv_std;
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (rows as usize) * (cols as usize)).expect("alloc y");

    let shape = [rows, cols];
    let stride_x = [outer_stride, 2i64];
    let stride_y = [cols as i64, 1i64];
    let stride_save = [1i64, 0i64];
    let numel = rows as i64 * cols as i64;
    let mask: i32 = 1 << (shape.len() - 1) as i32;

    let status = unsafe {
        sys::baracuda_kernels_layer_norm_f32_strided_run(
            eps,
            numel,
            shape.len() as i32,
            shape.as_ptr(),
            stride_x.as_ptr(),
            stride_y.as_ptr(),
            stride_save.as_ptr(),
            mask,
            cols,
            dev_x.as_slice().as_raw().0 as *const c_void,
            core::ptr::null(),
            core::ptr::null(),
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "layer_norm_f32_strided_run returned {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (rows as usize) * (cols as usize)];
    dev_y.copy_to_host(&mut got).expect("download");

    let tol = 1e-5f32;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= tol.max(e.abs() * tol),
            "layer_norm_f32_strided @ {i}: got {g} expected {e}"
        );
    }
}

/// Softmax FW strided sibling, f32. Non-contig input over last axis with
/// stride 2 (over a 2x-padded buffer).
#[test]
#[ignore]
fn softmax_f32_strided_smoke() {
    let (ctx, stream) = setup();
    let rows = 4i32;
    let cols = 32i32;
    let outer_stride = 2i64 * cols as i64;
    let buf_len = (rows as i64 * outer_stride) as usize;

    let host_x: Vec<f32> = (0..buf_len).map(|i| ((i % 11) as f32) * 0.5 - 2.0).collect();

    let mut expected = vec![0f32; (rows as usize) * (cols as usize)];
    for r in 0..rows as usize {
        let mut max_v = f32::NEG_INFINITY;
        for c in 0..cols as usize {
            let v = host_x[r * outer_stride as usize + c * 2];
            if v > max_v { max_v = v; }
        }
        let mut sum = 0.0f32;
        for c in 0..cols as usize {
            sum += (host_x[r * outer_stride as usize + c * 2] - max_v).exp();
        }
        for c in 0..cols as usize {
            let v = host_x[r * outer_stride as usize + c * 2];
            expected[r * cols as usize + c] = (v - max_v).exp() / sum;
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (rows as usize) * (cols as usize)).expect("alloc y");

    let shape = [rows, cols];
    let stride_x = [outer_stride, 2i64];
    let stride_y = [cols as i64, 1i64];
    let numel = rows as i64 * cols as i64;
    let softmax_axis = (shape.len() - 1) as i32;

    let status = unsafe {
        sys::baracuda_kernels_softmax_f32_strided_run(
            numel,
            shape.len() as i32,
            shape.as_ptr(),
            stride_x.as_ptr(),
            stride_y.as_ptr(),
            softmax_axis,
            cols,
            2i64,
            1i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "softmax_f32_strided_run returned {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (rows as usize) * (cols as usize)];
    dev_y.copy_to_host(&mut got).expect("download");

    let tol = 1e-5f32;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= tol.max(e.abs() * tol),
            "softmax_f32_strided @ {i}: got {g} expected {e}"
        );
    }
}

/// LogSoftmax FW strided sibling, f32.
#[test]
#[ignore]
fn log_softmax_f32_strided_smoke() {
    let (ctx, stream) = setup();
    let rows = 4i32;
    let cols = 32i32;
    let outer_stride = 2i64 * cols as i64;
    let buf_len = (rows as i64 * outer_stride) as usize;

    let host_x: Vec<f32> = (0..buf_len).map(|i| ((i % 9) as f32) * 0.5 - 1.5).collect();

    let mut expected = vec![0f32; (rows as usize) * (cols as usize)];
    for r in 0..rows as usize {
        let mut max_v = f32::NEG_INFINITY;
        for c in 0..cols as usize {
            let v = host_x[r * outer_stride as usize + c * 2];
            if v > max_v { max_v = v; }
        }
        let mut sum = 0.0f32;
        for c in 0..cols as usize {
            sum += (host_x[r * outer_stride as usize + c * 2] - max_v).exp();
        }
        let log_sum = sum.ln();
        for c in 0..cols as usize {
            let v = host_x[r * outer_stride as usize + c * 2];
            expected[r * cols as usize + c] = (v - max_v) - log_sum;
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (rows as usize) * (cols as usize)).expect("alloc y");

    let shape = [rows, cols];
    let stride_x = [outer_stride, 2i64];
    let stride_y = [cols as i64, 1i64];
    let numel = rows as i64 * cols as i64;
    let softmax_axis = (shape.len() - 1) as i32;

    let status = unsafe {
        sys::baracuda_kernels_log_softmax_f32_strided_run(
            numel,
            shape.len() as i32,
            shape.as_ptr(),
            stride_x.as_ptr(),
            stride_y.as_ptr(),
            softmax_axis,
            cols,
            2i64,
            1i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "log_softmax_f32_strided_run returned {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (rows as usize) * (cols as usize)];
    dev_y.copy_to_host(&mut got).expect("download");

    let tol = 1e-5f32;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= tol.max(e.abs() * tol),
            "log_softmax_f32_strided @ {i}: got {g} expected {e}"
        );
    }
}

/// Flip strided sibling, f32. Input is the transpose of a `[H, W]` row-major
/// matrix viewed with logical shape `[W, H]` and strides `[1, W]`. Flip
/// axis 0 (the logical W-axis = the physical contig axis of the underlying
/// matrix, accessed via the transposed view's stride 1).
#[test]
#[ignore]
fn flip_f32_strided_smoke() {
    let (ctx, stream) = setup();
    let h = 4i32;
    let w = 8i32;
    let host_x: Vec<f32> = (0..(h * w) as usize).map(|i| i as f32).collect();

    // Logical shape [W, H] with strides [1, W] = transposed view.
    let logical_shape = [w, h];
    let stride_x = [1i64, w as i64];
    let stride_y = [h as i64, 1i64];  // contig output [W, H]
    let flip_axes = [1i32, 0i32];     // flip axis 0 (logical W-dim)
    let numel = (w * h) as i64;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (w * h) as usize).expect("alloc");

    let status = unsafe {
        sys::baracuda_kernels_flip_f32_strided_run(
            numel,
            logical_shape.len() as i32,
            logical_shape.as_ptr(),
            flip_axes.as_ptr(),
            stride_x.as_ptr(),
            stride_y.as_ptr(),
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "flip_f32_strided_run returned {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (w * h) as usize];
    dev_y.copy_to_host(&mut got).expect("download");

    // Reference: logical_shape[w, h] with stride[1, w]; flip axis 0 (logical
    // W-dim). For logical coord (c0, c1): src uses flipped c0 = (w-1-c0).
    let mut expected = vec![0f32; (w * h) as usize];
    for c0 in 0..w as usize {
        for c1 in 0..h as usize {
            let flipped_w = (w as usize - 1) - c0;
            let src_off = flipped_w * 1 + c1 * w as usize;
            let dst_off = c0 * h as usize + c1;
            expected[dst_off] = host_x[src_off];
        }
    }

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "flip_f32_strided @ {i}: got {g} expected {e}");
    }
}

/// Roll strided sibling, f32. Stride-2 input over a 2x-padded buffer; roll
/// the last axis by 3 positions.
#[test]
#[ignore]
fn roll_f32_strided_smoke() {
    let (ctx, stream) = setup();
    let rows = 3i32;
    let cols = 8i32;
    let outer_stride = 2i64 * cols as i64;
    let buf_len = (rows as i64 * outer_stride) as usize;
    let shift = 3i32;

    let host_x: Vec<f32> = (0..buf_len).map(|i| i as f32).collect();
    let shape = [rows, cols];
    let shifts = [0i32, shift];
    let stride_x = [outer_stride, 2i64];
    let stride_y = [cols as i64, 1i64];
    let numel = (rows * cols) as i64;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (rows * cols) as usize).expect("alloc");

    let status = unsafe {
        sys::baracuda_kernels_roll_f32_strided_run(
            numel,
            shape.len() as i32,
            shape.as_ptr(),
            shifts.as_ptr(),
            stride_x.as_ptr(),
            stride_y.as_ptr(),
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "roll_f32_strided_run returned {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (rows * cols) as usize];
    dev_y.copy_to_host(&mut got).expect("download");

    // Reference: roll axis 1 by `shift` for each row in the stride-2 view.
    let mut expected = vec![0f32; (rows * cols) as usize];
    for r in 0..rows as usize {
        for c in 0..cols as usize {
            let src_c = ((c as i32 - shift).rem_euclid(cols)) as usize;
            expected[r * cols as usize + c] = host_x[r * outer_stride as usize + src_c * 2];
        }
    }

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "roll_f32_strided @ {i}: got {g} expected {e}");
    }
}

/// Permute strided sibling, f32. Input is `[H, W]` row-major; permute
/// `dims = [1, 0]` produces the transpose. Reads from contig input
/// (stride `[W, 1]`) and writes to contig output (stride `[H, 1]`).
#[test]
#[ignore]
fn permute_f32_strided_smoke() {
    let (ctx, stream) = setup();
    let h = 4i32;
    let w = 8i32;
    let host_x: Vec<f32> = (0..(h * w) as usize).map(|i| i as f32).collect();

    let shape = [h, w];
    let dims = [1i32, 0i32];
    let stride_x = [w as i64, 1i64];
    let stride_y = [h as i64, 1i64];
    let input_numel = (h * w) as i64;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (h * w) as usize).expect("alloc");

    let status = unsafe {
        sys::baracuda_kernels_permute_f32_strided_run(
            input_numel,
            shape.len() as i32,
            shape.as_ptr(),
            dims.as_ptr(),
            stride_x.as_ptr(),
            stride_y.as_ptr(),
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "permute_f32_strided_run returned {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (h * w) as usize];
    dev_y.copy_to_host(&mut got).expect("download");

    // Reference: `y[i,j] = x[j,i]` (transpose).
    let mut expected = vec![0f32; (h * w) as usize];
    for i in 0..w as usize {
        for j in 0..h as usize {
            expected[i * h as usize + j] = host_x[j * w as usize + i];
        }
    }

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "permute_f32_strided @ {i}: got {g} expected {e}");
    }
}
