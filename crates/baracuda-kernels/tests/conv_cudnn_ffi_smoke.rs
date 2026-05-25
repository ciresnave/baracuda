#![cfg(feature = "cudnn")]
//! Phase 19.2 — Real-GPU smoke tests for the cuDNN convolution FFI
//! facade exposed by `baracuda-kernels-sys`.
//!
//! Each test invokes the C-ABI `baracuda_kernels_conv_*_run` /
//! `baracuda_kernels_conv_transpose_*_run` symbols directly and
//! compares the device-side output against the equivalent Rust plan
//! (`Conv2dPlan`, `Conv1dPlan`, `ConvTranspose2dPlan`) run with the
//! same inputs. Bit-exact match is expected (the FFI wrapper and the
//! Rust plan both pin the same cuDNN algorithm).
//!
//! Coverage in this smoke set:
//!   - Conv2d f32 FW + BW data + BW filter.
//!   - Conv2d f32 FW depthwise (groups = in_channels).
//!   - Conv1d f16 FW (verifies the FFI-side pad-to-rank-4 logic).
//!   - ConvTranspose2d f32 FW.
//!
//! All tests `#[ignore]` — need a real CUDA device + cuDNN at runtime.

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Conv1dArgs, Conv1dDescriptor, Conv1dPlan, Conv2dArgs, Conv2dBwArgs,
    Conv2dDescriptor, Conv2dDwArgs, Conv2dPlan, ConvTranspose2dArgs, ConvTranspose2dDescriptor,
    ConvTranspose2dPlan, ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::f16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// ============================================================================
// Conv2d FFI tests
// ============================================================================

#[derive(Copy, Clone, Debug)]
struct Conv2dDims {
    n: i32,
    c_in: i32,
    h_in: i32,
    w_in: i32,
    c_out: i32,
    h_filt: i32,
    w_filt: i32,
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    dilation_h: i32,
    dilation_w: i32,
    groups: i32,
}

impl Conv2dDims {
    fn output_dims(&self) -> (i32, i32) {
        let h_eff = self.dilation_h * (self.h_filt - 1) + 1;
        let w_eff = self.dilation_w * (self.w_filt - 1) + 1;
        (
            (self.h_in + 2 * self.pad_h - h_eff) / self.stride_h + 1,
            (self.w_in + 2 * self.pad_w - w_eff) / self.stride_w + 1,
        )
    }
    fn to_descriptor(&self, elem: ElementKind) -> Conv2dDescriptor {
        Conv2dDescriptor {
            batch: self.n,
            c_in: self.c_in,
            h_in: self.h_in,
            w_in: self.w_in,
            c_out: self.c_out,
            h_filt: self.h_filt,
            w_filt: self.w_filt,
            pad_h: self.pad_h,
            pad_w: self.pad_w,
            stride_h: self.stride_h,
            stride_w: self.stride_w,
            dilation_h: self.dilation_h,
            dilation_w: self.dilation_w,
            groups: self.groups,
            element: elem,
        }
    }
}

const TRAILBLAZER_2D: Conv2dDims = Conv2dDims {
    n: 1,
    c_in: 3,
    h_in: 5,
    w_in: 5,
    c_out: 4,
    h_filt: 3,
    w_filt: 3,
    pad_h: 1,
    pad_w: 1,
    stride_h: 1,
    stride_w: 1,
    dilation_h: 1,
    dilation_w: 1,
    groups: 1,
};

fn deterministic_f32(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| ((seed.wrapping_add(i as u32) as f32) * 0.013).sin())
        .collect()
}

/// Run the Rust plan to get the reference output for a Conv2d FW.
fn rust_plan_conv2d_fw_f32(
    ctx: &Context,
    stream: &Stream,
    d: &Conv2dDims,
    x: &[f32],
    w: &[f32],
) -> Vec<f32> {
    let dev_x = DeviceBuffer::from_slice(ctx, x).expect("up x");
    let dev_w = DeviceBuffer::from_slice(ctx, w).expect("up w");
    let (h_out, w_out) = d.output_dims();
    let y_numel = (d.n * d.c_out * h_out * w_out) as usize;
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, y_numel).expect("alloc y");
    let desc = d.to_descriptor(ElementKind::F32);
    let plan = Conv2dPlan::<f32>::select(stream, &desc, PlanPreference::default()).expect("sel");
    let ws_bytes = plan.query_fw_workspace_size(stream).expect("ws");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(ctx, ws_bytes.max(1)).expect("alloc ws");
    let x_shape = [d.n, d.c_in, d.h_in, d.w_in];
    let w_shape = [d.c_out, d.c_in / d.groups, d.h_filt, d.w_filt];
    let y_shape = [d.n, d.c_out, h_out, w_out];
    let workspace = if ws_bytes == 0 {
        Workspace::None
    } else {
        Workspace::Borrowed(dev_ws.as_slice_mut())
    };
    plan.run_fw(
        stream,
        workspace,
        Conv2dArgs::<f32> {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            w: TensorRef {
                data: dev_w.as_slice(),
                shape: w_shape,
                stride: contiguous_stride(w_shape),
            },
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
        },
    )
    .expect("run_fw");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; y_numel];
    dev_y.copy_to_host(&mut got).expect("dl y");
    got
}

/// Call the FFI directly with the same inputs.
fn ffi_conv2d_fw_f32(
    ctx: &Context,
    stream: &Stream,
    d: &Conv2dDims,
    x: &[f32],
    w: &[f32],
) -> Vec<f32> {
    let dev_x = DeviceBuffer::from_slice(ctx, x).expect("up x");
    let dev_w = DeviceBuffer::from_slice(ctx, w).expect("up w");
    let (h_out, w_out) = d.output_dims();
    let y_numel = (d.n * d.c_out * h_out * w_out) as usize;
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, y_numel).expect("alloc y");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_conv_2d_fw_f32_run(
            d.n, d.c_in, d.c_out,
            d.h_in, d.w_in, h_out, w_out,
            d.h_filt, d.w_filt,
            d.stride_h, d.stride_w,
            d.pad_h, d.pad_w,
            d.dilation_h, d.dilation_w,
            d.groups,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_w.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "FFI Conv2d FW f32 failed");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; y_numel];
    dev_y.copy_to_host(&mut got).expect("dl y");
    got
}

#[test]
#[ignore]
fn ffi_conv2d_f32_fw_matches_plan() {
    let (ctx, stream) = setup();
    let d = TRAILBLAZER_2D;
    let x = deterministic_f32((d.n * d.c_in * d.h_in * d.w_in) as usize, 0xA11C_E001);
    let w_numel = (d.c_out * (d.c_in / d.groups) * d.h_filt * d.w_filt) as usize;
    let w = deterministic_f32(w_numel, 0xBEEF_0001);
    let expected = rust_plan_conv2d_fw_f32(&ctx, &stream, &d, &x, &w);
    let got = ffi_conv2d_fw_f32(&ctx, &stream, &d, &x, &w);
    assert_eq!(expected.len(), got.len());
    // FFI and Rust plan pin the same cuDNN algorithm → bit-exact match.
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "FFI vs plan mismatch @ {i}: ffi={g}, plan={e}");
    }
}

#[test]
#[ignore]
fn ffi_conv2d_f32_bw_data_matches_plan() {
    let (ctx, stream) = setup();
    let d = TRAILBLAZER_2D;
    let (h_out, w_out) = d.output_dims();
    let w_numel = (d.c_out * (d.c_in / d.groups) * d.h_filt * d.w_filt) as usize;
    let w = deterministic_f32(w_numel, 0xBEEF_0002);
    let dy = deterministic_f32((d.n * d.c_out * h_out * w_out) as usize, 0xCAFE_BABE);
    let dx_numel = (d.n * d.c_in * d.h_in * d.w_in) as usize;

    // --- Rust plan reference ---
    let dev_w = DeviceBuffer::from_slice(&ctx, &w).expect("up w");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc dx");
    let desc = d.to_descriptor(ElementKind::F32);
    let plan = Conv2dPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let ws_bytes = plan.query_bw_data_workspace_size(&stream).expect("ws");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bytes.max(1)).expect("alloc ws");
    let w_shape = [d.c_out, d.c_in / d.groups, d.h_filt, d.w_filt];
    let dy_shape = [d.n, d.c_out, h_out, w_out];
    let dx_shape = [d.n, d.c_in, d.h_in, d.w_in];
    let workspace = if ws_bytes == 0 {
        Workspace::None
    } else {
        Workspace::Borrowed(dev_ws.as_slice_mut())
    };
    plan.run_bw_data(
        &stream,
        workspace,
        Conv2dBwArgs::<f32> {
            w: TensorRef {
                data: dev_w.as_slice(),
                shape: w_shape,
                stride: contiguous_stride(w_shape),
            },
            dy: TensorRef {
                data: dev_dy.as_slice(),
                shape: dy_shape,
                stride: contiguous_stride(dy_shape),
            },
            dx: TensorMut {
                data: dev_dx.as_slice_mut(),
                shape: dx_shape,
                stride: contiguous_stride(dx_shape),
            },
        },
    )
    .expect("run_bw_data");
    stream.synchronize().expect("sync");
    let mut expected = vec![0f32; dx_numel];
    dev_dx.copy_to_host(&mut expected).expect("dl dx");

    // --- FFI under test ---
    let dev_w2 = DeviceBuffer::from_slice(&ctx, &w).expect("up w");
    let dev_dy2 = DeviceBuffer::from_slice(&ctx, &dy).expect("up dy");
    let mut dev_dx2: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc dx");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_conv_2d_bw_data_f32_run(
            d.n, d.c_in, d.c_out,
            d.h_in, d.w_in, h_out, w_out,
            d.h_filt, d.w_filt,
            d.stride_h, d.stride_w,
            d.pad_h, d.pad_w,
            d.dilation_h, d.dilation_w,
            d.groups,
            dev_w2.as_slice().as_raw().0 as *const c_void,
            dev_dy2.as_slice().as_raw().0 as *const c_void,
            dev_dx2.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "FFI Conv2d BW data f32 failed");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; dx_numel];
    dev_dx2.copy_to_host(&mut got).expect("dl dx");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "FFI vs plan BW data mismatch @ {i}: ffi={g}, plan={e}");
    }
}

#[test]
#[ignore]
fn ffi_conv2d_f32_bw_filter_matches_plan() {
    let (ctx, stream) = setup();
    let d = TRAILBLAZER_2D;
    let (h_out, w_out) = d.output_dims();
    let dw_numel = (d.c_out * (d.c_in / d.groups) * d.h_filt * d.w_filt) as usize;
    let x = deterministic_f32((d.n * d.c_in * d.h_in * d.w_in) as usize, 0xABCD_1234);
    let dy = deterministic_f32((d.n * d.c_out * h_out * w_out) as usize, 0x4321_DCBA);

    let dev_x = DeviceBuffer::from_slice(&ctx, &x).expect("up x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy).expect("up dy");
    let mut dev_dw: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, dw_numel).expect("alloc dw");
    let desc = d.to_descriptor(ElementKind::F32);
    let plan = Conv2dPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let ws_bytes = plan.query_bw_filter_workspace_size(&stream).expect("ws");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bytes.max(1)).expect("alloc ws");
    let x_shape = [d.n, d.c_in, d.h_in, d.w_in];
    let dy_shape = [d.n, d.c_out, h_out, w_out];
    let dw_shape = [d.c_out, d.c_in / d.groups, d.h_filt, d.w_filt];
    let workspace = if ws_bytes == 0 {
        Workspace::None
    } else {
        Workspace::Borrowed(dev_ws.as_slice_mut())
    };
    plan.run_dw(
        &stream,
        workspace,
        Conv2dDwArgs::<f32> {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            dy: TensorRef {
                data: dev_dy.as_slice(),
                shape: dy_shape,
                stride: contiguous_stride(dy_shape),
            },
            dw: TensorMut {
                data: dev_dw.as_slice_mut(),
                shape: dw_shape,
                stride: contiguous_stride(dw_shape),
            },
        },
    )
    .expect("run_dw");
    stream.synchronize().expect("sync");
    let mut expected = vec![0f32; dw_numel];
    dev_dw.copy_to_host(&mut expected).expect("dl dw");

    let dev_x2 = DeviceBuffer::from_slice(&ctx, &x).expect("up x");
    let dev_dy2 = DeviceBuffer::from_slice(&ctx, &dy).expect("up dy");
    let mut dev_dw2: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, dw_numel).expect("alloc dw");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_conv_2d_bw_filter_f32_run(
            d.n, d.c_in, d.c_out,
            d.h_in, d.w_in, h_out, w_out,
            d.h_filt, d.w_filt,
            d.stride_h, d.stride_w,
            d.pad_h, d.pad_w,
            d.dilation_h, d.dilation_w,
            d.groups,
            dev_x2.as_slice().as_raw().0 as *const c_void,
            dev_dy2.as_slice().as_raw().0 as *const c_void,
            dev_dw2.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "FFI Conv2d BW filter f32 failed");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; dw_numel];
    dev_dw2.copy_to_host(&mut got).expect("dl dw");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "FFI vs plan BW filter mismatch @ {i}: ffi={g}, plan={e}");
    }
}

#[test]
#[ignore]
fn ffi_conv2d_f32_depthwise_fw_matches_plan() {
    let (ctx, stream) = setup();
    let d = Conv2dDims {
        c_in: 4,
        c_out: 4, // groups = c_in = c_out → depthwise.
        groups: 4,
        ..TRAILBLAZER_2D
    };
    let x = deterministic_f32((d.n * d.c_in * d.h_in * d.w_in) as usize, 0xDEEF_DEEF);
    // Depthwise filter shape is [c_out, c_in/groups=1, h_filt, w_filt].
    let w_numel = (d.c_out * (d.c_in / d.groups) * d.h_filt * d.w_filt) as usize;
    let w = deterministic_f32(w_numel, 0x9ABC_DEF0);
    let expected = rust_plan_conv2d_fw_f32(&ctx, &stream, &d, &x, &w);
    let got = ffi_conv2d_fw_f32(&ctx, &stream, &d, &x, &w);
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "FFI vs plan depthwise FW mismatch @ {i}: ffi={g}, plan={e}");
    }
}

// ============================================================================
// Conv1d FFI tests — verify the pad-to-rank-4 logic through the FFI.
// ============================================================================

#[test]
#[ignore]
fn ffi_conv1d_f16_fw_matches_plan() {
    let (ctx, stream) = setup();
    let n = 2;
    let c_in = 3;
    let l_in = 7;
    let c_out = 4;
    let l_filt = 3;
    let pad_l = 1;
    let stride_l = 1;
    let dilation_l = 1;
    let groups = 1;
    let l_eff = dilation_l * (l_filt - 1) + 1;
    let l_out = (l_in + 2 * pad_l - l_eff) / stride_l + 1;

    let x_numel = (n * c_in * l_in) as usize;
    let w_numel = (c_out * (c_in / groups) * l_filt) as usize;
    let y_numel = (n * c_out * l_out) as usize;

    let host_x: Vec<f16> = deterministic_f32(x_numel, 0xF16F_16F1)
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let host_w: Vec<f16> = deterministic_f32(w_numel, 0xF161_F161)
        .into_iter()
        .map(f16::from_f32)
        .collect();

    // --- Rust plan reference ---
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up w");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, y_numel).expect("alloc y");
    let desc = Conv1dDescriptor {
        batch: n,
        c_in,
        l_in,
        c_out,
        l_filt,
        pad_l,
        stride_l,
        dilation_l,
        groups,
        element: ElementKind::F16,
    };
    let plan =
        Conv1dPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let ws_bytes = plan.query_fw_workspace_size(&stream).expect("ws");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bytes.max(1)).expect("alloc ws");
    let x_shape = [n, c_in, l_in];
    let w_shape = [c_out, c_in / groups, l_filt];
    let y_shape = [n, c_out, l_out];
    let workspace = if ws_bytes == 0 {
        Workspace::None
    } else {
        Workspace::Borrowed(dev_ws.as_slice_mut())
    };
    plan.run_fw(
        &stream,
        workspace,
        Conv1dArgs::<f16> {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            w: TensorRef {
                data: dev_w.as_slice(),
                shape: w_shape,
                stride: contiguous_stride(w_shape),
            },
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
        },
    )
    .expect("run_fw");
    stream.synchronize().expect("sync");
    let mut expected = vec![f16::from_f32(0.0); y_numel];
    dev_y.copy_to_host(&mut expected).expect("dl y");

    // --- FFI under test (rank-3 dims at the boundary; padding is internal) ---
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_w2 = DeviceBuffer::from_slice(&ctx, &host_w).expect("up w");
    let mut dev_y2: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, y_numel).expect("alloc y");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_conv_1d_fw_f16_run(
            n, c_in, c_out,
            l_in, l_out,
            l_filt,
            stride_l, pad_l, dilation_l,
            groups,
            dev_x2.as_slice().as_raw().0 as *const c_void,
            dev_w2.as_slice().as_raw().0 as *const c_void,
            dev_y2.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "FFI Conv1d FW f16 failed");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::from_f32(0.0); y_numel];
    dev_y2.copy_to_host(&mut got).expect("dl y");

    // Bit-exact comparison via the u16 storage bits.
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "FFI vs plan Conv1d f16 FW mismatch @ {i}: ffi={}, plan={}",
            g.to_f32(),
            e.to_f32(),
        );
    }
}

// ============================================================================
// ConvTranspose2d FFI test
// ============================================================================

#[test]
#[ignore]
fn ffi_conv_transpose_2d_f32_fw_matches_plan() {
    let (ctx, stream) = setup();
    // 1×2×3×3 → 1×3×5×5 with kH=kW=3, stride=2, pad=1, output_pad=0.
    let n = 1;
    let c_in = 2;
    let h_in = 3;
    let w_in = 3;
    let c_out = 3;
    let h_filt = 3;
    let w_filt = 3;
    let pad_h = 1;
    let pad_w = 1;
    let stride_h = 2;
    let stride_w = 2;
    let dilation_h = 1;
    let dilation_w = 1;
    let output_pad_h = 0;
    let output_pad_w = 0;
    let groups = 1;
    let h_out = (h_in - 1) * stride_h - 2 * pad_h + dilation_h * (h_filt - 1) + output_pad_h + 1;
    let w_out = (w_in - 1) * stride_w - 2 * pad_w + dilation_w * (w_filt - 1) + output_pad_w + 1;

    let x_numel = (n * c_in * h_in * w_in) as usize;
    let w_numel = (c_in * (c_out / groups) * h_filt * w_filt) as usize;
    let y_numel = (n * c_out * h_out * w_out) as usize;
    let host_x = deterministic_f32(x_numel, 0xC0FFEE_01);
    let host_w = deterministic_f32(w_numel, 0xDEAD_BEEF);

    // --- Rust plan reference ---
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up w");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, y_numel).expect("alloc y");
    let desc = ConvTranspose2dDescriptor {
        batch: n,
        c_in,
        h_in,
        w_in,
        c_out,
        h_filt,
        w_filt,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        output_pad_h,
        output_pad_w,
        groups,
        element: ElementKind::F32,
    };
    let plan = ConvTranspose2dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let ws_bytes = plan.query_fw_workspace_size(&stream).expect("ws");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bytes.max(1)).expect("alloc ws");
    let x_shape = [n, c_in, h_in, w_in];
    let w_shape = [c_in, c_out / groups, h_filt, w_filt];
    let y_shape = [n, c_out, h_out, w_out];
    let workspace = if ws_bytes == 0 {
        Workspace::None
    } else {
        Workspace::Borrowed(dev_ws.as_slice_mut())
    };
    plan.run_fw(
        &stream,
        workspace,
        ConvTranspose2dArgs::<f32> {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            w: TensorRef {
                data: dev_w.as_slice(),
                shape: w_shape,
                stride: contiguous_stride(w_shape),
            },
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
        },
    )
    .expect("run_fw");
    stream.synchronize().expect("sync");
    let mut expected = vec![0f32; y_numel];
    dev_y.copy_to_host(&mut expected).expect("dl y");

    // --- FFI under test ---
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_w2 = DeviceBuffer::from_slice(&ctx, &host_w).expect("up w");
    let mut dev_y2: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, y_numel).expect("alloc y");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_conv_transpose_2d_fw_f32_run(
            n, c_in, c_out,
            h_in, w_in, h_out, w_out,
            h_filt, w_filt,
            stride_h, stride_w,
            pad_h, pad_w,
            dilation_h, dilation_w,
            output_pad_h, output_pad_w,
            groups,
            dev_x2.as_slice().as_raw().0 as *const c_void,
            dev_w2.as_slice().as_raw().0 as *const c_void,
            dev_y2.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "FFI ConvTranspose2d FW f32 failed");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; y_numel];
    dev_y2.copy_to_host(&mut got).expect("dl y");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "FFI vs plan ConvTranspose2d FW mismatch @ {i}: ffi={g}, plan={e}");
    }
}
