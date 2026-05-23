//! Real-GPU smoke tests for the Phase 14.4 strided RoPE siblings.
//!
//! Coverage:
//!   * Contiguous case → fast path (sanity that the dispatch didn't
//!     break the existing canonical-contig fast-path).
//!   * Transposed `[batch, seq, heads, head_dim]` view (heads/seq
//!     swapped) → strided FFI.
//!   * Reject non-unit head_dim stride at the plan layer.
//!   * f32 + f16 dtype quick coverage.
//!   * One backward case to confirm BW strided also works.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test rope_strided_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, RopeArgs, RopeBackwardArgs,
    RopeBackwardDescriptor, RopeBackwardPlan, RopeDescriptor, RopePlan, TensorMut, TensorRef,
    Workspace, ROPE_DEFAULT_BASE,
};
use half::f16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference for the FW. Reads `x` at the given outer strides and
/// writes `y` at the canonical contig stride (caller passes `y` shape
/// `[B, H, S, D]`). Useful for cross-checking the strided GPU output
/// against the contig output reference.
fn host_rope_ref_f32(
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
    base: f32,
    x_phys: &[f32],
    stride_x: [i64; 4],
    positions: Option<&[i64]>,
) -> Vec<f32> {
    assert_eq!(head_dim % 2, 0, "head_dim must be even");
    assert_eq!(stride_x[3], 1, "head_dim stride must be 1");
    let total = batch * heads * seq * head_dim;
    let mut y = vec![0f32; total];
    let inv_d = 1.0 / (head_dim as f32);
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq {
                let pos = positions.map(|p| p[s]).unwrap_or(s as i64) as f32;
                let x_outer = (b as i64) * stride_x[0]
                    + (h as i64) * stride_x[1]
                    + (s as i64) * stride_x[2];
                for pair in 0..(head_dim / 2) {
                    let d_even = pair * 2;
                    let d_odd = d_even + 1;
                    let exponent = -((d_even as f32) * inv_d);
                    let freq = base.powf(exponent);
                    let theta = pos * freq;
                    let c = theta.cos();
                    let si = theta.sin();
                    let x_e = x_phys[(x_outer + d_even as i64) as usize];
                    let x_o = x_phys[(x_outer + d_odd as i64) as usize];
                    let y_off = ((b * heads + h) * seq + s) * head_dim;
                    y[y_off + d_even] = x_e * c - x_o * si;
                    y[y_off + d_odd] = x_o * c + x_e * si;
                }
            }
        }
    }
    y
}

/// 1. Contiguous path still works (sanity check).
#[test]
#[ignore]
fn rope_strided_contig_f32() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 4i32;
    let seq = 8i32;
    let head_dim = 16i32;
    let numel = (batch * heads * seq * head_dim) as usize;
    let base = ROPE_DEFAULT_BASE;
    let host_x: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.13 - 0.7).sin() * 1.2)
        .collect();
    let shape = [batch, heads, seq, head_dim];
    let stride = contiguous_stride(shape);
    let expected = host_rope_ref_f32(
        batch as usize,
        heads as usize,
        seq as usize,
        head_dim as usize,
        base,
        &host_x,
        stride,
        None,
    );

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let desc = RopeDescriptor {
        batch_size: batch,
        num_heads: heads,
        seq_len: seq,
        head_dim,
        base,
        element: ElementKind::F32,
    };
    let plan = RopePlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        RopeArgs {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape,
                stride,
            },
            positions: None,
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape,
                stride,
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("dl");
    let tol = 16.0 * f32::EPSILON;
    for i in 0..numel {
        let t = (expected[i].abs() * tol).max(tol);
        assert!(
            (got[i] - expected[i]).abs() <= t,
            "contig f32 @ {i}: got={} want={}",
            got[i],
            expected[i]
        );
    }
}

/// 2. Transposed view: physical buffer is `[B, S, H, D]` (seq and heads
/// swapped). We pretend it's `[B, H, S, D]` by reordering the strides.
/// This exercises the strided FFI: outer-dim strides are non-canonical.
#[test]
#[ignore]
fn rope_strided_transposed_view_f32() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 3i32;
    let seq = 6i32;
    let head_dim = 8i32;
    let numel = (batch * heads * seq * head_dim) as usize;
    let base = ROPE_DEFAULT_BASE;
    // Physical buffer layout [B, S, H, D] (canonical contig for that
    // shape). When viewing as [B, H, S, D], the strides become:
    //   stride_b = S*H*D
    //   stride_h = D            (was the innermost outer dim)
    //   stride_s = H*D          (was second from innermost outer dim)
    //   stride_d = 1
    let s_h_d = (heads * head_dim) as i64;
    let h_d   = head_dim as i64;
    let s_phys_b: i64 = (seq as i64) * s_h_d;
    let stride_x: [i64; 4] = [s_phys_b, h_d, s_h_d, 1];
    let host_x: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.11 - 0.4).cos() * 0.8)
        .collect();
    let expected = host_rope_ref_f32(
        batch as usize,
        heads as usize,
        seq as usize,
        head_dim as usize,
        base,
        &host_x,
        stride_x,
        None,
    );

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let shape = [batch, heads, seq, head_dim];
    let desc = RopeDescriptor {
        batch_size: batch,
        num_heads: heads,
        seq_len: seq,
        head_dim,
        base,
        element: ElementKind::F32,
    };
    let plan = RopePlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        RopeArgs {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape,
                stride: stride_x,
            },
            positions: None,
            y: TensorMut {
                // y is fresh contig in canonical [B, H, S, D] order
                data: dev_y.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("dl");
    let tol = 16.0 * f32::EPSILON;
    for i in 0..numel {
        let t = (expected[i].abs() * tol).max(tol);
        assert!(
            (got[i] - expected[i]).abs() <= t,
            "transposed f32 @ {i}: got={} want={}",
            got[i],
            expected[i]
        );
    }
}

/// 3. Reject non-unit head_dim stride at the plan layer.
#[test]
#[ignore]
fn rope_strided_rejects_non_unit_headdim_stride() {
    let (ctx, stream) = setup();
    let batch = 1i32;
    let heads = 1i32;
    let seq = 4i32;
    let head_dim = 4i32;
    let numel = (batch * heads * seq * head_dim * 2) as usize; // doubled for stride 2
    let host_x: Vec<f32> = (0..numel).map(|i| i as f32).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (batch * heads * seq * head_dim) as usize).expect("alloc");
    let desc = RopeDescriptor {
        batch_size: batch,
        num_heads: heads,
        seq_len: seq,
        head_dim,
        base: ROPE_DEFAULT_BASE,
        element: ElementKind::F32,
    };
    let plan = RopePlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let shape = [batch, heads, seq, head_dim];
    // Construct a view with stride[3] = 2 (non-unit)
    let bad_stride: [i64; 4] = [
        (heads as i64) * (seq as i64) * (head_dim as i64) * 2,
        (seq as i64) * (head_dim as i64) * 2,
        (head_dim as i64) * 2,
        2, // BAD: head_dim stride must be 1
    ];
    let res = plan.run(
        &stream,
        Workspace::None,
        RopeArgs {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape,
                stride: bad_stride,
            },
            positions: None,
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    );
    assert!(
        res.is_err(),
        "expected rejection of non-unit head_dim stride, got Ok"
    );
}

/// 4. f16 strided coverage.
#[test]
#[ignore]
fn rope_strided_transposed_f16() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 2i32;
    let seq = 4i32;
    let head_dim = 8i32;
    let numel = (batch * heads * seq * head_dim) as usize;
    let base = ROPE_DEFAULT_BASE;
    // Physical [B, S, H, D] view as [B, H, S, D]
    let s_h_d = (heads * head_dim) as i64;
    let h_d   = head_dim as i64;
    let s_phys_b: i64 = (seq as i64) * s_h_d;
    let stride_x: [i64; 4] = [s_phys_b, h_d, s_h_d, 1];
    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.07 - 0.3).sin() * 0.6)
        .collect();
    let expected = host_rope_ref_f32(
        batch as usize,
        heads as usize,
        seq as usize,
        head_dim as usize,
        base,
        &host_x_f32,
        stride_x,
        None,
    );

    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let shape = [batch, heads, seq, head_dim];
    let desc = RopeDescriptor {
        batch_size: batch,
        num_heads: heads,
        seq_len: seq,
        head_dim,
        base,
        element: ElementKind::F16,
    };
    let plan = RopePlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        RopeArgs {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape,
                stride: stride_x,
            },
            positions: None,
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("dl");
    // 16 * f16 eps ≈ 0.016
    let tol = 0.02f32;
    for i in 0..numel {
        let t = (expected[i].abs() * tol).max(tol);
        let diff = (got[i].to_f32() - expected[i]).abs();
        assert!(
            diff <= t,
            "transposed f16 @ {i}: diff={diff} want={}",
            expected[i]
        );
    }
}

/// 5. BW strided case — verifies the BW plan dispatches.
#[test]
#[ignore]
fn rope_strided_backward_transposed_f32() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 2i32;
    let seq = 4i32;
    let head_dim = 8i32;
    let numel = (batch * heads * seq * head_dim) as usize;
    let base = ROPE_DEFAULT_BASE;
    // Physical [B, S, H, D] view as [B, H, S, D]
    let s_h_d = (heads * head_dim) as i64;
    let h_d   = head_dim as i64;
    let s_phys_b: i64 = (seq as i64) * s_h_d;
    let stride_dy: [i64; 4] = [s_phys_b, h_d, s_h_d, 1];
    // CPU reference: same shape FW with reversed-sign sin (BW)
    let host_dy: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.09 - 0.5).cos() * 0.7)
        .collect();
    let inv_d = 1.0 / (head_dim as f32);
    let mut expected = vec![0f32; numel];
    for b in 0..batch as usize {
        for h in 0..heads as usize {
            for s in 0..seq as usize {
                let pos = s as f32;
                let dy_outer = (b as i64) * stride_dy[0]
                    + (h as i64) * stride_dy[1]
                    + (s as i64) * stride_dy[2];
                for pair in 0..(head_dim as usize / 2) {
                    let d_even = pair * 2;
                    let d_odd = d_even + 1;
                    let exponent = -((d_even as f32) * inv_d);
                    let freq = base.powf(exponent);
                    let theta = pos * freq;
                    let c = theta.cos();
                    let si = theta.sin();
                    let dy_e = host_dy[(dy_outer + d_even as i64) as usize];
                    let dy_o = host_dy[(dy_outer + d_odd as i64) as usize];
                    let dx_off = ((b * heads as usize + h) * seq as usize + s) * head_dim as usize;
                    // dx[2i]   = dy[2i] · cos + dy[2i+1] · sin
                    expected[dx_off + d_even] = dy_e * c + dy_o * si;
                    // dx[2i+1] = dy[2i+1] · cos - dy[2i] · sin
                    expected[dx_off + d_odd] = dy_o * c - dy_e * si;
                }
            }
        }
    }

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let shape = [batch, heads, seq, head_dim];
    let desc = RopeBackwardDescriptor {
        batch_size: batch,
        num_heads: heads,
        seq_len: seq,
        head_dim,
        base,
        element: ElementKind::F32,
    };
    let plan =
        RopeBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        RopeBackwardArgs {
            dy: TensorRef {
                data: dev_dy.as_slice(),
                shape,
                stride: stride_dy,
            },
            positions: None,
            dx: TensorMut {
                data: dev_dx.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let tol = 16.0 * f32::EPSILON;
    for i in 0..numel {
        let t = (expected[i].abs() * tol).max(tol);
        assert!(
            (got[i] - expected[i]).abs() <= t,
            "bw transposed f32 @ {i}: got={} want={}",
            got[i],
            expected[i]
        );
    }
}
