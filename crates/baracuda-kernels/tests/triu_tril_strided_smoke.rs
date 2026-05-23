//! Real-GPU smoke tests for the Triu / Tril strided siblings — Phase 14.3.
//!
//! Coverage:
//!   * Contiguous case → verifies fast path still works (sanity check
//!     that adding the strided branch didn't break the existing
//!     canonical-contig path).
//!   * Transposed input view (strided source, contig dest) → verifies
//!     strided FFI fires and the (i, j) coord decode handles the
//!     non-canonical stride.
//!   * Strided output (interior view of a larger buffer) → verifies
//!     only the strided region is written.
//!   * Per-dtype quick coverage: f32, f16, bf16, i32.
//!   * One backward case to confirm BW strided also works.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test triu_tril_strided_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, TrilArgs,
    TrilDescriptor, TrilPlan, TriuArgs, TriuBackwardArgs, TriuBackwardDescriptor,
    TriuBackwardPlan, TriuDescriptor, TriuPlan, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference: produces the mask-of-source as a contiguous output
/// tensor, given an input read at the given source strides. The
/// caller produces an N×M physical buffer; we apply the read pattern
/// described by `input_stride` and the triu predicate.
fn triu_ref_strided_f32(
    input_buf: &[f32],
    shape: [i32; 2],
    input_stride: [i64; 2],
    diagonal: i32,
) -> Vec<f32> {
    let (m, n) = (shape[0], shape[1]);
    let mut out = vec![0f32; (m * n) as usize];
    for i in 0..m {
        for j in 0..n {
            let src = (i as i64) * input_stride[0] + (j as i64) * input_stride[1];
            let dst = (i * n + j) as usize;
            out[dst] = if j >= i + diagonal {
                input_buf[src as usize]
            } else {
                0.0
            };
        }
    }
    out
}

// ============================================================================
// 1. Contiguous case — sanity check the fast path still works.
// ============================================================================

#[test]
#[ignore]
fn triu_strided_dispatch_contig_f32() {
    let (ctx, stream) = setup();
    let shape = [3i32, 3];
    let host_x: Vec<f32> = (1..=9).map(|i| i as f32).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 9).expect("alloc y");

    let desc = TriuDescriptor {
        shape,
        diagonal: 0,
        element: ElementKind::F32,
    };
    let plan =
        TriuPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = TriuArgs::<f32, 2> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    // Expected: standard triu(3x3, diag=0) over [1..9].
    let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0];
    let mut got = vec![0f32; 9];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "contig mismatch @ {i}");
    }
}

// ============================================================================
// 2. Transposed input view — strided source, contig dest. The view
//    swaps rows/cols of the underlying buffer; the kernel should
//    decode (i, j) coords against the LOGICAL shape, not the physical
//    layout.
// ============================================================================

#[test]
#[ignore]
fn triu_strided_transposed_input_f32() {
    let (ctx, stream) = setup();
    // Physical layout: 3 rows × 4 cols, row-major.
    let phys_cols = 4i32;
    let host_phys: Vec<f32> = (1..=(3 * phys_cols)).map(|i| i as f32).collect();

    // Logical view: 4 × 3 (transposed). stride[0] = 1 (was inner),
    // stride[1] = 4 (was outer).
    let logical_shape = [4i32, 3];
    let logical_stride = [1i64, phys_cols as i64];

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_phys).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 12).expect("alloc y");

    let desc = TriuDescriptor {
        shape: logical_shape,
        diagonal: 0,
        element: ElementKind::F32,
    };
    let plan =
        TriuPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = TriuArgs::<f32, 2> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: logical_shape,
            stride: logical_stride,
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: logical_shape,
            stride: contiguous_stride(logical_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let expected = triu_ref_strided_f32(&host_phys, logical_shape, logical_stride, 0);
    let mut got = vec![0f32; 12];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "transposed-input mismatch @ {i}: got={g}, expected={e}"
        );
    }
}

// ============================================================================
// 3. Strided output (interior view of a larger buffer) — the kernel
//    should write only into the strided region; surrounding bytes
//    stay at their initial value.
// ============================================================================

#[test]
#[ignore]
fn tril_strided_interior_output_f32() {
    let (ctx, stream) = setup();
    // Logical shape: 3x3. Output is an interior view of a 4x4 buffer
    // (rows 0..3, cols 0..3) — output stride is (4, 1), so written
    // elements land at flat offsets 0, 1, 2, 4, 5, 6, 8, 9, 10.
    let logical_shape = [3i32, 3];
    let in_stride = contiguous_stride(logical_shape);
    let out_stride = [4i64, 1];

    let host_x: Vec<f32> = (1..=9).map(|i| i as f32).collect();
    let sentinel = -7.5f32;
    let host_out_init: Vec<f32> = vec![sentinel; 16];

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y = DeviceBuffer::from_slice(&ctx, &host_out_init).expect("alloc y");

    let desc = TrilDescriptor {
        shape: logical_shape,
        diagonal: 0,
        element: ElementKind::F32,
    };
    let plan =
        TrilPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = TrilArgs::<f32, 2> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: logical_shape,
            stride: in_stride,
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: logical_shape,
            stride: out_stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 16];
    dev_y.copy_to_host(&mut got).expect("download");

    // Expected: tril(3x3, diag=0) over [1..9] = [1,0,0,4,5,0,7,8,9].
    let expected_tril: [f32; 9] = [1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0];
    // Mapping (i, j) → flat offset i*4 + j (out_stride = (4, 1)).
    for i in 0..3 {
        for j in 0..3 {
            let dst = (i * 4 + j) as usize;
            let src = (i * 3 + j) as usize;
            assert_eq!(
                got[dst].to_bits(),
                expected_tril[src].to_bits(),
                "interior-output written-region mismatch @ ({i},{j})"
            );
        }
    }
    // Untouched positions (col index 3 of each row) must retain sentinel.
    for i in 0..4 {
        let pad_off = (i * 4 + 3) as usize;
        if pad_off < 16 {
            assert_eq!(
                got[pad_off].to_bits(),
                sentinel.to_bits(),
                "pad column overwritten @ row {i}"
            );
        }
    }
    // Final 4 elements (row 3) must retain sentinel too — strided
    // output only covers rows 0..3.
    for k in 12..16 {
        assert_eq!(
            got[k].to_bits(),
            sentinel.to_bits(),
            "row-3 pad overwritten @ {k}"
        );
    }
}

// ============================================================================
// 4. Per-dtype quick coverage — f16, bf16, i32. Each test uses the
//    transposed-input pattern so we know the strided path is exercised.
// ============================================================================

#[test]
#[ignore]
fn triu_strided_transposed_input_f16() {
    let (ctx, stream) = setup();
    let phys_cols = 4i32;
    let host_phys: Vec<f16> = (1..=12).map(|i| f16::from_f32(i as f32)).collect();
    let logical_shape = [4i32, 3];
    let logical_stride = [1i64, phys_cols as i64];

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_phys).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 12).expect("alloc y");

    let desc = TriuDescriptor {
        shape: logical_shape,
        diagonal: 0,
        element: ElementKind::F16,
    };
    let plan =
        TriuPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = TriuArgs::<f16, 2> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: logical_shape,
            stride: logical_stride,
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: logical_shape,
            stride: contiguous_stride(logical_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let host_phys_f32: Vec<f32> = host_phys.iter().map(|x| x.to_f32()).collect();
    let expected = triu_ref_strided_f32(&host_phys_f32, logical_shape, logical_stride, 0);
    let mut got = vec![f16::from_f32(0.0); 12];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_f32().to_bits(),
            e.to_bits(),
            "f16 strided mismatch @ {i}"
        );
    }
}

#[test]
#[ignore]
fn triu_strided_transposed_input_bf16() {
    let (ctx, stream) = setup();
    let phys_cols = 4i32;
    let host_phys: Vec<bf16> = (1..=12).map(|i| bf16::from_f32(i as f32)).collect();
    let logical_shape = [4i32, 3];
    let logical_stride = [1i64, phys_cols as i64];

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_phys).expect("upload x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 12).expect("alloc y");

    let desc = TriuDescriptor {
        shape: logical_shape,
        diagonal: 0,
        element: ElementKind::Bf16,
    };
    let plan =
        TriuPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = TriuArgs::<bf16, 2> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: logical_shape,
            stride: logical_stride,
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: logical_shape,
            stride: contiguous_stride(logical_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let host_phys_f32: Vec<f32> = host_phys.iter().map(|x| x.to_f32()).collect();
    let expected = triu_ref_strided_f32(&host_phys_f32, logical_shape, logical_stride, 0);
    let mut got = vec![bf16::from_f32(0.0); 12];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_f32().to_bits(),
            e.to_bits(),
            "bf16 strided mismatch @ {i}"
        );
    }
}

#[test]
#[ignore]
fn triu_strided_transposed_input_i32() {
    let (ctx, stream) = setup();
    let phys_cols = 4i32;
    let host_phys: Vec<i32> = (1..=12).collect();
    let logical_shape = [4i32, 3];
    let logical_stride = [1i64, phys_cols as i64];

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_phys).expect("upload x");
    let mut dev_y: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 12).expect("alloc y");

    let desc = TriuDescriptor {
        shape: logical_shape,
        diagonal: 0,
        element: ElementKind::I32,
    };
    let plan =
        TriuPlan::<i32, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = TriuArgs::<i32, 2> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: logical_shape,
            stride: logical_stride,
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: logical_shape,
            stride: contiguous_stride(logical_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    // CPU reference for i32 same shape/stride.
    let (m, n) = (logical_shape[0], logical_shape[1]);
    let mut expected = vec![0i32; (m * n) as usize];
    for i in 0..m {
        for j in 0..n {
            let src = (i as i64) * logical_stride[0] + (j as i64) * logical_stride[1];
            let dst = (i * n + j) as usize;
            expected[dst] = if j >= i + 0 { host_phys[src as usize] } else { 0 };
        }
    }

    let mut got = vec![0i32; 12];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "i32 strided mismatch @ {i}");
    }
}

// ============================================================================
// 5. Backward — same strided plumbing.
// ============================================================================

#[test]
#[ignore]
fn triu_bw_strided_transposed_input_f32() {
    let (ctx, stream) = setup();
    // Same transposed-input pattern as the FW case — BW is just the
    // FW kernel re-routed at the `grad_output → grad_input` buffers.
    let phys_cols = 4i32;
    let host_phys: Vec<f32> = (1..=12).map(|i| (i as f32) * 0.5).collect();
    let logical_shape = [4i32, 3];
    let logical_stride = [1i64, phys_cols as i64];

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_phys).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 12).expect("alloc dx");

    let desc = TriuBackwardDescriptor {
        shape: logical_shape,
        diagonal: 0,
        element: ElementKind::F32,
    };
    let plan =
        TriuBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
            .expect("select");
    let args = TriuBackwardArgs::<f32, 2> {
        grad_output: TensorRef {
            data: dev_dy.as_slice(),
            shape: logical_shape,
            stride: logical_stride,
        },
        grad_input: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: logical_shape,
            stride: contiguous_stride(logical_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let expected = triu_ref_strided_f32(&host_phys, logical_shape, logical_stride, 0);
    let mut got = vec![0f32; 12];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "BW strided mismatch @ {i}");
    }
}
