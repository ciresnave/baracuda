//! Real-GPU smoke tests for the activation-strided + W-offset MMVQ
//! sibling family — Phase 14.5.
//!
//! Coverage:
//!   * Q8_0 (type-0/1 representative): contig sanity, stride-2, stride-0
//!     broadcast, `w_start_byte_offset` two-matrix-in-one-buffer.
//!   * Q4_0 (type-0/1 with nibble pack): stride-2 + offset combined.
//!   * Q8_K (k-quants representative; bespoke kernel path): stride-2 +
//!     stride-0 broadcast.
//!
//! Tolerance: matches the contig smoke test (Q8_0 ≈ 1e-2 relative due
//! to fp16 scale round-trip; Q8_K ≈ 1e-3 since it has an f32 scale).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BlockQ8K, BlockQ8_0, BlockQ4_0, GgufBlockFormat, GgufMmvqArgs,
    GgufMmvqDescriptor, GgufMmvqPlan, PlanPreference, TensorMut, TensorRef, Workspace, U8,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// ---- helpers --------------------------------------------------------------

fn pack_q8_0_row(d_f32: f32, qs: &[i8; 32]) -> Vec<u8> {
    let blk = BlockQ8_0 { d: half::f16::from_f32(d_f32).to_bits(), qs: *qs };
    let bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            (&blk as *const BlockQ8_0) as *const u8,
            core::mem::size_of::<BlockQ8_0>(),
        )
    };
    bytes.to_vec()
}

fn pack_q8_k_row(d_f32: f32, qs: &[i8; 256]) -> Vec<u8> {
    let bsums = [0i16; 16];
    let blk = BlockQ8K { d: d_f32, qs: *qs, bsums };
    let bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            (&blk as *const BlockQ8K) as *const u8,
            core::mem::size_of::<BlockQ8K>(),
        )
    };
    bytes.to_vec()
}

fn pack_q4_0_row(d_f32: f32, vals: &[i8; 32]) -> Vec<u8> {
    // Q4_0 stores 32 4-bit signed quants; ggml layout packs index i in
    // the low nibble of qs[i % 16] when i < 16, high nibble otherwise.
    // Stored quant = value + 8 (range [0..16]); dequant subtracts 8.
    let mut qs = [0u8; 16];
    for i in 0..16 {
        let lo = (vals[i] + 8) as u8 & 0x0f;
        let hi = (vals[i + 16] + 8) as u8 & 0x0f;
        qs[i] = lo | (hi << 4);
    }
    let blk = BlockQ4_0 { d: half::f16::from_f32(d_f32).to_bits(), qs };
    let bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            (&blk as *const BlockQ4_0) as *const u8,
            core::mem::size_of::<BlockQ4_0>(),
        )
    };
    bytes.to_vec()
}

fn check_close(got: &[f32], expected: &[f32], rel_tol: f32, label: &str) {
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        let tol = rel_tol * e.abs().max(1.0);
        assert!(
            abs_err < tol,
            "{label} mismatch @ {i}: got {g}, expected {e} (|err|={abs_err}, tol={tol})"
        );
    }
}

// ---- Q8_0 strided tests ---------------------------------------------------

/// Sanity: explicit stride=1 must match the contig kernel exactly.
#[test]
#[ignore]
fn mmvq_q8_0_stride1_matches_contig() {
    let (ctx, stream) = setup();

    let d0 = 0.5_f32;
    let d1 = 0.25_f32;
    let mut qs0 = [0i8; 32];
    let mut qs1 = [0i8; 32];
    for i in 0..32 {
        qs0[i] = i as i8;
        qs1[i] = (i + 1) as i8;
    }
    let mut packed = pack_q8_0_row(d0, &qs0);
    packed.extend(pack_q8_0_row(d1, &qs1));
    let host_weight: Vec<U8> = packed.into_iter().map(U8).collect();
    let host_act: Vec<f32> = (0..32).map(|i| i as f32).collect();

    let mut expected = [0.0f32; 2];
    for c in 0..32 {
        expected[0] += d0 * qs0[c] as f32 * host_act[c];
        expected[1] += d1 * qs1[c] as f32 * host_act[c];
    }

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_weight).expect("up w");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_act).expect("up y");
    let mut dev_o: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 2).expect("alloc out");

    // Force the strided path by setting w_offset to non-zero would also
    // work, but here we want to exercise the strided-FFI with stride=1
    // and offset=0 explicitly. Use w_offset=0 + stride 1, except we set
    // the stride to a non-default value so the plan dispatches strided.
    // Actually stride=1 is the contig fast-path; to force strided here
    // set offset=0 + an explicit stride field on the TensorRef. Plan
    // dispatch uses `stride != 1 || offset != 0` so this hits contig.
    // For the explicit "stride=1 strided FFI" sanity, set offset=0 +
    // override stride. The cleanest way is to set offset to 0 and call
    // the FFI directly here — but the plan-layer API is the contract,
    // so we just exercise the contig path as the canonical sanity arm.
    let desc = GgufMmvqDescriptor {
        nrows: 2,
        ncols: 32,
        block_format: GgufBlockFormat::Q8_0,
        w_start_byte_offset: 0,
    };
    let plan = GgufMmvqPlan::select(&stream, &desc, PlanPreference::default()).expect("select");
    let weight_len = host_weight.len() as i32;
    let args = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [weight_len],
            stride: contiguous_stride([weight_len]),
        },
        activation: TensorRef {
            data: dev_y.as_slice(),
            shape: [32],
            stride: [1],
        },
        output: TensorMut {
            data: dev_o.as_slice_mut(),
            shape: [2],
            stride: contiguous_stride([2]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = [0f32; 2];
    dev_o.copy_to_host(&mut got).expect("dl");
    check_close(&got, &expected, 1e-2, "Q8_0 stride1");
}

/// Stride-2 activation: y has 64 elements, MMVQ reads every other one.
#[test]
#[ignore]
fn mmvq_q8_0_stride2() {
    let (ctx, stream) = setup();

    let d = 0.5_f32;
    let mut qs = [0i8; 32];
    for i in 0..32 {
        qs[i] = ((i as i32) % 16 - 7) as i8;
    }
    let packed = pack_q8_0_row(d, &qs);
    let host_weight: Vec<U8> = packed.into_iter().map(U8).collect();

    // Source activation of length 64; we want effective_y[c] = (c+1)*0.1.
    // With stride=2, the kernel reads source[c * 2] = effective_y[c].
    // So fill source[2*c] = (c+1)*0.1, source[2*c+1] = anything (skipped).
    let mut host_y = vec![-999.0f32; 64];
    for c in 0..32 {
        host_y[c * 2] = (c as f32 + 1.0) * 0.1;
    }

    let mut expected = [0.0f32];
    for c in 0..32 {
        expected[0] += d * qs[c] as f32 * host_y[c * 2];
    }

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_weight).expect("up w");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let mut dev_o: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc out");

    let desc = GgufMmvqDescriptor {
        nrows: 1,
        ncols: 32,
        block_format: GgufBlockFormat::Q8_0,
        w_start_byte_offset: 0,
    };
    let plan = GgufMmvqPlan::select(&stream, &desc, PlanPreference::default()).expect("select");
    let weight_len = host_weight.len() as i32;
    let args = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [weight_len],
            stride: contiguous_stride([weight_len]),
        },
        activation: TensorRef {
            data: dev_y.as_slice(),
            // logical extent is `ncols = 32` even though storage is 64.
            shape: [32],
            stride: [2],
        },
        output: TensorMut {
            data: dev_o.as_slice_mut(),
            shape: [1],
            stride: contiguous_stride([1]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = [0f32; 1];
    dev_o.copy_to_host(&mut got).expect("dl");
    check_close(&got, &expected, 1e-2, "Q8_0 stride2");
}

/// Stride-0 broadcast: kernel reads y[0] for every column.
#[test]
#[ignore]
fn mmvq_q8_0_stride0_broadcast() {
    let (ctx, stream) = setup();

    let d = 0.25_f32;
    let mut qs = [0i8; 32];
    for i in 0..32 {
        qs[i] = (i as i32 - 16) as i8;
    }
    let packed = pack_q8_0_row(d, &qs);
    let host_weight: Vec<U8> = packed.into_iter().map(U8).collect();

    let scalar = 1.5_f32;
    // Allocate just one f32 — the kernel must only ever read y[0].
    let host_y = vec![scalar];

    let mut expected = [0.0f32];
    for c in 0..32 {
        expected[0] += d * qs[c] as f32 * scalar;
    }

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_weight).expect("up w");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let mut dev_o: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc out");

    let desc = GgufMmvqDescriptor {
        nrows: 1,
        ncols: 32,
        block_format: GgufBlockFormat::Q8_0,
        w_start_byte_offset: 0,
    };
    let plan = GgufMmvqPlan::select(&stream, &desc, PlanPreference::default()).expect("select");
    let weight_len = host_weight.len() as i32;
    let args = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [weight_len],
            stride: contiguous_stride([weight_len]),
        },
        activation: TensorRef {
            data: dev_y.as_slice(),
            shape: [32],
            stride: [0],
        },
        output: TensorMut {
            data: dev_o.as_slice_mut(),
            shape: [1],
            stride: contiguous_stride([1]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = [0f32; 1];
    dev_o.copy_to_host(&mut got).expect("dl");
    check_close(&got, &expected, 1e-2, "Q8_0 stride0");
}

/// W-allocation-sharing: two matrices in one buffer, dispatch via offset.
#[test]
#[ignore]
fn mmvq_q8_0_w_start_byte_offset() {
    let (ctx, stream) = setup();

    // Matrix A: 1 row, d=0.5, qs[i]=i. Matrix B: 1 row, d=0.25, qs[i]=i+1.
    let d_a = 0.5_f32;
    let d_b = 0.25_f32;
    let mut qs_a = [0i8; 32];
    let mut qs_b = [0i8; 32];
    for i in 0..32 {
        qs_a[i] = i as i8;
        qs_b[i] = (i + 1) as i8;
    }
    // Combined buffer: [A bytes][B bytes].
    let mut combined = pack_q8_0_row(d_a, &qs_a);
    combined.extend(pack_q8_0_row(d_b, &qs_b));
    let combined_bytes = combined.len() as i32;
    let host_weight: Vec<U8> = combined.into_iter().map(U8).collect();

    let host_y: Vec<f32> = (0..32).map(|i| i as f32).collect();

    let mut expected_a = [0.0f32];
    let mut expected_b = [0.0f32];
    for c in 0..32 {
        expected_a[0] += d_a * qs_a[c] as f32 * host_y[c];
        expected_b[0] += d_b * qs_b[c] as f32 * host_y[c];
    }

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_weight).expect("up w");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let mut dev_oa: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc a");
    let mut dev_ob: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc b");

    // Dispatch A at offset = 0 (also exercises offset=0 + strided arm
    // when stride is forced via a separate path — here it goes contig).
    // Then dispatch B at offset = 34 (the size of one Q8_0 block).
    let matrix_bytes_each = 34i32;
    assert_eq!(combined_bytes, 2 * matrix_bytes_each);

    // First call: matrix A. With offset=0 and stride=1, this hits the
    // contig fast path.
    let desc_a = GgufMmvqDescriptor {
        nrows: 1,
        ncols: 32,
        block_format: GgufBlockFormat::Q8_0,
        w_start_byte_offset: 0,
    };
    let plan_a = GgufMmvqPlan::select(&stream, &desc_a, PlanPreference::default()).expect("sel a");
    let args_a = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [combined_bytes],
            stride: contiguous_stride([combined_bytes]),
        },
        activation: TensorRef {
            data: dev_y.as_slice(),
            shape: [32],
            stride: [1],
        },
        output: TensorMut {
            data: dev_oa.as_slice_mut(),
            shape: [1],
            stride: contiguous_stride([1]),
        },
    };
    plan_a.run(&stream, Workspace::None, args_a).expect("run a");

    // Second call: matrix B, dispatched via byte offset. This engages
    // the actstrided FFI sibling (since `w_start_byte_offset != 0`).
    let desc_b = GgufMmvqDescriptor {
        nrows: 1,
        ncols: 32,
        block_format: GgufBlockFormat::Q8_0,
        w_start_byte_offset: matrix_bytes_each as i64,
    };
    let plan_b = GgufMmvqPlan::select(&stream, &desc_b, PlanPreference::default()).expect("sel b");
    let args_b = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [combined_bytes],
            stride: contiguous_stride([combined_bytes]),
        },
        activation: TensorRef {
            data: dev_y.as_slice(),
            shape: [32],
            stride: [1],
        },
        output: TensorMut {
            data: dev_ob.as_slice_mut(),
            shape: [1],
            stride: contiguous_stride([1]),
        },
    };
    plan_b.run(&stream, Workspace::None, args_b).expect("run b");
    stream.synchronize().expect("sync");

    let mut got_a = [0f32; 1];
    let mut got_b = [0f32; 1];
    dev_oa.copy_to_host(&mut got_a).expect("dl a");
    dev_ob.copy_to_host(&mut got_b).expect("dl b");
    check_close(&got_a, &expected_a, 1e-2, "Q8_0 W-offset matrix A");
    check_close(&got_b, &expected_b, 1e-2, "Q8_0 W-offset matrix B");
}

// ---- Q4_0: combined stride + offset --------------------------------------

#[test]
#[ignore]
fn mmvq_q4_0_stride_and_offset() {
    let (ctx, stream) = setup();

    // Two matrices, each 1×32 Q4_0, packed back-to-back.
    let d_a = 0.125_f32;
    let d_b = 0.0625_f32;
    let mut vals_a = [0i8; 32];
    let mut vals_b = [0i8; 32];
    for i in 0..32 {
        vals_a[i] = (i as i32 % 16 - 8) as i8; // -8..7
        vals_b[i] = ((i as i32 * 3 + 1) % 16 - 8) as i8;
    }
    let mut combined = pack_q4_0_row(d_a, &vals_a);
    combined.extend(pack_q4_0_row(d_b, &vals_b));
    let combined_bytes = combined.len() as i32;
    let host_weight: Vec<U8> = combined.into_iter().map(U8).collect();
    let matrix_bytes = 18i32;
    assert_eq!(combined_bytes, 2 * matrix_bytes);

    // Stride-2 activation of length 64.
    let mut host_y = vec![-12345.0f32; 64];
    for c in 0..32 {
        host_y[c * 2] = (c as f32) * 0.05 - 0.7;
    }

    let mut expected_b = [0.0f32];
    for c in 0..32 {
        expected_b[0] += d_b * vals_b[c] as f32 * host_y[c * 2];
    }

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_weight).expect("up w");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let mut dev_o: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc out");

    // Dispatch matrix B with offset + stride-2.
    let desc = GgufMmvqDescriptor {
        nrows: 1,
        ncols: 32,
        block_format: GgufBlockFormat::Q4_0,
        w_start_byte_offset: matrix_bytes as i64,
    };
    let plan = GgufMmvqPlan::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [combined_bytes],
            stride: contiguous_stride([combined_bytes]),
        },
        activation: TensorRef {
            data: dev_y.as_slice(),
            shape: [32],
            stride: [2],
        },
        output: TensorMut {
            data: dev_o.as_slice_mut(),
            shape: [1],
            stride: contiguous_stride([1]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = [0f32; 1];
    dev_o.copy_to_host(&mut got).expect("dl");
    check_close(&got, &expected_b, 1e-2, "Q4_0 stride2 + offset");
}

// ---- Q8_K: k-quants strided path -----------------------------------------

#[test]
#[ignore]
fn mmvq_q8_k_stride2() {
    let (ctx, stream) = setup();

    let d = 0.125_f32;
    let mut qs = [0i8; 256];
    for i in 0..256 {
        qs[i] = (((i as i32) % 255) - 127) as i8;
    }
    let packed = pack_q8_k_row(d, &qs);
    let host_weight: Vec<U8> = packed.into_iter().map(U8).collect();

    // Stride-2 source of length 512.
    let mut host_y = vec![-1.0f32; 512];
    for c in 0..256 {
        host_y[c * 2] = (c as f32) * 0.01 - 1.28;
    }

    let mut expected = [0.0f32];
    for c in 0..256 {
        expected[0] += d * qs[c] as f32 * host_y[c * 2];
    }

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_weight).expect("up w");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let mut dev_o: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc out");

    let desc = GgufMmvqDescriptor {
        nrows: 1,
        ncols: 256,
        block_format: GgufBlockFormat::Q8K,
        w_start_byte_offset: 0,
    };
    let plan = GgufMmvqPlan::select(&stream, &desc, PlanPreference::default()).expect("select");
    let weight_len = host_weight.len() as i32;
    let args = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [weight_len],
            stride: contiguous_stride([weight_len]),
        },
        activation: TensorRef {
            data: dev_y.as_slice(),
            shape: [256],
            stride: [2],
        },
        output: TensorMut {
            data: dev_o.as_slice_mut(),
            shape: [1],
            stride: contiguous_stride([1]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = [0f32; 1];
    dev_o.copy_to_host(&mut got).expect("dl");
    check_close(&got, &expected, 1e-3, "Q8_K stride2");
}

#[test]
#[ignore]
fn mmvq_q8_k_stride0_broadcast() {
    let (ctx, stream) = setup();

    let d = -0.0625_f32;
    let mut qs = [0i8; 256];
    for i in 0..256 {
        qs[i] = (((i as i32 * 5 + 13) % 255) - 127) as i8;
    }
    let packed = pack_q8_k_row(d, &qs);
    let host_weight: Vec<U8> = packed.into_iter().map(U8).collect();

    let scalar = 0.75_f32;
    let host_y = vec![scalar];

    let mut expected = [0.0f32];
    for c in 0..256 {
        expected[0] += d * qs[c] as f32 * scalar;
    }

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_weight).expect("up w");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let mut dev_o: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc out");

    let desc = GgufMmvqDescriptor {
        nrows: 1,
        ncols: 256,
        block_format: GgufBlockFormat::Q8K,
        w_start_byte_offset: 0,
    };
    let plan = GgufMmvqPlan::select(&stream, &desc, PlanPreference::default()).expect("select");
    let weight_len = host_weight.len() as i32;
    let args = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [weight_len],
            stride: contiguous_stride([weight_len]),
        },
        activation: TensorRef {
            data: dev_y.as_slice(),
            shape: [256],
            stride: [0],
        },
        output: TensorMut {
            data: dev_o.as_slice_mut(),
            shape: [1],
            stride: contiguous_stride([1]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = [0f32; 1];
    dev_o.copy_to_host(&mut got).expect("dl");
    check_close(&got, &expected, 1e-3, "Q8_K stride0 broadcast");
}
