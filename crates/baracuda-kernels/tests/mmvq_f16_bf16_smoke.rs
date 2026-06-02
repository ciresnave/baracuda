//! Phase 18.1 — real-GPU smoke tests for f16 / bf16 activation MMVQ.
//!
//! Coverage is representative, not exhaustive (the kernel template is
//! shared per-family — type-0/1 reuses `dequantize_mul_mat_vec`, k-quants
//! are bespoke per format, both sets share the same `mmvq_io<T>` cast
//! helpers in `baracuda_gguf.cuh`):
//!
//!   * Q8_0 + f16  (type-0/1 representative, contig)
//!   * Q8_0 + bf16 (type-0/1 representative, contig)
//!   * Q4_K + f16  (k-quants representative, contig)
//!   * Q8_K + bf16 (bespoke baracuda Q8_K kernel + bf16, contig)
//!   * Q8_0 + f16  + stride_y=2  (strided sibling path)
//!   * Q4_K + bf16 + stride_y=0  (broadcast-degenerate path)
//!
//! Tolerance: f16/bf16 round-trip on activation read + on dst write +
//! f32 accumulator dequant rounding compounds to a few-percent error.
//! Use 2% relative + a small absolute floor — matches the MoE WMMA mix
//! that Phase 15.3 tuned tolerances for.

// Test names use the upstream llama.cpp K-block notation (Q4_K etc.).
#![allow(non_snake_case)]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BlockQ4K, BlockQ8K, BlockQ8_0, GgufBlockFormat, GgufMmvqArgs,
    GgufMmvqDescriptor, GgufMmvqPlan, PlanPreference, TensorMut, TensorRef, Workspace, U8,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// ---- packing helpers (same convention as mmvq_strided_smoke.rs) ----------

fn pack_q8_0_row(d_f32: f32, qs: &[i8; 32]) -> Vec<u8> {
    let blk = BlockQ8_0 { d: f16::from_f32(d_f32).to_bits(), qs: *qs };
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

/// Pack a single block_q4_K row with a "trivial" scales-table: all
/// sub-block scales = 1, all sub-block mins = 0. dall (super-block scale)
/// times each sub-block scale × quant gives the dequant.
///
/// Layout: `dm[0]` = super-block scale (fp16), `dm[1]` = super-block min
/// (fp16, here 0). `scales[12]` = packed 6-bit scales/mins where the
/// `get_scale_min_k4` decoder reads pairs (sc, m) for is∈{0..7}. We set
/// all sub-block scales to 1 and mins to 0, which means every
/// `q4[l] * sc_byte` directly contributes to the inner-product (with
/// `sc[1] >> 4` and `sc[5] >> 4` for the upper nibbles also contributing
/// scaled-by-1/16; we structure the test so that only the lower-nibble
/// path contributes and the `sc[5]/16` term is zero).
///
/// The kernel multiplies the lower-nibble accumulator by `sc[0]` and the
/// upper-nibble accumulator by `sc[1] / 16`. We set sc[0]=1, sc[1]=0,
/// sc[4]=1, sc[5]=0 — so only the lower nibbles contribute, and the
/// expected result is `dall * Σ_c quant_low(c) * y[c]` for the lower
/// 128 columns (and same for upper 128 via `sc[4]`), with `dmin*smin=0`.
fn pack_q4_k_row_trivial(d_f32: f32, nibbles_low: &[u8; 128]) -> (Vec<u8>, Vec<u8>) {
    use baracuda_kernels::BlockQ4K;
    // Build the 12-byte scales array such that get_scale_min_k4 returns
    // sc∈{1,0,1,0,1,0,1,0} and m∈{0,0,0,0,0,0,0,0} for is∈{0..7}.
    // Looking at get_scale_min_k4 implementation:
    //   if j < 4: d = q[j] & 63 ; m = q[j+4] & 63
    //   else (j∈{4,5,6,7}): d = (q[j+4] & 0xF) | ((q[j-4]>>6)<<4);
    //                        m = (q[j+4] >> 4) | ((q[j-0]>>6) << 4)
    // So we want q[0..4] & 63 ∈ {1,0,1,0}, q[4..8] & 63 = 0 (m for j<4),
    // and for j∈{4..8}: q[j+4] = q[8..12]; need (q[8..12] & 0xF) ∈ {1,0,1,0},
    // and the high bits of q[0..4] all zero (we already have q[0..4]<=1<63),
    // m components also need to be zero: (q[8..12] >> 4) = 0 → q[8..12] ≤ 0xF.
    // Easiest scales array:  [1,0,1,0, 0,0,0,0, 1,0,1,0]
    let scales: [u8; 12] = [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0];

    // qs is 128 bytes, 256 nibbles. The kernel reads 32 bytes at q_offset
    // (varies per-thread), interprets via 0x0f0f0f0f / 0xf0f0f0f0 masks
    // to pull lower / upper nibbles. We want **only the lower nibbles** to
    // carry signal in this test, so we set the upper nibble = 0 everywhere:
    let mut qs = [0u8; 128];
    for i in 0..128 {
        qs[i] = nibbles_low[i] & 0xF; // low nibble is signal, high nibble = 0
    }

    let blk = BlockQ4K {
        dm: [f16::from_f32(d_f32).to_bits(), f16::from_f32(0.0).to_bits()],
        scales,
        qs,
    };
    let bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            (&blk as *const BlockQ4K) as *const u8,
            core::mem::size_of::<BlockQ4K>(),
        )
    };
    // Return both the bytes and the scales (echo) so the test ref can
    // skip having to re-derive them.
    (bytes.to_vec(), scales.to_vec())
}

fn check_close_f32(got: &[f32], expected: &[f32], rel_tol: f32, label: &str) {
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        let tol = rel_tol * e.abs().max(1e-3);
        assert!(
            abs_err < tol,
            "{label} mismatch @ {i}: got {g}, expected {e} (|err|={abs_err}, tol={tol})"
        );
    }
}

// =============================================================================
// Q8_0 + f16 contig
// =============================================================================

#[test]
#[ignore]
fn mmvq_q8_0_f16_smoke() {
    let (ctx, stream) = setup();

    let d = 0.5_f32;
    let mut qs = [0i8; 32];
    for i in 0..32 {
        qs[i] = (i as i32 - 16) as i8;
    }
    let packed = pack_q8_0_row(d, &qs);
    let host_w: Vec<U8> = packed.into_iter().map(U8).collect();

    // f16 activation
    let host_y_f32: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05 - 0.5).collect();
    let host_y_f16: Vec<f16> = host_y_f32.iter().map(|&x| f16::from_f32(x)).collect();

    // Reference: dequant + multiply, with the activation f16 round-trip
    // baked into the expected value.
    let mut expected = [0.0f32];
    for c in 0..32 {
        expected[0] += d * (qs[c] as f32) * host_y_f16[c].to_f32();
    }
    // And then the output is cast back to f16 — model that round-trip
    // by comparing as f32 with a tolerance that covers the cast.

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up w");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y_f16).expect("up y");
    let mut dev_o: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).expect("alloc out");

    let desc = GgufMmvqDescriptor {
        nrows: 1, ncols: 32,
        block_format: GgufBlockFormat::Q8_0,
        w_start_byte_offset: 0,
    };
    let plan = GgufMmvqPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [host_w.len() as i32],
            stride: contiguous_stride([host_w.len() as i32]),
        },
        activation: TensorRef { data: dev_y.as_slice(), shape: [32], stride: [1] },
        output: TensorMut {
            data: dev_o.as_slice_mut(),
            shape: [1],
            stride: contiguous_stride([1]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_f16 = [f16::from_f32(0.0); 1];
    dev_o.copy_to_host(&mut got_f16).expect("dl");
    let got = [got_f16[0].to_f32()];
    check_close_f32(&got, &expected, 0.02, "Q8_0 + f16");
}

// =============================================================================
// Q8_0 + bf16 contig
// =============================================================================

#[test]
#[ignore]
fn mmvq_q8_0_bf16_smoke() {
    let (ctx, stream) = setup();

    let d = 0.5_f32;
    let mut qs = [0i8; 32];
    for i in 0..32 {
        qs[i] = (i as i32 - 16) as i8;
    }
    let packed = pack_q8_0_row(d, &qs);
    let host_w: Vec<U8> = packed.into_iter().map(U8).collect();

    let host_y_f32: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05 - 0.5).collect();
    let host_y_bf16: Vec<bf16> = host_y_f32.iter().map(|&x| bf16::from_f32(x)).collect();

    let mut expected = [0.0f32];
    for c in 0..32 {
        expected[0] += d * (qs[c] as f32) * host_y_bf16[c].to_f32();
    }

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up w");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y_bf16).expect("up y");
    let mut dev_o: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).expect("alloc out");

    let desc = GgufMmvqDescriptor {
        nrows: 1, ncols: 32,
        block_format: GgufBlockFormat::Q8_0,
        w_start_byte_offset: 0,
    };
    let plan = GgufMmvqPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [host_w.len() as i32],
            stride: contiguous_stride([host_w.len() as i32]),
        },
        activation: TensorRef { data: dev_y.as_slice(), shape: [32], stride: [1] },
        output: TensorMut {
            data: dev_o.as_slice_mut(),
            shape: [1],
            stride: contiguous_stride([1]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_bf16 = [bf16::from_f32(0.0); 1];
    dev_o.copy_to_host(&mut got_bf16).expect("dl");
    let got = [got_bf16[0].to_f32()];
    // bf16 has 7-bit mantissa, so larger tolerance for accumulation
    // round-off than f16 (10-bit mantissa).
    check_close_f32(&got, &expected, 0.03, "Q8_0 + bf16");
}

// =============================================================================
// Q4_K + f16 contig  (k-quant representative)
// =============================================================================

#[test]
#[ignore]
fn mmvq_q4_K_f16_smoke() {
    let (ctx, stream) = setup();

    // Trivial scales table → expected = dall × (sum over the 128 lower
    // sub-block bytes of nibble_low × y[c]) summed over both halves of
    // the super-block (lower 128 via sc[0]=1, upper 128 via sc[4]=1).
    let d = 0.0625_f32;
    let mut nibbles = [0u8; 128];
    for i in 0..128 {
        nibbles[i] = ((i % 16) as u8) & 0xF;
    }
    let (packed, _scales) = pack_q4_k_row_trivial(d, &nibbles);
    let host_w: Vec<U8> = packed.into_iter().map(U8).collect();

    let host_y_f32: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.0).collect();
    let host_y_f16: Vec<f16> = host_y_f32.iter().map(|&x| f16::from_f32(x)).collect();

    // Expected: kernel walks the super-block in two halves of 128 cols
    // each. Lower half (y[0..128]) uses sc[0]=1 → contribution = nibble.
    // Upper half (y[128..256]) uses sc[4]=1 with the qs layout being
    // 32 contiguous bytes per il group. The kernel layout is bespoke;
    // computing the *exact* reference here would re-implement the kernel.
    // Instead, this test just sanity-checks that the f16 dispatch
    // doesn't NaN/crash and produces a finite number reasonably close
    // to the f32 baseline run at the same inputs.
    let f32_ref = run_q4k_f32_ref(&ctx, &stream, &host_w, &host_y_f32);

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up w");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y_f16).expect("up y");
    let mut dev_o: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).expect("alloc out");

    let desc = GgufMmvqDescriptor {
        nrows: 1, ncols: 256,
        block_format: GgufBlockFormat::Q4K,
        w_start_byte_offset: 0,
    };
    let plan = GgufMmvqPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [host_w.len() as i32],
            stride: contiguous_stride([host_w.len() as i32]),
        },
        activation: TensorRef { data: dev_y.as_slice(), shape: [256], stride: [1] },
        output: TensorMut {
            data: dev_o.as_slice_mut(),
            shape: [1],
            stride: contiguous_stride([1]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_f16 = [f16::from_f32(0.0); 1];
    dev_o.copy_to_host(&mut got_f16).expect("dl");
    let got = [got_f16[0].to_f32()];
    // Reference here is the f32 kernel's own output (cross-dtype check
    // rather than a host re-implementation of the Q4_K math).
    check_close_f32(&got, &f32_ref, 0.02, "Q4_K + f16 vs f32 ref");
}

/// Helper: run the same Q4_K MMVQ with the f32 path so the f16 / bf16
/// tests can compare against the kernel's own output rather than a
/// reimplemented host reference.
fn run_q4k_f32_ref(
    ctx: &Context,
    stream: &Stream,
    host_w: &[U8],
    host_y_f32: &[f32],
) -> [f32; 1] {
    let dev_w = DeviceBuffer::from_slice(ctx, host_w).expect("up w");
    let dev_y = DeviceBuffer::from_slice(ctx, host_y_f32).expect("up y");
    let mut dev_o: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, 1).expect("alloc out");

    let desc = GgufMmvqDescriptor {
        nrows: 1, ncols: host_y_f32.len() as i32,
        block_format: GgufBlockFormat::Q4K,
        w_start_byte_offset: 0,
    };
    let plan = GgufMmvqPlan::<f32>::select(stream, &desc, PlanPreference::default()).expect("sel");
    let args = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [host_w.len() as i32],
            stride: contiguous_stride([host_w.len() as i32]),
        },
        activation: TensorRef {
            data: dev_y.as_slice(),
            shape: [host_y_f32.len() as i32],
            stride: [1],
        },
        output: TensorMut {
            data: dev_o.as_slice_mut(),
            shape: [1],
            stride: contiguous_stride([1]),
        },
    };
    plan.run(stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = [0f32; 1];
    dev_o.copy_to_host(&mut got).expect("dl");
    got
}

// =============================================================================
// Q8_K + bf16 contig  (bespoke baracuda kernel)
// =============================================================================

#[test]
#[ignore]
fn mmvq_q8_K_bf16_smoke() {
    let (ctx, stream) = setup();

    let d = 0.125_f32;
    let mut qs = [0i8; 256];
    for i in 0..256 {
        qs[i] = (((i as i32) % 127) - 63) as i8;
    }
    let packed = pack_q8_k_row(d, &qs);
    let host_w: Vec<U8> = packed.into_iter().map(U8).collect();

    let host_y_f32: Vec<f32> = (0..256).map(|i| (i as f32) * 0.005 - 0.6).collect();
    let host_y_bf16: Vec<bf16> = host_y_f32.iter().map(|&x| bf16::from_f32(x)).collect();

    let mut expected = [0.0f32];
    for c in 0..256 {
        expected[0] += d * (qs[c] as f32) * host_y_bf16[c].to_f32();
    }

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up w");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y_bf16).expect("up y");
    let mut dev_o: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).expect("alloc out");

    let desc = GgufMmvqDescriptor {
        nrows: 1, ncols: 256,
        block_format: GgufBlockFormat::Q8K,
        w_start_byte_offset: 0,
    };
    let plan = GgufMmvqPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [host_w.len() as i32],
            stride: contiguous_stride([host_w.len() as i32]),
        },
        activation: TensorRef { data: dev_y.as_slice(), shape: [256], stride: [1] },
        output: TensorMut {
            data: dev_o.as_slice_mut(),
            shape: [1],
            stride: contiguous_stride([1]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_bf16 = [bf16::from_f32(0.0); 1];
    dev_o.copy_to_host(&mut got_bf16).expect("dl");
    let got = [got_bf16[0].to_f32()];
    check_close_f32(&got, &expected, 0.03, "Q8_K + bf16");
}

// =============================================================================
// Strided path coverage
// =============================================================================

/// Q8_0 + f16 with stride_y=2 — exercises the actstrided_f16 launcher.
#[test]
#[ignore]
fn mmvq_q8_0_f16_stride2_smoke() {
    let (ctx, stream) = setup();

    let d = 0.5_f32;
    let mut qs = [0i8; 32];
    for i in 0..32 {
        qs[i] = ((i as i32) % 16 - 7) as i8;
    }
    let packed = pack_q8_0_row(d, &qs);
    let host_w: Vec<U8> = packed.into_iter().map(U8).collect();

    let mut host_y_f32 = vec![-999.0f32; 64];
    for c in 0..32 {
        host_y_f32[c * 2] = (c as f32 + 1.0) * 0.05;
    }
    let host_y_f16: Vec<f16> = host_y_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let mut expected = [0.0f32];
    for c in 0..32 {
        expected[0] += d * (qs[c] as f32) * host_y_f16[c * 2].to_f32();
    }

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up w");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y_f16).expect("up y");
    let mut dev_o: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).expect("alloc out");

    let desc = GgufMmvqDescriptor {
        nrows: 1, ncols: 32,
        block_format: GgufBlockFormat::Q8_0,
        w_start_byte_offset: 0,
    };
    let plan = GgufMmvqPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [host_w.len() as i32],
            stride: contiguous_stride([host_w.len() as i32]),
        },
        activation: TensorRef { data: dev_y.as_slice(), shape: [32], stride: [2] },
        output: TensorMut {
            data: dev_o.as_slice_mut(),
            shape: [1],
            stride: contiguous_stride([1]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_f16 = [f16::from_f32(0.0); 1];
    dev_o.copy_to_host(&mut got_f16).expect("dl");
    let got = [got_f16[0].to_f32()];
    check_close_f32(&got, &expected, 0.02, "Q8_0 + f16 stride2");
}

/// Q4_K + bf16 with stride_y=0 — broadcast: every column reads y[0].
#[test]
#[ignore]
fn mmvq_q4_K_bf16_stride0_broadcast_smoke() {
    let (ctx, stream) = setup();

    let d = 0.0625_f32;
    let mut nibbles = [0u8; 128];
    for i in 0..128 {
        nibbles[i] = ((i % 16) as u8) & 0xF;
    }
    let (packed, _scales) = pack_q4_k_row_trivial(d, &nibbles);
    let host_w: Vec<U8> = packed.into_iter().map(U8).collect();

    let scalar_f32 = 0.5_f32;
    let scalar_bf16 = bf16::from_f32(scalar_f32);
    let host_y_bf16 = vec![scalar_bf16];

    // Cross-check vs. the f32 kernel running with a fully-broadcast f32
    // y buffer (same broadcast — single-element y, stride=0).
    let host_y_f32_broadcast = vec![scalar_bf16.to_f32()];
    let f32_ref = run_q4k_f32_ref_strided(&ctx, &stream, &host_w, &host_y_f32_broadcast, 0);

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up w");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y_bf16).expect("up y");
    let mut dev_o: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).expect("alloc out");

    let desc = GgufMmvqDescriptor {
        nrows: 1, ncols: 256,
        block_format: GgufBlockFormat::Q4K,
        w_start_byte_offset: 0,
    };
    let plan = GgufMmvqPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [host_w.len() as i32],
            stride: contiguous_stride([host_w.len() as i32]),
        },
        activation: TensorRef { data: dev_y.as_slice(), shape: [256], stride: [0] },
        output: TensorMut {
            data: dev_o.as_slice_mut(),
            shape: [1],
            stride: contiguous_stride([1]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_bf16 = [bf16::from_f32(0.0); 1];
    dev_o.copy_to_host(&mut got_bf16).expect("dl");
    let got = [got_bf16[0].to_f32()];
    check_close_f32(&got, &f32_ref, 0.03, "Q4_K + bf16 stride0 broadcast");
}

/// Strided f32 Q4_K helper (mirrors `run_q4k_f32_ref` but for the
/// strided dispatch path).
fn run_q4k_f32_ref_strided(
    ctx: &Context,
    stream: &Stream,
    host_w: &[U8],
    host_y_f32: &[f32],
    stride_y: i64,
) -> [f32; 1] {
    let dev_w = DeviceBuffer::from_slice(ctx, host_w).expect("up w");
    let dev_y = DeviceBuffer::from_slice(ctx, host_y_f32).expect("up y");
    let mut dev_o: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, 1).expect("alloc out");

    let desc = GgufMmvqDescriptor {
        nrows: 1, ncols: 256,
        block_format: GgufBlockFormat::Q4K,
        w_start_byte_offset: 0,
    };
    let plan = GgufMmvqPlan::<f32>::select(stream, &desc, PlanPreference::default()).expect("sel");
    let args = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [host_w.len() as i32],
            stride: contiguous_stride([host_w.len() as i32]),
        },
        activation: TensorRef {
            data: dev_y.as_slice(),
            shape: [256],
            stride: [stride_y],
        },
        output: TensorMut {
            data: dev_o.as_slice_mut(),
            shape: [1],
            stride: contiguous_stride([1]),
        },
    };
    plan.run(stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = [0f32; 1];
    dev_o.copy_to_host(&mut got).expect("dl");
    got
}
