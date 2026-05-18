//! Real-GPU smoke test for `GgufMmvqPlan` — Phase 8 Milestone 8.4.
//!
//! Builds a `[nrows=2, ncols=32]` Q8_0-packed weight matrix on the host
//! and a length-32 f32 activation vector. Verifies the GPU MMVQ output
//! against a hand-computed full-precision dequant-then-matmul reference.
//! Q8_0 is chosen for the smoke test because its dequant rule
//! (`w[i] = qs[i] * d`) is the simplest of all block formats — any
//! mismatch isolates the MMVQ launcher / kernel rather than the
//! dequant math.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BlockQ8K, BlockQ8_0, GgufBlockFormat, GgufMmvqArgs, GgufMmvqDescriptor,
    GgufMmvqPlan, PlanPreference, TensorMut, TensorRef, Workspace, U8,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn gguf_mmvq_q8_0_2x32() {
    let (ctx, stream) = setup();

    // Build two Q8_0 weight rows on the host.
    // Row 0: d = 0.5, qs[i] = i (so w[i] = 0.5 * i, i ∈ [0, 32))
    // Row 1: d = 0.25, qs[i] = i + 1 (so w[i] = 0.25 * (i+1))
    let d0 = 0.5_f32;
    let d1 = 0.25_f32;

    let mut qs0 = [0i8; 32];
    let mut qs1 = [0i8; 32];
    for i in 0..32 {
        qs0[i] = i as i8;
        qs1[i] = (i + 1) as i8;
    }
    let row0 = BlockQ8_0 { d: half::f16::from_f32(d0).to_bits(), qs: qs0 };
    let row1 = BlockQ8_0 { d: half::f16::from_f32(d1).to_bits(), qs: qs1 };

    let mut packed_bytes: Vec<u8> = Vec::with_capacity(2 * 34);
    let blocks = [row0, row1];
    for blk in blocks.iter() {
        let bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                (blk as *const BlockQ8_0) as *const u8,
                core::mem::size_of::<BlockQ8_0>(),
            )
        };
        packed_bytes.extend_from_slice(bytes);
    }
    assert_eq!(packed_bytes.len(), 68);
    let host_weight: Vec<U8> = packed_bytes.into_iter().map(U8).collect();

    // Activation: y[i] = i (i ∈ [0, 32)).
    let host_activation: Vec<f32> = (0..32).map(|i| i as f32).collect();

    // Reference: out[r] = Σ_c w[r, c] * y[c].
    let mut expected = [0.0_f32; 2];
    for c in 0..32 {
        expected[0] += d0 * (qs0[c] as f32) * host_activation[c];
        expected[1] += d1 * (qs1[c] as f32) * host_activation[c];
    }

    let nrows: i32 = 2;
    let ncols: i32 = 32;
    let weight_bytes_len = host_weight.len() as i32;

    let dev_weight = DeviceBuffer::from_slice(&ctx, &host_weight).expect("up weight");
    let dev_activation = DeviceBuffer::from_slice(&ctx, &host_activation).expect("up act");
    let mut dev_out: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 2).expect("alloc out");

    let desc = GgufMmvqDescriptor {
        nrows,
        ncols,
        block_format: GgufBlockFormat::Q8_0,
    };
    let plan = GgufMmvqPlan::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_weight.as_slice(),
            shape: [weight_bytes_len],
            stride: contiguous_stride([weight_bytes_len]),
        },
        activation: TensorRef {
            data: dev_activation.as_slice(),
            shape: [ncols],
            stride: contiguous_stride([ncols]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [nrows],
            stride: contiguous_stride([nrows]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = [0f32; 2];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        // Q8_0 dequant goes through fp16 scale → tolerate fp16 round
        // (relative ≈ 1e-3 worst case for the scale, additive over 32
        // products → a few ULPs of fp32).
        let tol = 1e-2_f32 * e.abs().max(1.0);
        assert!(
            abs_err < tol,
            "mmvq Q8_0 mismatch @ {i}: got {g}, expected {e} (|err| = {abs_err}, tol = {tol})",
        );
    }
}

#[test]
#[ignore]
fn gguf_mmvq_q8_K_2x256() {
    // Phase 11.4 — Q8_K MMVQ is now a supported bespoke kernel (closing
    // the Fuel team's feedback gap). Geometry: 2 rows × 1 super-block
    // (ncols = QK_K = 256). Q8_K block layout: f32 scale `d`, 256 i8
    // quants, 16 i16 bsums (unused on the f32-activation MMVQ path).
    let (ctx, stream) = setup();

    let d0 = 0.125_f32;
    let d1 = -0.0625_f32;

    let mut qs0 = [0i8; 256];
    let mut qs1 = [0i8; 256];
    for i in 0..256 {
        // Mix sign + range. qs in [-127, 127].
        qs0[i] = (((i as i32) % 255) - 127) as i8;
        qs1[i] = (((i as i32 * 3 + 5) % 255) - 127) as i8;
    }
    let bsums = [0i16; 16];
    let row0 = BlockQ8K { d: d0, qs: qs0, bsums };
    let row1 = BlockQ8K { d: d1, qs: qs1, bsums };

    let mut packed_bytes: Vec<u8> = Vec::with_capacity(2 * core::mem::size_of::<BlockQ8K>());
    for blk in [&row0, &row1] {
        let bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                (blk as *const BlockQ8K) as *const u8,
                core::mem::size_of::<BlockQ8K>(),
            )
        };
        packed_bytes.extend_from_slice(bytes);
    }
    assert_eq!(packed_bytes.len(), 2 * 292);
    let host_weight: Vec<U8> = packed_bytes.into_iter().map(U8).collect();

    // Activation: y[i] = sin-ish ramp to avoid trivial integer products.
    let host_activation: Vec<f32> =
        (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();

    // Reference: out[r] = d_r * Σ_c qs[r,c] * y[c].
    let mut expected = [0.0_f32; 2];
    for c in 0..256 {
        expected[0] += d0 * (qs0[c] as f32) * host_activation[c];
        expected[1] += d1 * (qs1[c] as f32) * host_activation[c];
    }

    let nrows: i32 = 2;
    let ncols: i32 = 256;
    let weight_bytes_len = host_weight.len() as i32;

    let dev_weight = DeviceBuffer::from_slice(&ctx, &host_weight).expect("up weight");
    let dev_activation = DeviceBuffer::from_slice(&ctx, &host_activation).expect("up act");
    let mut dev_out: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 2).expect("alloc out");

    let desc = GgufMmvqDescriptor {
        nrows,
        ncols,
        block_format: GgufBlockFormat::Q8K,
    };
    let plan = GgufMmvqPlan::select(&stream, &desc, PlanPreference::default())
        .expect("Q8_K MMVQ select (Phase 11.4) should succeed");
    let args = GgufMmvqArgs {
        weight: TensorRef {
            data: dev_weight.as_slice(),
            shape: [weight_bytes_len],
            stride: contiguous_stride([weight_bytes_len]),
        },
        activation: TensorRef {
            data: dev_activation.as_slice(),
            shape: [ncols],
            stride: contiguous_stride([ncols]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [nrows],
            stride: contiguous_stride([nrows]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = [0f32; 2];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        // Q8_K uses an f32 scale (no fp16 round-trip like Q8_0); the
        // only error source is f32 accumulation reorder under the
        // 32-way warp-shuffle reduction. Tolerance ≈ 256 * eps * |e|.
        let tol = 1e-3_f32 * e.abs().max(1.0);
        assert!(
            abs_err < tol,
            "mmvq Q8_K mismatch @ {i}: got {g}, expected {e} (|err| = {abs_err}, tol = {tol})",
        );
    }
}
