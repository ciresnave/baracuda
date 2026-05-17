//! Real-GPU smoke test for `GgufDequantizePlan` — Phase 8 Milestone 8.4.
//!
//! Builds one `block_q4_0` by hand on the host with:
//!   - `d = 1.0` (fp16)
//!   - `qs = [0xFF; 16]`   (every nibble = 0xF = 15)
//!
//! The Q4_0 dequant rule is `y = (q_nibble - 8) * d`, so every output
//! element should equal `(15 - 8) * 1.0 = 7.0`. Confirms struct layout,
//! block-format dispatch, and ABI alignment between the FFI launchers
//! and the Rust `BlockQ4_0` struct.
//!
//! Marked `#[ignore]` per the project convention for real-GPU tests.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BlockQ4_0, GgufBlockFormat, GgufDequantizeArgs, GgufDequantizeDescriptor,
    GgufDequantizePlan, PlanPreference, TensorMut, TensorRef, Workspace, U8,
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
fn gguf_dequantize_q4_0_all_high_nibbles() {
    let (ctx, stream) = setup();

    // Build one block_q4_0 on the host: scale d = 1.0 (fp16),
    // every nibble = 0xF. Per the Q4_0 decode rule
    // (`y = (q & 0xF - 8) * d` for low nibbles,
    //  `y = (q >> 4  - 8) * d` for high nibbles), every output
    // element should equal (15 - 8) * 1.0 = 7.0.
    let d_fp16: u16 = half::f16::from_f32(1.0).to_bits();
    let block = BlockQ4_0 {
        d: d_fp16,
        qs: [0xFF; 16],
    };
    let block_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            (&block as *const BlockQ4_0) as *const u8,
            core::mem::size_of::<BlockQ4_0>(),
        )
    };
    let host_bytes_storage: Vec<U8> = block_bytes.iter().copied().map(U8).collect();
    let numel: i64 = 32;
    let bytes_len = host_bytes_storage.len() as i32;
    assert_eq!(bytes_len as usize, 18);

    let dev_packed = DeviceBuffer::from_slice(&ctx, &host_bytes_storage).expect("up packed");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel as usize).expect("alloc out");

    let desc = GgufDequantizeDescriptor {
        numel,
        block_format: GgufBlockFormat::Q4_0,
    };
    let plan = GgufDequantizePlan::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = GgufDequantizeArgs {
        input: TensorRef {
            data: dev_packed.as_slice(),
            shape: [bytes_len],
            stride: contiguous_stride([bytes_len]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel as usize];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, &g) in got.iter().enumerate() {
        assert_eq!(
            g.to_bits(),
            7.0_f32.to_bits(),
            "dequant Q4_0 mismatch @ {i}: got {g}, expected 7.0",
        );
    }
}

#[test]
#[ignore]
fn gguf_dequantize_q8_0_known_values() {
    use baracuda_kernels::BlockQ8_0;

    let (ctx, stream) = setup();

    // block_q8_0: d = 0.25, qs = [1, 2, 3, ..., 32] (i8).
    // Dequant rule: y[i] = d * qs[i] → [0.25, 0.5, 0.75, ..., 8.0].
    let mut qs = [0i8; 32];
    for (i, q) in qs.iter_mut().enumerate() {
        *q = (i + 1) as i8;
    }
    let block = BlockQ8_0 {
        d: half::f16::from_f32(0.25).to_bits(),
        qs,
    };
    let block_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            (&block as *const BlockQ8_0) as *const u8,
            core::mem::size_of::<BlockQ8_0>(),
        )
    };
    assert_eq!(block_bytes.len(), 34);
    let host_bytes: Vec<U8> = block_bytes.iter().copied().map(U8).collect();
    let bytes_len = host_bytes.len() as i32;
    let numel: i64 = 32;

    let dev_packed = DeviceBuffer::from_slice(&ctx, &host_bytes).expect("up packed");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel as usize).expect("alloc out");

    let desc = GgufDequantizeDescriptor {
        numel,
        block_format: GgufBlockFormat::Q8_0,
    };
    let plan = GgufDequantizePlan::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = GgufDequantizeArgs {
        input: TensorRef {
            data: dev_packed.as_slice(),
            shape: [bytes_len],
            stride: contiguous_stride([bytes_len]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel as usize];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, &g) in got.iter().enumerate() {
        let expected = 0.25_f32 * ((i + 1) as f32);
        let abs_err = (g - expected).abs();
        assert!(
            abs_err < 1e-6,
            "dequant Q8_0 mismatch @ {i}: got {g}, expected {expected} (|err| = {abs_err})",
        );
    }
}
