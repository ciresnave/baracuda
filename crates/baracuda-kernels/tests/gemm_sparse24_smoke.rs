//! Real-GPU smoke test for `GemmSparse24Plan` (Phase 54).
//!
//! Two assertions:
//!
//! 1. **2:4 GEMM matches dense reference**. Pre-zero the 2:4-equivalent
//!    cells in a dense weight matrix; compute Y = X @ W^T via a host-
//!    side reference; compare against the sparse-24 plan's output.
//!    Tolerance is loose (f32 reference vs f32 device — ±1e-3 abs).
//! 2. **K-mismatch is rejected**. K not divisible by 8 must
//!    `Error::Unsupported` from `select`.
//!
//! `#[ignore]` by default — requires a real CUDA device + the
//! `xformers_sparse24` cargo feature.

#![cfg(feature = "xformers_sparse24")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, GemmSparse24Args, GemmSparse24Descriptor, GemmSparse24Plan,
    PlanPreference, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

const N: i32 = 4;
const M: i32 = 8;
const K: i32 = 16; // multiple of 8

/// Build a 2:4-compatible weight matrix. For each (m, k_group=k/4)
/// pair, choose 2 positions out of 4 to be non-zero, and assign them
/// pseudo-random values from a base seed.
///
/// Returns `(compressed, metadata, dense)`.
fn build_sparse24_weights() -> (Vec<f32>, Vec<u16>, Vec<f32>) {
    let mut compressed = vec![0f32; (M * K / 2) as usize];
    let mut metadata = vec![0u16; (M * K / 8) as usize];
    let mut dense = vec![0f32; (M * K) as usize];

    for m in 0..M as usize {
        for kg in 0..(K / 4) as usize {
            // Pick 2 positions out of {0, 1, 2, 3}. Use kg parity for
            // determinism.
            let (pos0, pos1) = match kg & 3 {
                0 => (0u8, 1u8),
                1 => (1u8, 3u8),
                2 => (0u8, 2u8),
                _ => (2u8, 3u8),
            };
            // Two non-zero values for this group.
            let v0 = ((m * 100 + kg * 7) as f32 * 0.013).sin() * 0.5;
            let v1 = ((m * 100 + kg * 7 + 1) as f32 * 0.013).cos() * 0.5;
            compressed[m * (K / 2) as usize + kg * 2 + 0] = v0;
            compressed[m * (K / 2) as usize + kg * 2 + 1] = v1;
            // Encode metadata: byte = (pos1 << 2) | pos0.
            let mbyte: u8 = (pos1 << 2) | pos0;
            // Pack 2 bytes per uint16: kg even = low byte, kg odd = high.
            let word_idx = m * (K / 8) as usize + kg / 2;
            if (kg & 1) == 0 {
                metadata[word_idx] = (metadata[word_idx] & 0xFF00) | (mbyte as u16);
            } else {
                metadata[word_idx] = (metadata[word_idx] & 0x00FF) | ((mbyte as u16) << 8);
            }
            // Fill dense reconstruction.
            let dense_base = m * K as usize + kg * 4;
            dense[dense_base + pos0 as usize] = v0;
            dense[dense_base + pos1 as usize] = v1;
        }
    }
    (compressed, metadata, dense)
}

fn run_sparse24_gemm(
    ctx: &Context,
    stream: &Stream,
    x: &[f32],
    compressed: &[f32],
    metadata: &[u16],
) -> Vec<f32> {
    let dx = DeviceBuffer::from_slice(ctx, x).expect("up x");
    let dwc = DeviceBuffer::from_slice(ctx, compressed).expect("up wc");
    let dwm = DeviceBuffer::from_slice(ctx, metadata).expect("up wm");
    let mut dy: DeviceBuffer<f32> =
        DeviceBuffer::zeros(ctx, (N * M) as usize).expect("alloc y");

    let desc = GemmSparse24Descriptor {
        n: N,
        m: M,
        k: K,
        element: ElementKind::F32,
    };
    let plan = GemmSparse24Plan::<f32>::select(stream, &desc, PlanPreference::default())
        .expect("plan select");

    // Workspace: M * K * sizeof(f32) bytes.
    let ws_bytes = plan.workspace_size();
    assert_eq!(ws_bytes, (M as usize) * (K as usize) * core::mem::size_of::<f32>());
    let mut dws: DeviceBuffer<u8> = DeviceBuffer::zeros(ctx, ws_bytes).expect("alloc ws");

    plan.run(
        stream,
        Workspace::Borrowed(dws.as_slice_mut()),
        GemmSparse24Args {
            w_compressed: TensorRef {
                data: dwc.as_slice(),
                shape: [M, K / 2],
                stride: contiguous_stride([M, K / 2]),
            },
            w_metadata: TensorRef {
                data: dwm.as_slice(),
                shape: [M, K / 8],
                stride: contiguous_stride([M, K / 8]),
            },
            x: TensorRef {
                data: dx.as_slice(),
                shape: [N, K],
                stride: contiguous_stride([N, K]),
            },
            y: TensorMut {
                data: dy.as_slice_mut(),
                shape: [N, M],
                stride: contiguous_stride([N, M]),
            },
        },
    )
    .expect("plan run");
    stream.synchronize().expect("sync");

    let mut y = vec![0f32; (N * M) as usize];
    dy.copy_to_host(&mut y).expect("dl");
    y
}

/// Host-side reference: Y[n, m] = Σ_k X[n, k] * W_dense[m, k].
fn host_gemm_reference(x: &[f32], w_dense: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; (N * M) as usize];
    for n in 0..N as usize {
        for m in 0..M as usize {
            let mut acc = 0f64;
            for k in 0..K as usize {
                acc += (x[n * K as usize + k] as f64) * (w_dense[m * K as usize + k] as f64);
            }
            y[n * M as usize + m] = acc as f32;
        }
    }
    y
}

#[test]
#[ignore]
fn sparse24_gemm_matches_dense_reference() {
    let (ctx, stream) = setup();

    let (compressed, metadata, dense) = build_sparse24_weights();
    let x: Vec<f32> = (0..(N * K) as usize)
        .map(|i| ((i as f32) * 0.017 - 0.3).sin() * 0.4)
        .collect();

    let y_sparse = run_sparse24_gemm(&ctx, &stream, &x, &compressed, &metadata);
    let y_ref = host_gemm_reference(&x, &dense);

    let mut max_abs = 0f32;
    for i in 0..y_sparse.len() {
        let abs = (y_sparse[i] - y_ref[i]).abs();
        max_abs = max_abs.max(abs);
    }
    assert!(
        max_abs < 1e-3,
        "sparse24 vs host reference mismatch: max_abs={max_abs}"
    );
}

#[test]
#[ignore]
fn sparse24_rejects_k_not_multiple_of_8() {
    let (ctx, stream) = setup();
    let _ = ctx;
    let desc = GemmSparse24Descriptor {
        n: 4,
        m: 8,
        k: 12, // not divisible by 8
        element: ElementKind::F32,
    };
    let result = GemmSparse24Plan::<f32>::select(&stream, &desc, PlanPreference::default());
    assert!(result.is_err(), "expected error for K=12 (not div by 8)");
}
