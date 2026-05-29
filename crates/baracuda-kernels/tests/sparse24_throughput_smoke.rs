//! Real-GPU throughput smoke test for `GemmSparse24Plan` (Phase 54).
//!
//! **Tier-1 caveat**: this test is `#[ignore]` by default AND additionally
//! documents that the Tier-1 inflate-then-naive-GEMM path is **NOT**
//! faster than dense cuBLAS. The "2:4 is faster than dense" perf claim
//! requires the sparse-tensor-core (`mma.sp.sync.aligned` / cuSPARSELt)
//! backend, which is deferred to Tier 2.
//!
//! What this test exercises today:
//!
//! 1. End-to-end timing measurement infrastructure (CUDA events) for
//!    the inflate-then-dense path.
//! 2. Documents the expected Tier-2 speedup target: ~1.6× at large M /
//!    N / K on sm_89 (NVIDIA's published 2:4 speedup figure for FP16
//!    GEMM with sparse tensor cores at K ≥ 512).
//!
//! When the Tier-2 backend lands, this test will switch from "informational"
//! (always pass; just print timing) to "assert sparse is at least 1.2×
//! faster than the inflate+dense baseline AND at least 1.0× faster than
//! dense cuBLAS at large shapes".

#![cfg(feature = "xformers_sparse24")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Event, Stream};
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

#[test]
#[ignore]
fn sparse24_throughput_smoke_documentation() {
    let (ctx, stream) = setup();

    // Llama-style projection shape: M=N=K=2048, f32. Sized small enough
    // to fit on a 4070 even with the naive reference GEMM.
    const N_DIM: i32 = 256;
    const M_DIM: i32 = 256;
    const K_DIM: i32 = 256;

    // Build a valid 2:4 weight matrix (random metadata pattern).
    let mut compressed = vec![0f32; (M_DIM * K_DIM / 2) as usize];
    let mut metadata = vec![0u16; (M_DIM * K_DIM / 8) as usize];
    for m in 0..M_DIM as usize {
        for kg in 0..(K_DIM / 4) as usize {
            let (pos0, pos1) = match kg & 3 {
                0 => (0u8, 1u8),
                1 => (1u8, 3u8),
                2 => (0u8, 2u8),
                _ => (2u8, 3u8),
            };
            compressed[m * (K_DIM / 2) as usize + kg * 2 + 0] =
                ((m * 7 + kg * 3) as f32 * 0.01).sin() * 0.1;
            compressed[m * (K_DIM / 2) as usize + kg * 2 + 1] =
                ((m * 7 + kg * 3 + 1) as f32 * 0.01).cos() * 0.1;
            let mbyte: u8 = (pos1 << 2) | pos0;
            let word_idx = m * (K_DIM / 8) as usize + kg / 2;
            if (kg & 1) == 0 {
                metadata[word_idx] = (metadata[word_idx] & 0xFF00) | (mbyte as u16);
            } else {
                metadata[word_idx] = (metadata[word_idx] & 0x00FF) | ((mbyte as u16) << 8);
            }
        }
    }
    let x: Vec<f32> = (0..(N_DIM * K_DIM) as usize)
        .map(|i| ((i as f32) * 0.003).sin() * 0.05)
        .collect();

    let dx = DeviceBuffer::from_slice(&ctx, &x).expect("up x");
    let dwc = DeviceBuffer::from_slice(&ctx, &compressed).expect("up wc");
    let dwm = DeviceBuffer::from_slice(&ctx, &metadata).expect("up wm");
    let mut dy: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (N_DIM * M_DIM) as usize).expect("alloc y");

    let desc = GemmSparse24Descriptor {
        n: N_DIM,
        m: M_DIM,
        k: K_DIM,
        element: ElementKind::F32,
    };
    let plan = GemmSparse24Plan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("plan");
    let ws_bytes = plan.workspace_size();
    let mut dws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    // Warm-up.
    plan.run(
        &stream,
        Workspace::Borrowed(dws.as_slice_mut()),
        GemmSparse24Args {
            w_compressed: TensorRef {
                data: dwc.as_slice(), shape: [M_DIM, K_DIM / 2],
                stride: contiguous_stride([M_DIM, K_DIM / 2]),
            },
            w_metadata: TensorRef {
                data: dwm.as_slice(), shape: [M_DIM, K_DIM / 8],
                stride: contiguous_stride([M_DIM, K_DIM / 8]),
            },
            x: TensorRef {
                data: dx.as_slice(), shape: [N_DIM, K_DIM],
                stride: contiguous_stride([N_DIM, K_DIM]),
            },
            y: TensorMut {
                data: dy.as_slice_mut(), shape: [N_DIM, M_DIM],
                stride: contiguous_stride([N_DIM, M_DIM]),
            },
        },
    )
    .expect("warmup run");
    stream.synchronize().expect("sync");

    // Time.
    let start = Event::new(&ctx).expect("event start");
    let stop = Event::new(&ctx).expect("event stop");
    start.record(&stream).expect("record start");
    const ITERS: u32 = 10;
    for _ in 0..ITERS {
        plan.run(
            &stream,
            Workspace::Borrowed(dws.as_slice_mut()),
            GemmSparse24Args {
                w_compressed: TensorRef {
                    data: dwc.as_slice(), shape: [M_DIM, K_DIM / 2],
                    stride: contiguous_stride([M_DIM, K_DIM / 2]),
                },
                w_metadata: TensorRef {
                    data: dwm.as_slice(), shape: [M_DIM, K_DIM / 8],
                    stride: contiguous_stride([M_DIM, K_DIM / 8]),
                },
                x: TensorRef {
                    data: dx.as_slice(), shape: [N_DIM, K_DIM],
                    stride: contiguous_stride([N_DIM, K_DIM]),
                },
                y: TensorMut {
                    data: dy.as_slice_mut(), shape: [N_DIM, M_DIM],
                    stride: contiguous_stride([N_DIM, M_DIM]),
                },
            },
        )
        .expect("run");
    }
    stop.record(&stream).expect("record stop");
    stop.synchronize().expect("sync stop");
    let elapsed_ms = Event::elapsed_time_ms(&start, &stop).expect("elapsed");

    let per_iter_us = (elapsed_ms * 1000.0) / (ITERS as f32);
    eprintln!(
        "[INFO] sparse24 Tier-1 (inflate + naive GEMM) at N={} M={} K={}: {} us/iter (avg over {} iters)",
        N_DIM, M_DIM, K_DIM, per_iter_us, ITERS
    );
    eprintln!(
        "[INFO] Tier-2 target: 1.6× over dense cuBLAS at this shape on sm_89 \
         (via mma.sp.sync or cuSPARSELt backend; not yet implemented)"
    );

    // Tier-1 doesn't enforce a speedup claim — just that the plan
    // completed without error. The "is sparse faster than dense" claim
    // belongs to the Tier-2 follow-up.
    assert!(
        per_iter_us > 0.0,
        "expected positive timing; got {}",
        per_iter_us
    );
}
