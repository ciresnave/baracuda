//! FlashDecoding — bench the seq_q=1 decode kernel against
//! `FlashSdpaPlan` (bespoke + FA2) at decode-typical shapes.
//!
//! Phase 73 follow-up. The standard `FlashSdpaPlan` is tuned for
//! prefill (`seq_q × seq_k ≥ 1M` triggers FA2; otherwise the bespoke
//! kernel with Br=64 q-tile shape is used). At decode (`seq_q == 1`):
//!
//! - The FA2 heuristic doesn't fire (1 × K is too small).
//! - The bespoke kernel pads one Q row up to 64 and wastes 63/64 of
//!   the q-tile.
//!
//! Both paths under-utilize the GPU. FlashDecoding flips parallelism
//! to the K dimension and saturates SMs even at seq_q = 1.
//!
//! Shapes: B=1, H ∈ {32}, D=128, K ∈ {1024, 2048, 4096, 8192} —
//! covering Llama-2 7B / Mistral 7B / Llama-3 8B decode shapes at
//! various KV-cache lengths.

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    contiguous_stride, FlashDecodingArgs, FlashDecodingDescriptor, FlashDecodingPlan,
    FlashSdpaArgs, FlashSdpaDescriptor, FlashSdpaPlan, PlanPreference, TensorMut, TensorRef,
    Workspace,
};
use baracuda_kernels_bench::{
    append_csv_row, measure_median_ns, setup_device, time_with_events, warmup,
    PhaseTwentyNineRow, PytorchBaseline,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use half::{bf16, f16};

const BENCH_NAME: &str = "flash_decoding";

const BATCH: i32 = 1;
const NUM_HEADS: i32 = 32;
const HEAD_DIM: i32 = 128;
const K_SWEEP: &[i32] = &[1024, 2048, 4096, 8192];

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

fn bench<T>(
    c: &mut Criterion,
    dtype_label: &str,
    fill: T,
    _baseline: Option<&PytorchBaseline>,
) where
    T: baracuda_kernels::Element + Copy + 'static,
{
    let (ctx, stream) = setup_device();

    for &k_len in K_SWEEP {
        let label = format!("B{BATCH}_H{NUM_HEADS}_K{k_len}_D{HEAD_DIM}");

        // ---- FlashDecoding (the new kernel) ----
        let q_numel = (BATCH * NUM_HEADS * HEAD_DIM) as usize;
        let kv_numel = (BATCH * NUM_HEADS * k_len * HEAD_DIM) as usize;
        let host_q: Vec<T> = vec![fill; q_numel];
        let host_kv: Vec<T> = vec![fill; kv_numel];

        let dq = match DeviceBuffer::from_slice(&ctx, &host_q) { Ok(b) => b, Err(_) => continue };
        let dk = match DeviceBuffer::from_slice(&ctx, &host_kv) { Ok(b) => b, Err(_) => continue };
        let dv = match DeviceBuffer::from_slice(&ctx, &host_kv) { Ok(b) => b, Err(_) => continue };
        let mut dy: DeviceBuffer<T> =
            match DeviceBuffer::zeros(&ctx, q_numel) { Ok(b) => b, Err(_) => continue };

        let fd_desc = FlashDecodingDescriptor::new(BATCH, NUM_HEADS, k_len, HEAD_DIM, T::KIND);
        let fd_plan = match FlashDecodingPlan::<T>::select(&stream, &fd_desc, PlanPreference::default()) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("flash_decoding: skipping {label}/{dtype_label}: {e:?}");
                continue;
            }
        };
        let mut fd_ws: DeviceBuffer<u8> =
            match DeviceBuffer::zeros(&ctx, fd_plan.workspace_size()) { Ok(b) => b, Err(_) => continue };

        let sq = [BATCH, NUM_HEADS, HEAD_DIM];
        let sk = [BATCH, NUM_HEADS, k_len, HEAD_DIM];
        let sv = sk;
        let sy = sq;

        warmup(&stream, || {
            let args = FlashDecodingArgs::<T> {
                q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                y: TensorMut { data: dy.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
            };
            fd_plan
                .run(&stream, Workspace::Borrowed(fd_ws.as_slice_mut()), args)
                .expect("fd run");
        });
        let fd_ns = measure_median_ns(&ctx, &stream, 11, 20, || {
            let args = FlashDecodingArgs::<T> {
                q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                y: TensorMut { data: dy.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
            };
            fd_plan
                .run(&stream, Workspace::Borrowed(fd_ws.as_slice_mut()), args)
                .expect("fd run");
        });

        // ---- FlashSdpaPlan reference (bespoke or FA2 depending on
        //      heuristic — at seq_q=1 it's bespoke unless K is huge). ----
        let q4_numel = (BATCH * NUM_HEADS * 1 * HEAD_DIM) as usize;
        let lse_numel = (BATCH * NUM_HEADS * 1) as usize;
        let host_q4: Vec<T> = vec![fill; q4_numel];
        let host_lse: Vec<T> = vec![fill; lse_numel];

        let dq4 = match DeviceBuffer::from_slice(&ctx, &host_q4) { Ok(b) => b, Err(_) => continue };
        let mut dy4: DeviceBuffer<T> =
            match DeviceBuffer::zeros(&ctx, q4_numel) { Ok(b) => b, Err(_) => continue };
        let mut dlse: DeviceBuffer<T> =
            match DeviceBuffer::from_slice(&ctx, &host_lse) { Ok(b) => b, Err(_) => continue };

        let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
        let sdpa_desc = FlashSdpaDescriptor::new(
            BATCH, NUM_HEADS, 1, k_len, HEAD_DIM, HEAD_DIM,
            scale, false, T::KIND,
        );
        let sdpa_plan_opt = FlashSdpaPlan::<T>::select(&stream, &sdpa_desc, PlanPreference::default()).ok();
        let sdpa_ns: f64 = if let Some(sdpa_plan) = sdpa_plan_opt {
            let mut sdpa_ws: DeviceBuffer<u8> =
                match DeviceBuffer::zeros(&ctx, sdpa_plan.workspace_size()) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
            let sq4 = [BATCH, NUM_HEADS, 1, HEAD_DIM];
            let sk4 = [BATCH, NUM_HEADS, k_len, HEAD_DIM];
            let sv4 = sk4;
            let sy4 = sq4;
            let sl = [BATCH, NUM_HEADS, 1];
            let probe = FlashSdpaArgs::<T> {
                q: TensorRef { data: dq4.as_slice(), shape: sq4, stride: contiguous_stride(sq4) },
                k: TensorRef { data: dk.as_slice(), shape: sk4, stride: contiguous_stride(sk4) },
                v: TensorRef { data: dv.as_slice(), shape: sv4, stride: contiguous_stride(sv4) },
                y: TensorMut { data: dy4.as_slice_mut(), shape: sy4, stride: contiguous_stride(sy4) },
                lse: TensorMut { data: dlse.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
                mask: None,
                alibi_slopes: None,
            };
            if sdpa_plan.can_implement(&probe).is_err() {
                f64::NAN
            } else {
                warmup(&stream, || {
                    let args = FlashSdpaArgs::<T> {
                        q: TensorRef { data: dq4.as_slice(), shape: sq4, stride: contiguous_stride(sq4) },
                        k: TensorRef { data: dk.as_slice(), shape: sk4, stride: contiguous_stride(sk4) },
                        v: TensorRef { data: dv.as_slice(), shape: sv4, stride: contiguous_stride(sv4) },
                        y: TensorMut { data: dy4.as_slice_mut(), shape: sy4, stride: contiguous_stride(sy4) },
                        lse: TensorMut { data: dlse.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
                        mask: None,
                        alibi_slopes: None,
                    };
                    let ws = match sdpa_plan.workspace_size() {
                        0 => Workspace::None,
                        _ => Workspace::Borrowed(sdpa_ws.as_slice_mut()),
                    };
                    sdpa_plan.run(&stream, ws, args).expect("sdpa run");
                });
                measure_median_ns(&ctx, &stream, 11, 20, || {
                    let args = FlashSdpaArgs::<T> {
                        q: TensorRef { data: dq4.as_slice(), shape: sq4, stride: contiguous_stride(sq4) },
                        k: TensorRef { data: dk.as_slice(), shape: sk4, stride: contiguous_stride(sk4) },
                        v: TensorRef { data: dv.as_slice(), shape: sv4, stride: contiguous_stride(sv4) },
                        y: TensorMut { data: dy4.as_slice_mut(), shape: sy4, stride: contiguous_stride(sy4) },
                        lse: TensorMut { data: dlse.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
                        mask: None,
                        alibi_slopes: None,
                    };
                    let ws = match sdpa_plan.workspace_size() {
                        0 => Workspace::None,
                        _ => Workspace::Borrowed(sdpa_ws.as_slice_mut()),
                    };
                    sdpa_plan.run(&stream, ws, args).expect("sdpa run");
                })
            }
        } else {
            f64::NAN
        };

        append_csv_row(
            BENCH_NAME,
            &PhaseTwentyNineRow {
                op: "flash_decoding",
                shape: label.clone(),
                dtype: leak_str(dtype_label),
                baracuda_ns: fd_ns,
                reference_ns: if sdpa_ns.is_nan() { None } else { Some(sdpa_ns) },
                reference: "FlashSdpaPlan",
                pytorch_ns: None,
            },
        );

        // criterion bars for visual diffing.
        let mut group = c.benchmark_group(format!("flash_decoding/{dtype_label}"));
        group.bench_with_input(BenchmarkId::new("baracuda", &label), &(), |bb, _| {
            bb.iter_custom(|iters| {
                time_with_events(&ctx, &stream, iters, || {
                    let args = FlashDecodingArgs::<T> {
                        q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                        k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                        v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                        y: TensorMut { data: dy.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
                    };
                    fd_plan
                        .run(&stream, Workspace::Borrowed(fd_ws.as_slice_mut()), args)
                        .expect("fd run");
                })
            });
        });
        group.finish();
    }
}

/// Top-level criterion entry — invoked by criterion_main!.
fn benches(c: &mut Criterion) {
    let baseline = PytorchBaseline::load_default();
    bench::<f16>(c, "f16", f16::from_f32(0.01), baseline.as_ref());
    bench::<bf16>(c, "bf16", bf16::from_f32(0.01), baseline.as_ref());
}

// `criterion_group!` expands into a `pub fn` whose signature is fixed
// by the macro — can't doc-comment it directly, so suppress the
// workspace `missing_docs = deny` lint on the generated fn.
#[allow(missing_docs)]
mod criterion_glue {
    use super::*;
    criterion_group!(benches_grp, benches);
}
criterion_main!(criterion_glue::benches_grp);
