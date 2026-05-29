//! Phase 42 A/B: FA2 vs baracuda's bespoke Flash SDPA forward.
//!
//! Sweeps four shape regimes to validate the dispatch heuristic:
//!
//! 1. **Decode** (`Sq=1`, `Sk=2048`) — bespoke should win (FA2 has
//!    minimal tile-by-tile reuse to amortize over 1 query row).
//! 2. **Small prefill** (`Sq=Sk=512`) — bespoke tends to win;
//!    long-context heuristic stays off (work = 256K < 1M).
//! 3. **Large prefill** (`Sq=Sk=2048`) — FA2 should win
//!    (work = 4M ≥ 1M; FA2's CUTLASS tiling amortizes weight loads).
//! 4. **Long-context** (`Sq=Sk=4096`) — FA2 should dominate
//!    (work = 16M).
//!
//! All measurements use the same launcher (`FlashSdpaPlan::run`) with
//! `PlanPreference::prefer_backend` forced to each backend in turn.
//! Reports speedup as `bespoke_ns / fa2_ns`; values >1 mean FA2 wins.
//!
//! Only compiles with `--features fa2`. Without it `main` is a no-op
//! so `cargo bench` still links cleanly.

#[cfg(not(feature = "fa2"))]
fn main() {
    eprintln!("sdpa_vs_fa2: built without the `fa2` feature; bench is a no-op.");
}

#[cfg(feature = "fa2")]
fn main() {
    fa2_bench::run();
}

#[cfg(feature = "fa2")]
mod fa2_bench {
    use baracuda_driver::DeviceBuffer;
    use baracuda_kernels::{
        contiguous_stride, BackendKind, ElementKind, FlashSdpaArgs, FlashSdpaDescriptor,
        FlashSdpaPlan, PlanPreference, TensorMut, TensorRef, Workspace,
    };
    use baracuda_kernels_bench::{
        measure_median_ns, setup_device, warmup,
    };
    use half::{bf16, f16};

    /// (batch, num_heads, seq_q, seq_k, head_dim, label).
    const SHAPES: &[(i32, i32, i32, i32, i32, &str)] = &[
        // Decode — bespoke expected to win.
        (1, 32, 1,    2048, 128, "decode_Sq1_Sk2k"),
        // Small prefill — bespoke expected to win (work < heuristic threshold).
        (1, 32, 512,  512,  128, "small_prefill_512x512"),
        // Large prefill — FA2 expected to win.
        (1, 32, 2048, 2048, 128, "large_prefill_2kx2k"),
        // Long-context — FA2 expected to dominate.
        (1, 16, 4096, 4096, 128, "long_ctx_4kx4k"),
    ];

    pub fn run() {
        for &(b, h, sq, sk, d, label) in SHAPES {
            run_one_f16(b, h, sq, sk, d, label);
            run_one_bf16(b, h, sq, sk, d, label);
        }
    }

    fn run_one_f16(b: i32, h: i32, sq: i32, sk: i32, d: i32, label: &str) {
        run_one::<f16>(b, h, sq, sk, d, label, "f16", f16::from_f32(0.123));
    }
    fn run_one_bf16(b: i32, h: i32, sq: i32, sk: i32, d: i32, label: &str) {
        run_one::<bf16>(b, h, sq, sk, d, label, "bf16", bf16::from_f32(0.123));
    }

    fn run_one<T>(
        b: i32, h: i32, sq: i32, sk: i32, d: i32,
        label: &str, dtype_label: &str, fill: T,
    ) where
        T: baracuda_kernels::Element + Copy + 'static,
    {
        let (ctx, stream) = setup_device();

        let q_n = (b * h * sq * d) as usize;
        let k_n = (b * h * sk * d) as usize;
        let v_n = (b * h * sk * d) as usize;
        let y_n = (b * h * sq * d) as usize;
        let lse_n = (b * h * sq) as usize;

        let host_q: Vec<T> = vec![fill; q_n];
        let host_k: Vec<T> = vec![fill; k_n];
        let host_v: Vec<T> = vec![fill; v_n];

        let dq = DeviceBuffer::from_slice(&ctx, &host_q).expect("up q");
        let dk = DeviceBuffer::from_slice(&ctx, &host_k).expect("up k");
        let dv = DeviceBuffer::from_slice(&ctx, &host_v).expect("up v");
        let mut dy: DeviceBuffer<T> = DeviceBuffer::zeros(&ctx, y_n).expect("alloc y");
        let mut dlse: DeviceBuffer<T> = DeviceBuffer::zeros(&ctx, lse_n).expect("alloc lse");

        let sq_shape = [b, h, sq, d];
        let sk_shape = [b, h, sk, d];
        let sv_shape = [b, h, sk, d];
        let sy_shape = [b, h, sq, d];
        let sl_shape = [b, h, sq];
        let scale = 1.0_f32 / (d as f32).sqrt();

        let desc = FlashSdpaDescriptor::new(
            b,
            h,
            sq,
            sk,
            d,
            d,
            scale,
            false,
            T::KIND,
        );

        // Bespoke backend.
        let pref_b = PlanPreference {
            prefer_backend: Some(BackendKind::Bespoke),
            ..Default::default()
        };
        let plan_b = match FlashSdpaPlan::<T>::select(&stream, &desc, pref_b) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{label}/{dtype_label}: bespoke select error: {e:?}");
                return;
            }
        };
        assert_eq!(plan_b.backend(), BackendKind::Bespoke);

        // FA2 backend (Tier-1: d==128, dtype ∈ {f16,bf16}, num_heads_k == num_heads).
        let pref_f = PlanPreference {
            prefer_backend: Some(BackendKind::FlashAttentionV2),
            ..Default::default()
        };
        let plan_f = match FlashSdpaPlan::<T>::select(&stream, &desc, pref_f) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{label}/{dtype_label}: fa2 select error: {e:?}");
                return;
            }
        };
        let backend_f = plan_f.backend();

        let st_q = contiguous_stride(sq_shape);
        let st_k = contiguous_stride(sk_shape);
        let st_v = contiguous_stride(sv_shape);
        let st_y = contiguous_stride(sy_shape);
        let st_l = contiguous_stride(sl_shape);

        // FA2 workspace (allocated once; reused across all measurements
        // since FA2 writes are overwriting, not accumulating).
        let ws_bytes = plan_f.workspace_size();
        let mut ws_buf: DeviceBuffer<u8> =
            DeviceBuffer::zeros(&ctx, ws_bytes.max(1)).expect("alloc fa2 ws");

        // Warm bespoke.
        warmup(&stream, || {
            let args = FlashSdpaArgs::<T> {
                q: TensorRef { data: dq.as_slice(), shape: sq_shape, stride: st_q },
                k: TensorRef { data: dk.as_slice(), shape: sk_shape, stride: st_k },
                v: TensorRef { data: dv.as_slice(), shape: sv_shape, stride: st_v },
                y: TensorMut { data: dy.as_slice_mut(), shape: sy_shape, stride: st_y },
                lse: TensorMut { data: dlse.as_slice_mut(), shape: sl_shape, stride: st_l },
                            mask: None,
                            alibi_slopes: None,
            };
            plan_b.run(&stream, Workspace::None, args).expect("bespoke warm");
        });

        if matches!(backend_f, BackendKind::FlashAttentionV2) {
            warmup(&stream, || {
                let args = FlashSdpaArgs::<T> {
                    q: TensorRef { data: dq.as_slice(), shape: sq_shape, stride: st_q },
                    k: TensorRef { data: dk.as_slice(), shape: sk_shape, stride: st_k },
                    v: TensorRef { data: dv.as_slice(), shape: sv_shape, stride: st_v },
                    y: TensorMut { data: dy.as_slice_mut(), shape: sy_shape, stride: st_y },
                    lse: TensorMut { data: dlse.as_slice_mut(), shape: sl_shape, stride: st_l },
                                    mask: None,
                                    alibi_slopes: None,
                };
                plan_f
                    .run(&stream, Workspace::Borrowed(ws_buf.as_slice_mut()), args)
                    .expect("fa2 warm");
            });
        }

        // Measure bespoke.
        let bespoke_ns = measure_median_ns(&ctx, &stream, 9, 20, || {
            let args = FlashSdpaArgs::<T> {
                q: TensorRef { data: dq.as_slice(), shape: sq_shape, stride: st_q },
                k: TensorRef { data: dk.as_slice(), shape: sk_shape, stride: st_k },
                v: TensorRef { data: dv.as_slice(), shape: sv_shape, stride: st_v },
                y: TensorMut { data: dy.as_slice_mut(), shape: sy_shape, stride: st_y },
                lse: TensorMut { data: dlse.as_slice_mut(), shape: sl_shape, stride: st_l },
                            mask: None,
                            alibi_slopes: None,
            };
            plan_b.run(&stream, Workspace::None, args).expect("bespoke");
        });
        // Measure FA2 (or 0 if heuristic / select put it on bespoke).
        let fa2_ns = if matches!(backend_f, BackendKind::FlashAttentionV2) {
            measure_median_ns(&ctx, &stream, 9, 20, || {
                let args = FlashSdpaArgs::<T> {
                    q: TensorRef { data: dq.as_slice(), shape: sq_shape, stride: st_q },
                    k: TensorRef { data: dk.as_slice(), shape: sk_shape, stride: st_k },
                    v: TensorRef { data: dv.as_slice(), shape: sv_shape, stride: st_v },
                    y: TensorMut { data: dy.as_slice_mut(), shape: sy_shape, stride: st_y },
                    lse: TensorMut { data: dlse.as_slice_mut(), shape: sl_shape, stride: st_l },
                                    mask: None,
                                    alibi_slopes: None,
                };
                plan_f
                    .run(&stream, Workspace::Borrowed(ws_buf.as_slice_mut()), args)
                    .expect("fa2");
            })
        } else {
            0.0
        };

        let speedup = if fa2_ns > 0.0 {
            bespoke_ns / fa2_ns
        } else {
            0.0
        };
        let heuristic_picks_fa2 =
            should_use_fa2_check(&desc, T::KIND, sq as i64, sk as i64);
        println!(
            "[sdpa_vs_fa2] {label:<24} {dtype_label}  bespoke_us={:>7.1}  fa2_us={:>7.1}  speedup_fa2={:.2}x  heuristic={}",
            bespoke_ns / 1000.0,
            fa2_ns / 1000.0,
            speedup,
            if heuristic_picks_fa2 { "fa2" } else { "bespoke" },
        );
    }

    fn should_use_fa2_check(
        desc: &FlashSdpaDescriptor,
        elem: ElementKind,
        sq: i64,
        sk: i64,
    ) -> bool {
        desc.d_k == 128
            && desc.d_v == 128
            && matches!(elem, ElementKind::F16 | ElementKind::Bf16)
            && sq * sk >= 1024 * 1024
    }
}
