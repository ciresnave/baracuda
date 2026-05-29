//! Flash Attention throughput bench — TFLOPS-equivalent across LLM-
//! typical shapes × dtypes.
//!
//! Sweep:
//! - `B = 1`, `H ∈ {8, 16, 32}`, `Q = K ∈ {512, 1024, 2048, 4096}`,
//!   `D ∈ {64, 128}`.
//! - Dtypes: `f32` / `f16` / `bf16`.
//!
//! Reports `Throughput::Elements(flops)` where `flops ≈ 4·B·H·Q·K·D`
//! (the two GEMMs in `softmax(Q·K^T)·V`). Divide criterion's printed
//! `elem/sec` by `1e12` for TFLOPS-equivalent.

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    contiguous_stride, ElementKind, FlashSdpaArgs, FlashSdpaDescriptor, FlashSdpaPlan,
    PlanPreference, TensorMut, TensorRef, Workspace,
};
use baracuda_kernels_bench::{
    flash_flops, setup_device, time_with_events, warmup, FLASH_B, FLASH_D_SWEEP, FLASH_H_SWEEP,
    FLASH_QK_SWEEP,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use half::{bf16, f16};

/// Generic Flash SDPA bench body. Takes `fill` for the QKV buffers
/// (`1.0`-valued in dtype-appropriate units).
fn bench_flash<T>(c: &mut Criterion, dtype_label: &str, _kind: ElementKind, fill: T)
where
    T: baracuda_kernels::Element + Copy + 'static,
{
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group(format!("flash_attention/{dtype_label}"));

    for &d in FLASH_D_SWEEP {
        for &h in FLASH_H_SWEEP {
            for &qk in FLASH_QK_SWEEP {
                let b = FLASH_B;
                let q = qk;
                let k = qk;
                let shape = format!("B{b}_H{h}_Q{q}_K{k}_D{d}");
                let scale = 1.0_f32 / (d as f32).sqrt();

                let qkv_dk_numel = (b * h * q * d) as usize;
                let kv_numel = (b * h * k * d) as usize;
                let y_numel = (b * h * q * d) as usize;
                let lse_numel = (b * h * q) as usize;

                let host_q: Vec<T> = vec![fill; qkv_dk_numel];
                let host_k: Vec<T> = vec![fill; kv_numel];
                let host_v: Vec<T> = vec![fill; kv_numel];

                let dq = match DeviceBuffer::from_slice(&ctx, &host_q) {
                    Ok(x) => x,
                    Err(_) => continue,
                };
                let dk = match DeviceBuffer::from_slice(&ctx, &host_k) {
                    Ok(x) => x,
                    Err(_) => continue,
                };
                let dv = match DeviceBuffer::from_slice(&ctx, &host_v) {
                    Ok(x) => x,
                    Err(_) => continue,
                };
                let mut dy: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, y_numel) {
                    Ok(x) => x,
                    Err(_) => continue,
                };
                let mut dlse: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, lse_numel) {
                    Ok(x) => x,
                    Err(_) => continue,
                };

                let desc = FlashSdpaDescriptor::new(
                    b,
                    h,
                    q,
                    k,
                    d,
                    d,
                    scale,
                    false,
                    T::KIND,
                );
                let plan = match FlashSdpaPlan::<T>::select(
                    &stream,
                    &desc,
                    PlanPreference::default(),
                ) {
                    Ok(p) => p,
                    Err(_) => continue,
                };

                let sq = [b, h, q, d];
                let sk = [b, h, k, d];
                let sv = [b, h, k, d];
                let sy = [b, h, q, d];
                let sl = [b, h, q];
                let stq = contiguous_stride(sq);
                let stk = contiguous_stride(sk);
                let stv = contiguous_stride(sv);
                let sty = contiguous_stride(sy);
                let stl = contiguous_stride(sl);

                group.throughput(Throughput::Elements(flash_flops(b, h, q, k, d)));
                group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
                    warmup(&stream, || {
                        let args = FlashSdpaArgs::<T> {
                            q: TensorRef {
                                data: dq.as_slice(),
                                shape: sq,
                                stride: stq,
                            },
                            k: TensorRef {
                                data: dk.as_slice(),
                                shape: sk,
                                stride: stk,
                            },
                            v: TensorRef {
                                data: dv.as_slice(),
                                shape: sv,
                                stride: stv,
                            },
                            y: TensorMut {
                                data: dy.as_slice_mut(),
                                shape: sy,
                                stride: sty,
                            },
                            lse: TensorMut {
                                data: dlse.as_slice_mut(),
                                shape: sl,
                                stride: stl,
                            },
                                                    mask: None,
                                                    alibi_slopes: None,
                        };
                        plan.run(&stream, Workspace::None, args)
                            .expect("flash sdpa warmup run");
                    });

                    bb.iter_custom(|iters| {
                        time_with_events(&ctx, &stream, iters, || {
                            let args = FlashSdpaArgs::<T> {
                                q: TensorRef {
                                    data: dq.as_slice(),
                                    shape: sq,
                                    stride: stq,
                                },
                                k: TensorRef {
                                    data: dk.as_slice(),
                                    shape: sk,
                                    stride: stk,
                                },
                                v: TensorRef {
                                    data: dv.as_slice(),
                                    shape: sv,
                                    stride: stv,
                                },
                                y: TensorMut {
                                    data: dy.as_slice_mut(),
                                    shape: sy,
                                    stride: sty,
                                },
                                lse: TensorMut {
                                    data: dlse.as_slice_mut(),
                                    shape: sl,
                                    stride: stl,
                                },
                                                            mask: None,
                                                            alibi_slopes: None,
                            };
                            plan.run(&stream, Workspace::None, args).expect("flash sdpa run");
                        })
                    });
                });
            }
        }
    }
    group.finish();
}

fn flash_benches(c: &mut Criterion) {
    bench_flash::<f32>(c, "f32", ElementKind::F32, 1.0_f32);
    bench_flash::<f16>(c, "f16", ElementKind::F16, f16::ONE);
    bench_flash::<bf16>(c, "bf16", ElementKind::Bf16, bf16::ONE);

    // Phase 10 Milestone 10.3 — sm_89-specialized Flash Attention FW
    // bench group. Same shape sweep as the baseline so criterion's
    // BenchmarkId.from_parameter strings line up between the two groups
    // ("flash_attention/{dtype}" vs "flash_attention_sm89/{dtype}"),
    // making side-by-side speedup comparison trivial. f16 / bf16 only —
    // f32 / f64 stay on the sm_80 baseline.
    #[cfg(feature = "sm89")]
    {
        bench_flash_sm89::<f16>(c, "f16", ElementKind::F16, f16::ONE);
        bench_flash_sm89::<bf16>(c, "bf16", ElementKind::Bf16, bf16::ONE);
    }
}

/// Sibling of `bench_flash` that routes through `FlashSdpaSm89Plan`.
/// Behind `#[cfg(feature = "sm89")]` so the bench binary still builds on
/// non-sm89 configurations.
#[cfg(feature = "sm89")]
fn bench_flash_sm89<T>(c: &mut Criterion, dtype_label: &str, _kind: ElementKind, fill: T)
where
    T: baracuda_kernels::Element + Copy + 'static,
{
    use baracuda_kernels::{FlashSdpaSm89Args, FlashSdpaSm89Descriptor, FlashSdpaSm89Plan};

    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group(format!("flash_attention_sm89/{dtype_label}"));

    for &d in FLASH_D_SWEEP {
        for &h in FLASH_H_SWEEP {
            for &qk in FLASH_QK_SWEEP {
                let b = FLASH_B;
                let q = qk;
                let k = qk;
                let shape = format!("B{b}_H{h}_Q{q}_K{k}_D{d}");
                let scale = 1.0_f32 / (d as f32).sqrt();

                let qkv_dk_numel = (b * h * q * d) as usize;
                let kv_numel = (b * h * k * d) as usize;
                let y_numel = (b * h * q * d) as usize;
                let lse_numel = (b * h * q) as usize;

                let host_q: Vec<T> = vec![fill; qkv_dk_numel];
                let host_k: Vec<T> = vec![fill; kv_numel];
                let host_v: Vec<T> = vec![fill; kv_numel];

                let dq = match DeviceBuffer::from_slice(&ctx, &host_q) {
                    Ok(x) => x,
                    Err(_) => continue,
                };
                let dk = match DeviceBuffer::from_slice(&ctx, &host_k) {
                    Ok(x) => x,
                    Err(_) => continue,
                };
                let dv = match DeviceBuffer::from_slice(&ctx, &host_v) {
                    Ok(x) => x,
                    Err(_) => continue,
                };
                let mut dy: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, y_numel) {
                    Ok(x) => x,
                    Err(_) => continue,
                };
                let mut dlse: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, lse_numel) {
                    Ok(x) => x,
                    Err(_) => continue,
                };

                let desc = FlashSdpaSm89Descriptor {
                    batch_size: b,
                    num_heads: h,
                    query_len: q,
                    key_len: k,
                    d_k: d,
                    d_v: d,
                    scale,
                    is_causal: false,
                    element: T::KIND,
                };
                let plan = match FlashSdpaSm89Plan::<T>::select(
                    &stream,
                    &desc,
                    PlanPreference::default(),
                ) {
                    Ok(p) => p,
                    Err(_) => continue,
                };

                let sq = [b, h, q, d];
                let sk = [b, h, k, d];
                let sv = [b, h, k, d];
                let sy = [b, h, q, d];
                let sl = [b, h, q];
                let stq = contiguous_stride(sq);
                let stk = contiguous_stride(sk);
                let stv = contiguous_stride(sv);
                let sty = contiguous_stride(sy);
                let stl = contiguous_stride(sl);

                group.throughput(Throughput::Elements(flash_flops(b, h, q, k, d)));
                group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
                    warmup(&stream, || {
                        let args = FlashSdpaSm89Args::<T> {
                            q: TensorRef {
                                data: dq.as_slice(),
                                shape: sq,
                                stride: stq,
                            },
                            k: TensorRef {
                                data: dk.as_slice(),
                                shape: sk,
                                stride: stk,
                            },
                            v: TensorRef {
                                data: dv.as_slice(),
                                shape: sv,
                                stride: stv,
                            },
                            y: TensorMut {
                                data: dy.as_slice_mut(),
                                shape: sy,
                                stride: sty,
                            },
                            lse: TensorMut {
                                data: dlse.as_slice_mut(),
                                shape: sl,
                                stride: stl,
                            },
                        };
                        plan.run(&stream, Workspace::None, args)
                            .expect("flash sdpa sm89 warmup run");
                    });

                    bb.iter_custom(|iters| {
                        time_with_events(&ctx, &stream, iters, || {
                            let args = FlashSdpaSm89Args::<T> {
                                q: TensorRef {
                                    data: dq.as_slice(),
                                    shape: sq,
                                    stride: stq,
                                },
                                k: TensorRef {
                                    data: dk.as_slice(),
                                    shape: sk,
                                    stride: stk,
                                },
                                v: TensorRef {
                                    data: dv.as_slice(),
                                    shape: sv,
                                    stride: stv,
                                },
                                y: TensorMut {
                                    data: dy.as_slice_mut(),
                                    shape: sy,
                                    stride: sty,
                                },
                                lse: TensorMut {
                                    data: dlse.as_slice_mut(),
                                    shape: sl,
                                    stride: stl,
                                },
                            };
                            plan.run(&stream, Workspace::None, args)
                                .expect("flash sdpa sm89 run");
                        })
                    });
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, flash_benches);
criterion_main!(benches);
