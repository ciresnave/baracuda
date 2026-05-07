//! Real-GPU smoke test for grouped GEMM (Phase 4d).
//!
//! Models the Fuel team's headline use case: an MoE expert MLP, where each
//! "group" is one expert with its own token count `M` but shared
//! `(K, N)`. We instantiate four expert-like groups, run the grouped
//! kernel against expert weights, and validate per-group output against
//! a host f32 reference.
//!
//! Marked `#[ignore]` so `cargo test --workspace` is GPU-free by default.
//! Run with:
//!
//! ```text
//! cargo test -p baracuda-cutlass --release --test grouped_gemm_smoke -- --ignored
//! ```

use baracuda_cutlass::{
    EpilogueKind, GroupedGemmPlan, GroupedPlanPreference, GroupedProblem, MatrixMut, MatrixRef,
    Workspace,
};
use baracuda_driver::{init, CaptureMode, Context, Device, DeviceBuffer, Stream};
use half::{bf16, f16};

/// Per-expert: A is row-major `[M, K]`, B is column-major `[K, N]`,
/// D is row-major `[M, N]`.
fn cpu_gemm_rcr_f32(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    d: &mut [f32],
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let a_val = a[i * k + kk]; // row-major A[i, kk]
                let b_val = b[j * k + kk]; // column-major B[kk, j]
                acc += a_val * b_val;
            }
            d[i * n + j] = acc;
        }
    }
}

#[test]
#[ignore]
fn grouped_gemm_f16_moe_4_experts() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    // Four "experts" with shared K and N, variable M (token counts).
    // Sized for an MLP-style up-projection: K = hidden, N = intermediate.
    let k = 64; // hidden dim
    let n = 128; // intermediate dim
    let token_counts = [32usize, 64, 128, 16]; // varying tokens per expert
    let group_count = token_counts.len();

    // Per-expert host data, generated deterministically from the expert
    // index so the CPU reference is reproducible.
    let mut expert_a_f32: Vec<Vec<f32>> = Vec::new();
    let mut expert_b_f32: Vec<Vec<f32>> = Vec::new();
    let mut expert_d_ref: Vec<Vec<f32>> = Vec::new();

    for (e, &m) in token_counts.iter().enumerate() {
        let a: Vec<f32> = (0..(m * k))
            .map(|i| ((i + e * 31) as f32 * 0.01).sin())
            .collect();
        let b: Vec<f32> = (0..(k * n))
            .map(|i| ((i + e * 17) as f32 * 0.013).cos())
            .collect();
        let mut d = vec![0.0f32; m * n];
        cpu_gemm_rcr_f32(m, n, k, &a, &b, &mut d);
        expert_a_f32.push(a);
        expert_b_f32.push(b);
        expert_d_ref.push(d);
    }

    // Upload f16 inputs and zero-initialize device outputs.
    let dev_a: Vec<DeviceBuffer<f16>> = expert_a_f32
        .iter()
        .map(|a| {
            let host: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
            DeviceBuffer::from_slice(&ctx, &host).expect("upload A")
        })
        .collect();
    let dev_b: Vec<DeviceBuffer<f16>> = expert_b_f32
        .iter()
        .map(|b| {
            let host: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
            DeviceBuffer::from_slice(&ctx, &host).expect("upload B")
        })
        .collect();
    let mut dev_d: Vec<DeviceBuffer<f16>> = token_counts
        .iter()
        .map(|&m| DeviceBuffer::zeros(&ctx, m * n).expect("alloc D"))
        .collect();

    // Build per-group views. Note: GroupedProblem borrows from the
    // DeviceBuffers, so dev_d must be split into mutable references via
    // iter_mut to satisfy aliasing.
    let mut groups: Vec<GroupedProblem<f16>> = Vec::with_capacity(group_count);
    let mut dev_d_iter = dev_d.iter_mut();
    for (i, &m) in token_counts.iter().enumerate() {
        let m_i32 = m as i32;
        let n_i32 = n as i32;
        let k_i32 = k as i32;
        let d_buf = dev_d_iter.next().unwrap();
        groups.push(GroupedProblem {
            m: m_i32,
            n: n_i32,
            k: k_i32,
            a: MatrixRef {
                data: dev_a[i].as_slice(),
                rows: m_i32,
                cols: k_i32,
                ld: k_i32 as i64,
            },
            b: MatrixRef {
                data: dev_b[i].as_slice(),
                rows: k_i32,
                cols: n_i32,
                ld: k_i32 as i64,
            },
            c: None,
            d: MatrixMut {
                data: d_buf.as_slice_mut(),
                rows: m_i32,
                cols: n_i32,
                ld: n_i32 as i64,
            },
            alpha: 1.0,
            beta: 0.0,
        });
    }

    let plan = GroupedGemmPlan::<f16>::select(
        &stream,
        EpilogueKind::Identity,
        GroupedPlanPreference::default(),
    )
    .expect("plan select");

    let prepared = plan.prepare(&groups).expect("prepare");

    let workspace_bytes = prepared.workspace_size();
    println!(
        "grouped GEMM prepared: {} groups, workspace {} bytes",
        prepared.group_count(),
        workspace_bytes
    );

    let mut workspace: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, workspace_bytes).expect("alloc workspace");

    prepared
        .run(&stream, Workspace::Borrowed(workspace.as_slice_mut()))
        .expect("grouped run");

    // Drop groups before reading dev_d (the groups borrow dev_d mutably).
    drop(groups);

    // Validate each expert's output.
    let tol = (k as f32) * 5e-3;
    for (e, &m) in token_counts.iter().enumerate() {
        let mut host_d_out = vec![f16::ZERO; m * n];
        dev_d[e]
            .copy_to_host(&mut host_d_out)
            .expect("download D");

        let mut max_err = 0.0f32;
        for (got, want) in host_d_out.iter().zip(expert_d_ref[e].iter()) {
            let err = (got.to_f32() - want).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < tol,
            "expert {e} (M={m}, N={n}, K={k}): max abs err {max_err} exceeded tolerance {tol}"
        );
        println!(
            "  expert {e}: M={m} N={n} K={k} max abs err {max_err} (tol {tol}) ✅"
        );
    }
}

#[test]
#[ignore]
fn grouped_gemm_bf16_moe_3_experts() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let k = 32;
    let n = 64;
    let token_counts = [16usize, 48, 8];
    let group_count = token_counts.len();

    let mut expert_a_f32: Vec<Vec<f32>> = Vec::new();
    let mut expert_b_f32: Vec<Vec<f32>> = Vec::new();
    let mut expert_d_ref: Vec<Vec<f32>> = Vec::new();
    for (e, &m) in token_counts.iter().enumerate() {
        let a: Vec<f32> = (0..(m * k))
            .map(|i| ((i + e * 31) as f32 * 0.01).sin())
            .collect();
        let b: Vec<f32> = (0..(k * n))
            .map(|i| ((i + e * 17) as f32 * 0.013).cos())
            .collect();
        let mut d = vec![0.0f32; m * n];
        cpu_gemm_rcr_f32(m, n, k, &a, &b, &mut d);
        expert_a_f32.push(a);
        expert_b_f32.push(b);
        expert_d_ref.push(d);
    }

    let dev_a: Vec<DeviceBuffer<bf16>> = expert_a_f32
        .iter()
        .map(|a| {
            let host: Vec<bf16> = a.iter().map(|&x| bf16::from_f32(x)).collect();
            DeviceBuffer::from_slice(&ctx, &host).expect("upload A")
        })
        .collect();
    let dev_b: Vec<DeviceBuffer<bf16>> = expert_b_f32
        .iter()
        .map(|b| {
            let host: Vec<bf16> = b.iter().map(|&x| bf16::from_f32(x)).collect();
            DeviceBuffer::from_slice(&ctx, &host).expect("upload B")
        })
        .collect();
    let mut dev_d: Vec<DeviceBuffer<bf16>> = token_counts
        .iter()
        .map(|&m| DeviceBuffer::zeros(&ctx, m * n).expect("alloc D"))
        .collect();

    let mut groups: Vec<GroupedProblem<bf16>> = Vec::with_capacity(group_count);
    let mut dev_d_iter = dev_d.iter_mut();
    for (i, &m) in token_counts.iter().enumerate() {
        let m_i32 = m as i32;
        let n_i32 = n as i32;
        let k_i32 = k as i32;
        let d_buf = dev_d_iter.next().unwrap();
        groups.push(GroupedProblem {
            m: m_i32,
            n: n_i32,
            k: k_i32,
            a: MatrixRef {
                data: dev_a[i].as_slice(),
                rows: m_i32,
                cols: k_i32,
                ld: k_i32 as i64,
            },
            b: MatrixRef {
                data: dev_b[i].as_slice(),
                rows: k_i32,
                cols: n_i32,
                ld: k_i32 as i64,
            },
            c: None,
            d: MatrixMut {
                data: d_buf.as_slice_mut(),
                rows: m_i32,
                cols: n_i32,
                ld: n_i32 as i64,
            },
            alpha: 1.0,
            beta: 0.0,
        });
    }

    let plan = GroupedGemmPlan::<bf16>::select(
        &stream,
        EpilogueKind::Identity,
        GroupedPlanPreference::default(),
    )
    .expect("plan select");
    let prepared = plan.prepare(&groups).expect("prepare");

    let workspace_bytes = prepared.workspace_size();
    let mut workspace: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, workspace_bytes).expect("alloc workspace");
    prepared
        .run(&stream, Workspace::Borrowed(workspace.as_slice_mut()))
        .expect("grouped run");

    drop(groups);

    let tol = (k as f32) * 5e-3;
    for (e, &m) in token_counts.iter().enumerate() {
        let mut host_d_out = vec![bf16::ZERO; m * n];
        dev_d[e]
            .copy_to_host(&mut host_d_out)
            .expect("download D");

        let mut max_err = 0.0f32;
        for (got, want) in host_d_out.iter().zip(expert_d_ref[e].iter()) {
            let err = (got.to_f32() - want).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < tol,
            "bf16 expert {e}: max abs err {max_err} exceeded tolerance {tol}"
        );
        println!(
            "  bf16 expert {e}: M={m} N={n} K={k} max abs err {max_err} (tol {tol}) ✅"
        );
    }
}

/// Capture-safety regression: wrap `prepared.run` inside stream capture,
/// instantiate the resulting graph, and replay it twice. Verifies that
/// (a) the metadata H2D upload is async + capturable (only true if the
/// pinned-buffer source is in place), and (b) both replays produce
/// numerically correct output identical to a non-captured run.
#[test]
#[ignore]
fn grouped_gemm_capture_replay() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let k = 32;
    let n = 64;
    let token_counts = [16usize, 32, 8];

    let mut expert_a_f32: Vec<Vec<f32>> = Vec::new();
    let mut expert_b_f32: Vec<Vec<f32>> = Vec::new();
    let mut expert_d_ref: Vec<Vec<f32>> = Vec::new();
    for (e, &m) in token_counts.iter().enumerate() {
        let a: Vec<f32> = (0..(m * k))
            .map(|i| ((i + e * 31) as f32 * 0.01).sin())
            .collect();
        let b: Vec<f32> = (0..(k * n))
            .map(|i| ((i + e * 17) as f32 * 0.013).cos())
            .collect();
        let mut d = vec![0.0f32; m * n];
        cpu_gemm_rcr_f32(m, n, k, &a, &b, &mut d);
        expert_a_f32.push(a);
        expert_b_f32.push(b);
        expert_d_ref.push(d);
    }

    let dev_a: Vec<DeviceBuffer<f16>> = expert_a_f32
        .iter()
        .map(|a| {
            let host: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
            DeviceBuffer::from_slice(&ctx, &host).unwrap()
        })
        .collect();
    let dev_b: Vec<DeviceBuffer<f16>> = expert_b_f32
        .iter()
        .map(|b| {
            let host: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
            DeviceBuffer::from_slice(&ctx, &host).unwrap()
        })
        .collect();
    let mut dev_d: Vec<DeviceBuffer<f16>> = token_counts
        .iter()
        .map(|&m| DeviceBuffer::zeros(&ctx, m * n).unwrap())
        .collect();

    // Build groups in a tight scope so the mutable borrow on dev_d
    // ends as soon as `prepare` returns (`prepared` no longer holds a
    // Rust borrow on dev_d after the refactor — only raw device
    // pointers stored in its pinned buffer).
    let plan = GroupedGemmPlan::<f16>::select(
        &stream,
        EpilogueKind::Identity,
        GroupedPlanPreference::default(),
    )
    .expect("plan select");

    let prepared = {
        let mut groups: Vec<GroupedProblem<f16>> = Vec::with_capacity(token_counts.len());
        let mut dev_d_iter = dev_d.iter_mut();
        for (i, &m) in token_counts.iter().enumerate() {
            let m_i32 = m as i32;
            let n_i32 = n as i32;
            let k_i32 = k as i32;
            let d_buf = dev_d_iter.next().unwrap();
            groups.push(GroupedProblem {
                m: m_i32,
                n: n_i32,
                k: k_i32,
                a: MatrixRef {
                    data: dev_a[i].as_slice(),
                    rows: m_i32,
                    cols: k_i32,
                    ld: k_i32 as i64,
                },
                b: MatrixRef {
                    data: dev_b[i].as_slice(),
                    rows: k_i32,
                    cols: n_i32,
                    ld: k_i32 as i64,
                },
                c: None,
                d: MatrixMut {
                    data: d_buf.as_slice_mut(),
                    rows: m_i32,
                    cols: n_i32,
                    ld: n_i32 as i64,
                },
                alpha: 1.0,
                beta: 0.0,
            });
        }
        plan.prepare(&groups).expect("prepare")
    };
    // After this scope: `groups` is dropped (releasing borrows on dev_d),
    // but `prepared` lives on with raw pointers cached in pinned memory.

    let workspace_bytes = prepared.workspace_size();
    let mut workspace: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, workspace_bytes).expect("alloc workspace");

    // Capture the launch into a graph. Both `prepared` (for the pinned
    // metadata source) and `workspace` / `dev_*` (for the device-side
    // pointers) must outlive any replay of `exec`.
    let graph = stream
        .capture(CaptureMode::ThreadLocal, |s| {
            prepared
                .run(s, Workspace::Borrowed(workspace.as_slice_mut()))
                .expect("grouped run inside capture");
            Ok(())
        })
        .expect("capture failed — likely the metadata H2D wasn't async (pinned source missing?)");

    let exec = graph.instantiate().expect("instantiate graph");

    // Replay twice; each launch should produce the same result.
    for replay_idx in 0..2 {
        for d_buf in dev_d.iter_mut() {
            let zeros = vec![f16::ZERO; d_buf.len()];
            d_buf.copy_from_host(&zeros).unwrap();
        }

        exec.launch(&stream).expect("graph launch");
        stream.synchronize().expect("sync");

        for (e, &m) in token_counts.iter().enumerate() {
            let mut host_d_out = vec![f16::ZERO; m * n];
            dev_d[e].copy_to_host(&mut host_d_out).unwrap();

            let mut max_err = 0.0f32;
            for (got, want) in host_d_out.iter().zip(expert_d_ref[e].iter()) {
                let err = (got.to_f32() - want).abs();
                if err > max_err {
                    max_err = err;
                }
            }
            let tol = (k as f32) * 5e-3;
            assert!(
                max_err < tol,
                "capture replay #{replay_idx}, expert {e}: max abs err {max_err} > tol {tol}"
            );
        }
        println!(
            "capture replay #{replay_idx}: all {} experts match CPU reference ✅",
            token_counts.len()
        );
    }

    // Explicit drop order: exec first (graph references), then prepared
    // (pinned metadata source), then workspace + dev_*.
    drop(exec);
    drop(prepared);
    drop(workspace);
}
