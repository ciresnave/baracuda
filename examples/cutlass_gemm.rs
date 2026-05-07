//! `cutlass_gemm` — end-to-end CUTLASS GEMM via [`baracuda_cutlass`].
//!
//! Runs both a single GEMM (`f16` 256×256×128) and a 4-group MoE-shaped
//! grouped GEMM (variable M per expert, shared K/N) on device 0,
//! validates each against a host f32 reference, and reports max abs
//! error per launch.
//!
//! Run with:
//!
//! ```text
//! cargo run -p baracuda-examples --bin cutlass_gemm --features cutlass-smoke --release
//! ```
//!
//! Requires CUDA 12.x at build time (CUTLASS 4.x is the default; opt down
//! via the `cutlass-2-11` feature on `baracuda-cutlass` if you need a
//! CUDA 11.4-compatible build) and an Ampere/Ada/Hopper-class NVIDIA GPU
//! at runtime.

use baracuda_cutlass::{
    EpilogueKind, GemmArgs, GemmDescriptor, GemmPlan, GroupedGemmPlan, GroupedPlanPreference,
    GroupedProblem, LayoutSku, MatrixMut, MatrixRef, PlanPreference, Workspace,
};
use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use half::f16;

fn cpu_gemm_rcr(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], d: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[j * k + kk];
            }
            d[i * n + j] = acc;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init()?;
    let device = Device::get(0)?;
    println!("cutlass_gemm: device 0 = {}", device.name()?);
    let ctx = Context::new(&device)?;
    let stream = Stream::new(&ctx)?;

    // ---------- single GEMM ----------
    {
        let m = 256i32;
        let n = 256i32;
        let k = 128i32;

        let host_a_f32: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.01).sin()).collect();
        let host_b_f32: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.013).cos()).collect();
        let mut host_d_ref = vec![0.0f32; (m * n) as usize];
        cpu_gemm_rcr(
            m as usize,
            n as usize,
            k as usize,
            &host_a_f32,
            &host_b_f32,
            &mut host_d_ref,
        );

        let host_a: Vec<f16> = host_a_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let host_b: Vec<f16> = host_b_f32.iter().map(|&x| f16::from_f32(x)).collect();

        let dev_a = DeviceBuffer::from_slice(&ctx, &host_a)?;
        let dev_b = DeviceBuffer::from_slice(&ctx, &host_b)?;
        let mut dev_d: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (m * n) as usize)?;

        let desc = GemmDescriptor {
            m,
            n,
            k,
            layout: LayoutSku::Rcr,
            epilogue: EpilogueKind::Identity,
        };
        let plan = GemmPlan::<f16>::select(&stream, &desc, PlanPreference::default())?;
        let args = GemmArgs::<f16> {
            a: MatrixRef {
                data: dev_a.as_slice(),
                rows: m,
                cols: k,
                ld: k as i64,
            },
            b: MatrixRef {
                data: dev_b.as_slice(),
                rows: k,
                cols: n,
                ld: k as i64,
            },
            c: None,
            d: MatrixMut {
                data: dev_d.as_slice_mut(),
                rows: m,
                cols: n,
                ld: n as i64,
            },
            alpha: 1.0,
            beta: 0.0,
        };

        plan.can_implement(&args)?;
        plan.run(&stream, Workspace::None, args)?;

        let mut host_d_out = vec![f16::ZERO; (m * n) as usize];
        dev_d.copy_to_host(&mut host_d_out)?;

        let max_err = host_d_out
            .iter()
            .zip(host_d_ref.iter())
            .map(|(g, w)| (g.to_f32() - w).abs())
            .fold(0.0f32, f32::max);
        let tol = (k as f32) * 5e-3;
        assert!(
            max_err < tol,
            "single GEMM max abs err {max_err} exceeded tolerance {tol}"
        );
        println!("  single GEMM f16 RCR sm_80 ({m}x{n}x{k}): max abs err {max_err} (tol {tol}) ✅");
    }

    // ---------- grouped GEMM (4 expert-shaped problems) ----------
    {
        let k = 64;
        let n = 128;
        let token_counts = [32usize, 64, 128, 16];

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
            cpu_gemm_rcr(m, n, k, &a, &b, &mut d);
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

        let plan = GroupedGemmPlan::<f16>::select(
            &stream,
            EpilogueKind::Identity,
            GroupedPlanPreference::default(),
        )?;

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
            plan.prepare(&groups)?
        };

        let mut workspace: DeviceBuffer<u8> =
            DeviceBuffer::zeros(&ctx, prepared.workspace_size())?;
        prepared.run(&stream, Workspace::Borrowed(workspace.as_slice_mut()))?;

        let tol = (k as f32) * 5e-3;
        for (e, &m) in token_counts.iter().enumerate() {
            let mut host_d_out = vec![f16::ZERO; m * n];
            dev_d[e].copy_to_host(&mut host_d_out)?;
            let max_err = host_d_out
                .iter()
                .zip(expert_d_ref[e].iter())
                .map(|(g, w)| (g.to_f32() - w).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < tol,
                "grouped expert {e}: max abs err {max_err} > tol {tol}"
            );
            println!(
                "  grouped GEMM f16 RCR sm_80 expert {e} (M={m}, N={n}, K={k}): max abs err {max_err} (tol {tol}) ✅"
            );
        }

        println!(
            "cutlass_gemm: ✅ pass — single GEMM + {}-group grouped GEMM both match CPU reference",
            token_counts.len()
        );
    }

    Ok(())
}
