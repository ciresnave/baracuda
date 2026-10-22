//! Marlin W4A16 GEMM vs a dequantise-then-GEMM reference at real shapes.
//!
//! The reference is built from the UNPACKED weights (the raw 4-bit values and
//! f16 scales the test generated), never from the packed buffer the kernel
//! reads, so it is independent of `gptq_to_marlin_repack` and the kernel alike:
//! a wrong permutation in either shows up as a mismatch, not as agreement.
//!
//! Each shape also runs a known-bad control: one scale is corrupted on the
//! device only, and the SAME comparison must then fail in exactly that column
//! (otherwise the comparison cannot tell a wrong answer from a right one).
//!
//! Device test (feature `device-tests`, not `#[ignore]`): run via
//! `cargo gpu-test -p baracuda-kernels --features sm89,marlin,device-tests --test marlin_reference`
//! (takes the machine-wide `gpu-run` lock); the self-hosted runner runs it too.

#![cfg(all(feature = "marlin", feature = "device-tests"))]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init, require};
use baracuda_kernels::{
    GptqWeights, Int4MarlinGemmArgs, Int4MarlinGemmDescriptor, Int4MarlinGemmPlan, PlanPreference,
    TensorMut, TensorRef, Workspace, contiguous_stride, gptq_to_marlin_repack,
};
use half::f16;

/// Deterministic xorshift so a failure reproduces from the shape alone.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

struct Case {
    a: Vec<f16>,      // [M, K]
    q: Vec<u8>,       // [K, N] values 0..=15 (dequant = scale * (q - 8))
    scales: Vec<f16>, // [K/G, N]
}

fn make_case(m: usize, n: usize, k: usize, group: usize) -> Case {
    let mut rng = Rng(0x9E3779B97F4A7C15 ^ ((m * 31 + n) * 31 + k) as u64);
    let a = (0..m * k)
        .map(|_| f16::from_f32(rng.unit() * 2.0 - 1.0))
        .collect();
    let q = (0..k * n).map(|_| (rng.next() & 0xF) as u8).collect();
    let scales = (0..(k / group) * n)
        .map(|_| f16::from_f32(0.01 + rng.unit() * 0.04))
        .collect();
    Case { a, q, scales }
}

/// GPTQ container for `case`: qweight `[K/8, N]` (8 nibbles along K), all
/// zero-points 8 so the symmetric Marlin form represents it exactly.
fn gptq_of(case: &Case, n: usize, k: usize, group: usize) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
    let mut qweight = vec![0i32; (k / 8) * n];
    for kb in 0..k / 8 {
        for col in 0..n {
            let mut w = 0u32;
            for nib in 0..8 {
                w |= (case.q[(kb * 8 + nib) * n + col] as u32) << (4 * nib);
            }
            qweight[kb * n + col] = w as i32;
        }
    }
    let scales = case.scales.iter().map(|s| s.to_f32()).collect();
    let qzeros = vec![0x88888888_u32 as i32; (k / group) * (n / 8)];
    (qweight, scales, qzeros)
}

/// `(reference, abs_sum)` per output cell, f64 accumulation over the f16
/// activation and the f16-scaled dequantised weight.
fn reference(case: &Case, m: usize, n: usize, k: usize, group: usize) -> (Vec<f64>, Vec<f64>) {
    let mut out = vec![0f64; m * n];
    let mut abs = vec![0f64; m * n];
    for row in 0..m {
        for col in 0..n {
            let (mut acc, mut asum) = (0f64, 0f64);
            for kk in 0..k {
                let w = case.scales[(kk / group) * n + col].to_f64()
                    * (case.q[kk * n + col] as f64 - 8.0);
                let term = case.a[row * k + kk].to_f64() * w;
                acc += term;
                asum += term.abs();
            }
            out[row * n + col] = acc;
            abs[row * n + col] = asum;
        }
    }
    (out, abs)
}

/// Run the kernel on `scales_dev` (already Marlin-permuted) and return `[M, N]`.
fn run_kernel(
    ctx: &Context,
    stream: &Stream,
    case: &Case,
    packed: &[i32],
    scales_dev: &[f16],
    (m, n, k, group): (usize, usize, usize, usize),
) -> Vec<f16> {
    let max_par = 16i32;
    let dev_a = DeviceBuffer::from_slice(ctx, &case.a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(ctx, packed).expect("upload B");
    let dev_s = DeviceBuffer::from_slice(ctx, scales_dev).expect("upload scales");
    let ws_len = (n / 128) * max_par as usize;
    let mut dev_ws = DeviceBuffer::from_slice(ctx, &vec![0i32; ws_len]).expect("upload ws");
    let mut dev_c: DeviceBuffer<f16> = DeviceBuffer::zeros(ctx, m * n).expect("alloc C");
    let (mi, ni, ki) = (m as i32, n as i32, k as i32);
    let desc = Int4MarlinGemmDescriptor::new(mi, ni, ki)
        .with_group_size(group as i32)
        .with_max_par(max_par);
    let plan = Int4MarlinGemmPlan::<f16>::select(stream, &desc, PlanPreference::default())
        .expect("plan select");
    let args = Int4MarlinGemmArgs::<f16> {
        activation: TensorRef::new(dev_a.as_slice(), [mi, ki], contiguous_stride([mi, ki]))
            .unwrap(),
        weight_packed: TensorRef::new(dev_b.as_slice(), [packed.len() as i32], [1]).unwrap(),
        scales: TensorRef::new(
            dev_s.as_slice(),
            [ki / group as i32, ni],
            contiguous_stride([ki / group as i32, ni]),
        )
        .unwrap(),
        workspace: TensorMut::new(dev_ws.as_slice_mut(), [ws_len as i32], [1]).unwrap(),
        output: TensorMut::new(dev_c.as_slice_mut(), [mi, ni], contiguous_stride([mi, ni]))
            .unwrap(),
    };
    plan.run(stream, Workspace::None, args).expect("marlin run");
    stream.synchronize().expect("sync");
    let mut host = vec![f16::ZERO; m * n];
    dev_c.copy_to_host(&mut host).expect("download C");
    host
}

/// Cells whose |got - ref| exceeds 1% of the cell's own sum of |terms| (+ an
/// f16 rounding floor), as `(row, col)`. The abs-sum scale makes the bound
/// independent of cancellation in the signed sum.
fn mismatches(got: &[f16], want: &[f64], abs: &[f64], n: usize) -> Vec<(usize, usize)> {
    (0..got.len())
        .filter(|&i| {
            let g = got[i].to_f64();
            !g.is_finite() || (g - want[i]).abs() > 0.01 * abs[i] + 1e-3
        })
        .map(|i| (i / n, i % n))
        .collect()
}

fn check_shape(m: usize, n: usize, k: usize, group: usize) {
    init().expect("driver init");
    let dev = require!(Device::get(0), "a CUDA device (Marlin reference test)");
    let ctx = Context::new(&dev).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let mut case = make_case(m, n, k, group);
    // Tag one group-0 scale with a value outside the generated range so its
    // permuted slot can be found by value, without re-deriving the permutation.
    let probe_col = n / 2 + 3;
    let probe_val = f16::from_f32(0.0777);
    case.scales[probe_col] = probe_val;
    let (qweight, scales_f32, qzeros) = gptq_of(&case, n, k, group);
    let marlin = gptq_to_marlin_repack(&GptqWeights {
        qweight: &qweight,
        scales: &scales_f32,
        qzeros: &qzeros,
        g_idx: None,
        ic: k,
        oc: n,
        group_size: group,
    })
    .expect("repack");
    let scales_dev: Vec<f16> = marlin.scales.iter().map(|&s| f16::from_f32(s)).collect();
    let (want, abs) = reference(&case, m, n, k, group);
    let cfg = (m, n, k, group);

    // Real run: must match everywhere.
    let got = run_kernel(
        &ctx,
        &stream,
        &case,
        &marlin.weight_packed,
        &scales_dev,
        cfg,
    );
    let bad = mismatches(&got, &want, &abs, n);
    assert!(
        bad.is_empty(),
        "M={m} N={n} K={k} G={group}: {} of {} cells mismatch the dequant+GEMM reference; first {:?}",
        bad.len(),
        m * n,
        &bad[..bad.len().min(8)]
    );

    // Known-bad control: scale the tagged device-side scale by 16 (a x2 error in one of 32 groups
    // can hide inside the 1% bound through sign cancellation). The reference is
    // unchanged, so the comparison must now fail, and only in the tagged
    // column (group 0 is the only group holding that scale).
    let slot = scales_dev
        .iter()
        .position(|&s| s == probe_val)
        .expect("tagged scale present exactly once after repack");
    assert_eq!(scales_dev.iter().filter(|&&s| s == probe_val).count(), 1);
    let mut corrupted = scales_dev.clone();
    corrupted[slot] = f16::from_f32(corrupted[slot].to_f32() * 16.0);
    let got_bad = run_kernel(&ctx, &stream, &case, &marlin.weight_packed, &corrupted, cfg);
    let bad_cells = mismatches(&got_bad, &want, &abs, n);
    assert!(
        !bad_cells.is_empty(),
        "M={m} N={n} K={k} G={group}: corrupting a scale did NOT change the result beyond tolerance; \
         the comparison cannot detect a wrong answer"
    );
    assert!(
        bad_cells.iter().all(|&(_, c)| c == probe_col),
        "M={m} N={n} K={k} G={group}: corruption leaked outside column {probe_col}: {:?}",
        &bad_cells[..bad_cells.len().min(8)]
    );
}

#[test]
fn marlin_matches_dequant_gemm_m1() {
    check_shape(1, 4096, 4096, 128);
}

#[test]
fn marlin_matches_dequant_gemm_m16() {
    check_shape(16, 4096, 4096, 128);
}

#[test]
fn marlin_matches_dequant_gemm_small_k_wide_n() {
    check_shape(16, 1024, 1024, 128);
}
