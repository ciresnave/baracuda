//! Real-GPU smoke test for the bespoke FP8 E4M3 RRR Identity kernel in
//! `baracuda-kernels-sys`.
//!
//! Mirror of `fp8_e4m3_rcr_smoke.rs` for the RRR layout: B is row-major
//! `[K, N]` (indexed `b[k * ldb + j]`) instead of col-major. Same E4M3
//! grid-step tolerance and same value patterns.

#![cfg(feature = "sm89")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    EpilogueKind, Fp8E4M3, Fp8GemmArgs, Fp8GemmDescriptor, Fp8GemmPlan, LayoutSku, MatrixMut,
    MatrixRef, PlanPreference, Workspace,
};
use float8::F8E4M3;

fn e4m3_grid_spacing(v: f32) -> f32 {
    let a = v.abs();
    if a == 0.0 || a < 1.0 / 64.0 {
        return 1.0 / 512.0;
    }
    let e_unb = a.log2().floor() as i32;
    2f32.powi(e_unb - 3)
}

fn quantize_e4m3(x: f32) -> u8 {
    F8E4M3::from_f32(x).to_bits()
}

fn dequantize_e4m3(bits: u8) -> f32 {
    F8E4M3::from_bits(bits).to_f32()
}

/// CPU reference, RRR layout (row-major B, indexed `b[k * ldb + j]`).
#[allow(clippy::too_many_arguments)]
fn cpu_fp8_e4m3_gemm_rrr(
    m: usize,
    n: usize,
    k: usize,
    a_bits: &[u8],
    lda: usize,
    b_bits: &[u8],
    ldb: usize,
    alpha: f32,
    expected_bits: &mut [u8],
    expected_f32: &mut [f32],
    ldd: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0;
            for kk in 0..k {
                let a_val = dequantize_e4m3(a_bits[i * lda + kk]);
                let b_val = dequantize_e4m3(b_bits[kk * ldb + j]);
                acc += a_val * b_val;
            }
            let z = alpha * acc;
            expected_f32[i * ldd + j] = z;
            expected_bits[i * ldd + j] = quantize_e4m3(z);
        }
    }
}

fn run_fp8_e4m3_rrr_identity(m: i32, n: i32, k: i32) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;

    let host_a_bits: Vec<u8> = (0..(mu * ku))
        .map(|i| {
            let v = (((i as i32 * 5) % 13) as f32 - 6.0) * 0.125;
            quantize_e4m3(v)
        })
        .collect();
    let host_b_bits: Vec<u8> = (0..(ku * nu))
        .map(|i| {
            let v = (((i as i32 * 7) % 11) as f32 - 5.0) * 0.125;
            quantize_e4m3(v)
        })
        .collect();

    let alpha: f32 = 0.25;
    let beta: f32 = 0.0;

    let mut expected_bits = vec![0u8; mu * nu];
    let mut expected_f32 = vec![0f32; mu * nu];
    cpu_fp8_e4m3_gemm_rrr(
        mu, nu, ku,
        &host_a_bits, ku,
        &host_b_bits, nu, // B is row-major [K, N] with row stride = N
        alpha,
        &mut expected_bits,
        &mut expected_f32,
        nu,
    );

    let dev_a_bytes = DeviceBuffer::from_slice(&ctx, &host_a_bits).expect("upload A");
    let dev_b_bytes = DeviceBuffer::from_slice(&ctx, &host_b_bits).expect("upload B");
    let dev_a = dev_a_bytes.view_as::<Fp8E4M3>();
    let dev_b = dev_b_bytes.view_as::<Fp8E4M3>();
    let mut dev_d: DeviceBuffer<Fp8E4M3> =
        DeviceBuffer::zeros(&ctx, mu * nu).expect("alloc D");

    let desc = Fp8GemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rrr,
        epilogue: EpilogueKind::Identity,
    };
    let plan = Fp8GemmPlan::<Fp8E4M3>::select(&stream, &desc, PlanPreference::default())
        .expect("select FP8 E4M3 RRR plan");

    let args = Fp8GemmArgs::<Fp8E4M3> {
        a: MatrixRef { data: dev_a, rows: m, cols: k, ld: k as i64 },
        // RRR: B is row-major [K, N] with row stride = N
        b: MatrixRef { data: dev_b, rows: k, cols: n, ld: n as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        bias: None,
        alpha,
        beta,
    };
    plan.run(&stream, Workspace::None, args)
        .expect("FP8 E4M3 RRR GEMM run");
    stream.synchronize().expect("stream sync");

    let mut host_d_bits = vec![Fp8E4M3(0); mu * nu];
    dev_d.copy_to_host(&mut host_d_bits).expect("download D");

    let mut mismatches = 0usize;
    let mut first_mismatch: Option<(usize, u8, u8, f32, f32, f32)> = None;
    for (idx, (got, &expected)) in host_d_bits.iter().zip(expected_bits.iter()).enumerate() {
        let got_f = dequantize_e4m3(got.0);
        let exp_f = dequantize_e4m3(expected);
        let spacing = e4m3_grid_spacing(got_f.abs().max(exp_f.abs()));
        let delta = (got_f - exp_f).abs();
        if delta > spacing + 1e-7 {
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((idx, got.0, expected, got_f, exp_f, expected_f32[idx]));
            }
        }
    }
    if mismatches > 0 {
        let (idx, gb, eb, gf, ef, raw) = first_mismatch.unwrap();
        panic!(
            "{mismatches} mismatches across {} cells \
             (M={m} N={n} K={k}); first @ idx {idx}: \
             got bits=0x{gb:02x} ({gf}) expected bits=0x{eb:02x} ({ef}); \
             pre-quant f32 ref = {raw}",
            host_d_bits.len(),
        );
    }
}

#[test] #[ignore]
fn fp8_e4m3_rrr_identity_64_64_32() {
    run_fp8_e4m3_rrr_identity(64, 64, 32);
}

#[test] #[ignore]
fn fp8_e4m3_rrr_identity_128_128_128() {
    run_fp8_e4m3_rrr_identity(128, 128, 128);
}

#[test] #[ignore]
fn fp8_e4m3_rrr_identity_256_128_64() {
    run_fp8_e4m3_rrr_identity(256, 128, 64);
}

#[test] #[ignore]
fn fp8_e4m3_rrr_identity_100_70_50() {
    run_fp8_e4m3_rrr_identity(100, 70, 50);
}
