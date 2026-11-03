//! Real-GPU test that `CastPlan` rounds f64 -> f16/bf16 and 32/64-bit integer ->
//! bf16 ONCE (round-to-nearest-even straight from the source), not twice through
//! f32.
//!
//! Before alpha.88 these casts went `source -> float -> half/bf16`. When the f32
//! rounding lands exactly on a half/bf16 tie, ties-to-even then picks the wrong
//! neighbour: `1 + 2^-11 + 2^-40` is above the f16 midpoint between `1.0` and
//! `1 + 2^-10`, so the correct result is `1.0009765625`, but the f32 hop gave
//! `1.0`. This is NOT the `half` crate's `f16::from_f64` bug (reported separately,
//! half 2.7.1); the reference here is an independent integer-arithmetic rounding
//! implemented in this file and does not call `from_f64`.
//!
//! Gated behind `device-tests`; run each test exe under `scripts/gpu-run.ps1`:
//! `cargo test -p baracuda-kernels --features device-tests --test cast_exact_rounding_smoke`.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    CastArgs, CastDescriptor, CastPlan, Element, ElementKind, PlanPreference, TensorMut, TensorRef,
    Workspace, contiguous_stride,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Round a finite, non-zero `x` (normal in f64, and whose result is normal and
/// finite in the target format) to `p` significant bits, ties to even, using only
/// integer arithmetic on the f64's exact mantissa. `p` = 11 for f16, 8 for bf16.
fn round_to_bits(x: f64, p: u32) -> f64 {
    let bits = x.to_bits();
    let neg = (bits >> 63) != 0;
    let mant = (bits & ((1u64 << 52) - 1)) | (1u64 << 52); // 53 bits, exact
    let exp = ((bits >> 52) & 0x7ff) as i64 - 1075; // x = mant * 2^exp
    let drop = 53 - p;
    let low = mant & ((1u64 << drop) - 1);
    let half = 1u64 << (drop - 1);
    let mut q = mant >> drop;
    if low > half || (low == half && (q & 1) == 1) {
        q += 1;
    }
    let e = exp + drop as i64; // result = q * 2^e, e within f64 normal range here
    let scale = f64::from_bits(((e + 1023) as u64) << 52);
    let r = (q as f64) * scale;
    if neg { -r } else { r }
}

/// Inputs that sit within a hair of a `p`-bit tie, on both sides, plus the exact
/// tie, for several magnitudes. These all trip a double rounding through f32.
fn near_tie_inputs(p: u32, bases: &[f64]) -> Vec<f64> {
    let mut v = Vec::new();
    for &b in bases {
        // b is exactly representable with p bits; ulp = 2^(floor(log2 b) - (p-1)).
        let e = ((b.to_bits() >> 52) & 0x7ff) as i64 - 1023;
        let ulp = f64::from_bits(((e - (p as i64 - 1) + 1023) as u64) << 52);
        let mid = b + ulp / 2.0;
        for rel in [2f64.powi(-40), 2f64.powi(-30), 2f64.powi(-45)] {
            v.push(mid * (1.0 + rel));
            v.push(mid * (1.0 - rel));
            v.push(-(mid * (1.0 + rel)));
            v.push(-(mid * (1.0 - rel)));
        }
        v.push(mid); // exact tie: ties-to-even
        v.push(-mid);
    }
    v
}

fn lcg_values(n: usize, lo: f64, hi: f64) -> Vec<f64> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
            lo + (hi - lo) * u
        })
        .collect()
}

fn run_cast<TIn: Element + Copy, TOut: Element + Copy + Default>(
    ctx: &Context,
    stream: &Stream,
    input: &[TIn],
    in_kind: ElementKind,
    out_kind: ElementKind,
) -> Vec<TOut> {
    let n = input.len();
    let dev_x = DeviceBuffer::from_slice(ctx, input).expect("upload");
    let mut dev_y: DeviceBuffer<TOut> = DeviceBuffer::zeros(ctx, n).expect("alloc");
    let desc = CastDescriptor {
        numel: n as i32,
        input_element: in_kind,
        output_element: out_kind,
    };
    let plan =
        CastPlan::<TIn, TOut>::select(stream, &desc, PlanPreference::default()).expect("select");
    let args = CastArgs {
        input: TensorRef::new(dev_x.as_slice(), [n as i32], contiguous_stride([n as i32])).unwrap(),
        output: TensorMut::new(
            dev_y.as_slice_mut(),
            [n as i32],
            contiguous_stride([n as i32]),
        )
        .unwrap(),
    };
    plan.run(stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![TOut::default(); n];
    dev_y.copy_to_host(&mut got).expect("download");
    got
}

fn f16_inputs() -> Vec<f64> {
    // Bases are exact f16 values in the normal range, so the midpoint/result stay
    // normal and finite in f16.
    let bases = [
        1.0,
        1.5,
        3.0,
        0.75,
        100.0,
        1000.5,
        0.0625,
        7.0,
        2048.0,
        0.045166015625,
    ];
    let mut v = near_tie_inputs(11, &bases);
    v.push(1.0 + 2f64.powi(-11) + 2f64.powi(-40)); // the documented counterexample
    v.push(0.04518128032270489); // the value unpopped reported against half 2.7.1
    v.extend(lcg_values(4096, 1e-3, 1e3));
    v.extend(lcg_values(512, -1e3, -1e-3));
    v
}

fn bf16_inputs() -> Vec<f64> {
    let bases = [
        1.0, 1.5, 3.0, 0.75, 100.0, 1000.0, 0.0625, 7.0, 65536.0, 1e10,
    ];
    // make every base exactly bf16-representable
    let bases: Vec<f64> = bases.iter().map(|&b| round_to_bits(b, 8)).collect();
    let mut v = near_tie_inputs(8, &bases);
    v.push(1.0 + 2f64.powi(-8) + 2f64.powi(-40)); // above the bf16 midpoint of 1.0 / 1+2^-7
    v.extend(lcg_values(4096, 1e-3, 1e3));
    v.extend(lcg_values(512, -1e3, -1e-3));
    v
}

#[test]
fn cast_f64_to_f16_rounds_once() {
    let (ctx, stream) = setup();
    let xs = f16_inputs();
    let got: Vec<f16> = run_cast(&ctx, &stream, &xs, ElementKind::F64, ElementKind::F16);
    let mut bad = Vec::new();
    for (i, (&x, g)) in xs.iter().zip(got.iter()).enumerate() {
        let want = round_to_bits(x, 11);
        if g.to_f64() != want {
            bad.push((i, x, g.to_f64(), want));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {} f64->f16 casts differ from exact round-to-nearest-even; first: {:?}",
        bad.len(),
        xs.len(),
        &bad[..bad.len().min(5)]
    );
    println!("f64->f16: {} values checked, 0 mismatches", xs.len());
}

#[test]
fn cast_f64_to_bf16_rounds_once() {
    let (ctx, stream) = setup();
    let xs = bf16_inputs();
    let got: Vec<bf16> = run_cast(&ctx, &stream, &xs, ElementKind::F64, ElementKind::Bf16);
    let mut bad = Vec::new();
    for (i, (&x, g)) in xs.iter().zip(got.iter()).enumerate() {
        let want = round_to_bits(x, 8);
        if g.to_f64() != want {
            bad.push((i, x, g.to_f64(), want));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {} f64->bf16 casts differ from exact round-to-nearest-even; first: {:?}",
        bad.len(),
        xs.len(),
        &bad[..bad.len().min(5)]
    );
    println!("f64->bf16: {} values checked, 0 mismatches", xs.len());
}

#[test]
fn cast_wide_integers_to_bf16_round_once() {
    let (ctx, stream) = setup();
    // Each value is one unit above a bf16 midpoint; the unit is lost when the
    // integer is first rounded to f32, which then lands on the tie.
    let i32s: Vec<i32> = vec![
        (1 << 30) + (1 << 22) + 1,
        -((1 << 30) + (1 << 22) + 1),
        (1 << 28) + (1 << 20) + 1,
        (1 << 25) + (1 << 17) + 1,
        1 << 30,
        12345,
        -7,
    ];
    let got: Vec<bf16> = run_cast(&ctx, &stream, &i32s, ElementKind::I32, ElementKind::Bf16);
    for (x, g) in i32s.iter().zip(got.iter()) {
        let want = round_to_bits(*x as f64, 8);
        assert_eq!(g.to_f64(), want, "i32 {x} -> bf16");
    }
    // u32 -> bf16 (also fixed, `__uint2bfloat16_rn`) is not reachable through
    // `CastPlan<u32, _>` (u32 is not an `Element`), so it has no device test here.
    // Values below 2^53 are exact in f64, so the f64 reference is exact.
    let i64s: Vec<i64> = vec![
        (1i64 << 40) + (1 << 32) + 1,
        -((1i64 << 40) + (1 << 32) + 1),
        1 << 50,
    ];
    let got: Vec<bf16> = run_cast(&ctx, &stream, &i64s, ElementKind::I64, ElementKind::Bf16);
    for (x, g) in i64s.iter().zip(got.iter()) {
        let want = round_to_bits(*x as f64, 8);
        assert_eq!(g.to_f64(), want, "i64 {x} -> bf16");
    }
}

/// Host-side self-check of the reference rounding (runs on the device-tests build
/// but needs no GPU): the documented counterexample and a few hand-computed ties.
#[test]
fn reference_rounding_is_correct() {
    assert_eq!(
        round_to_bits(1.0 + 2f64.powi(-11) + 2f64.powi(-40), 11),
        1.0009765625
    );
    // exact tie between 1.0 (even) and 1+2^-10 (odd) -> 1.0
    assert_eq!(round_to_bits(1.0 + 2f64.powi(-11), 11), 1.0);
    // exact tie between 1+2^-10 (odd) and 1+2^-9 (even) -> 1+2^-9
    assert_eq!(
        round_to_bits(1.0 + 3.0 * 2f64.powi(-11), 11),
        1.0 + 2f64.powi(-9)
    );
    assert_eq!(round_to_bits(0.04518128032270489, 11), 0.045196533203125);
    assert_eq!(
        round_to_bits(-(1.0 + 2f64.powi(-11) + 2f64.powi(-40)), 11),
        -1.0009765625
    );
}
