//! Device check of the CUDA spellings of `Fma`, `Cast` and `Bitcast` (ROADMAP SA-43 / SA-44).
//!
//! Each case builds an elementwise op, generates it with [`Cuda`], compiles it with NVRTC,
//! runs it on the device and compares every output against Unpopped's oracle
//! (`unpopped::oracle::evaluate`, which computes casts from exact integers and IEEE bit
//! patterns). A disagreement is a defect in the emitter or the oracle; the two are
//! independent implementations.
//!
//! Run with `cargo gpu-test -p baracuda-cuda-emit --features nvrtc,device-tests --test typed_exprs`.
#![cfg(all(feature = "nvrtc", feature = "device-tests"))]

use ElementKind::*;
use baracuda_cuda_emit::{Cuda, NvrtcCompiler};
use baracuda_driver::{Context, Device, DeviceBuffer, Function, Module, Stream, init};
use unpopped::backend::GeneratedKernel;
use unpopped::ir::{Expr, OpDef, RoundMode, Saturation, input, konst};
use unpopped::jit::Compiler;
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::{build_plan, try_generate};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key};

struct Case {
    tag: String,
    body: Expr,
    dt: ElementKind,
    out: ElementKind,
    /// One value list per input operand (all the same length).
    xs: Vec<Vec<f64>>,
}

fn bytes_of(dt: ElementKind) -> usize {
    match dt {
        I8 | U8 => 1,
        I16 | U16 => 2,
        F32 | F32Strict | I32 | U32 => 4,
        F64 | I64 => 8,
        other => panic!("typed test: unsupported {other:?}"),
    }
}

/// Little-endian bytes of `v` stored as `dt`.
fn encode(v: f64, dt: ElementKind) -> Vec<u8> {
    match dt {
        F32 | F32Strict => (v as f32).to_le_bytes().to_vec(),
        F64 => v.to_le_bytes().to_vec(),
        other => panic!("typed test input must be a float, got {other:?}"),
    }
}

/// An exact comparable value of the stored `bits` as type `t`: `Ok(int)` or `Err(float)`.
fn value(bits: u128, t: ElementKind) -> Result<i128, f64> {
    match t {
        I8 => Ok(i128::from(bits as u8 as i8)),
        I16 => Ok(i128::from(bits as u16 as i16)),
        I32 => Ok(i128::from(bits as u32 as i32)),
        I64 => Ok(i128::from(bits as u64 as i64)),
        U8 => Ok(i128::from(bits as u8)),
        U16 => Ok(i128::from(bits as u16)),
        U32 => Ok(i128::from(bits as u32)),
        F32 | F32Strict => Err(f64::from(f32::from_bits(bits as u32))),
        F64 => Err(f64::from_bits(bits as u64)),
        other => panic!("typed test: unsupported {other:?}"),
    }
}

fn same(a: Result<i128, f64>, b: Result<i128, f64>) -> bool {
    match (a, b) {
        (Ok(x), Ok(y)) => x == y,
        (Err(x), Err(y)) => (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits(),
        _ => false,
    }
}

/// Everything one case needs before it touches the device.
struct Prepared {
    op: OpDef,
    key: StructureKey,
    operands: Vec<OperandDesc>,
    kernel: GeneratedKernel,
    /// The input columns actually used (padded to an odd lane count).
    xs: Vec<Vec<f64>>,
}

fn prepare(c: &Case) -> Result<Prepared, String> {
    // An odd lane count keeps the planner on the scalar schedule.
    let mut xs = c.xs.clone();
    if xs[0].len().is_multiple_of(2) {
        for x in &mut xs {
            x.push(7.0);
        }
    }
    let n = xs[0].len() as i64;
    let n_in = xs.len() as u8;
    let mut op = OpDef::elementwise(&c.tag, n_in, &[c.dt], c.body.clone());
    if c.out != c.dt {
        op.out_dtype = Some(c.out);
    }
    let a = OperandDesc::new(1, &[n], &[1], c.dt, 256);
    let o = OperandDesc::new(1, &[n], &[1], c.out, 256);
    let mut operands = vec![a; usize::from(n_in)];
    operands.push(o);
    let category = if n_in == 1 {
        OpCategory::UnaryElementwise
    } else {
        OpCategory::BinaryElementwise
    };
    let key = structure_key(category, &operands, ArchSku::Sm89);
    let kernel =
        try_generate(&op, &key, &Cuda).map_err(|e| format!("{}: generate: {e:?}", c.tag))?;
    Ok(Prepared {
        op,
        key,
        operands,
        kernel,
        xs,
    })
}

fn load_kernel(
    ctx: &Context,
    cc: &NvrtcCompiler,
    tag: &str,
    k: &GeneratedKernel,
) -> Result<(Module, String), String> {
    let ptx = cc
        .compile(&k.source, &k.name, 5000)
        .map_err(|e| format!("{tag}: nvrtc: {e}\n{}", k.source))?;
    let ptx = String::from_utf8(ptx).map_err(|e| format!("{tag}: ptx utf8: {e}"))?;
    let module = Module::load_ptx(ctx, &ptx).map_err(|e| format!("{tag}: load: {e:?}"))?;
    Ok((module, k.name.clone()))
}

/// Launch the generated scalar kernel `(const T* in0.., OT* out, long long n)`.
fn launch_kernel(
    func: &Function,
    stream: &Stream,
    ins: &[DeviceBuffer<u8>],
    out: &DeviceBuffer<u8>,
    n: i64,
) -> Result<(), String> {
    let block = 256u32;
    let grid = (n as u32).div_ceil(block);
    let in_ptrs: Vec<_> = ins.iter().map(DeviceBuffer::as_raw).collect();
    let out_ptr = out.as_raw();
    // SAFETY: the kernel's signature is (const T* in0.., OT* out, long long n); the arguments
    // are pushed in that order with those types, every buffer outlives the launch (the caller
    // synchronizes the stream before dropping them), and `n` is the element count of each.
    unsafe {
        let mut l = func
            .launch()
            .grid((grid, 1, 1))
            .block((block, 1, 1))
            .stream(stream);
        for p in &in_ptrs {
            l = l.arg(p);
        }
        l.arg(&out_ptr)
            .arg(&n)
            .launch()
            .map_err(|e| format!("launch: {e:?}"))
    }
}

/// Run the prepared kernel on the device and return the raw output bytes.
fn run_on_device(
    ctx: &Context,
    stream: &Stream,
    cc: &NvrtcCompiler,
    c: &Case,
    p: &Prepared,
) -> Result<Vec<u8>, String> {
    let (module, name) = load_kernel(ctx, cc, &c.tag, &p.kernel)?;
    let func = module
        .get_function(&name)
        .map_err(|e| format!("{}: function: {e:?}", c.tag))?;
    let n = p.xs[0].len();
    let d_in: Vec<DeviceBuffer<u8>> =
        p.xs.iter()
            .map(|col| {
                let bytes: Vec<u8> = col.iter().flat_map(|v| encode(*v, c.dt)).collect();
                DeviceBuffer::from_slice(ctx, &bytes).expect("upload")
            })
            .collect();
    let out_len = n * bytes_of(c.out);
    let d_out: DeviceBuffer<u8> = DeviceBuffer::zeros(ctx, out_len).expect("alloc out");
    launch_kernel(&func, stream, &d_in, &d_out, n as i64).map_err(|e| format!("{}: {e}", c.tag))?;
    stream
        .synchronize()
        .map_err(|e| format!("{}: sync: {e:?}", c.tag))?;
    let mut got = vec![0u8; out_len];
    d_out
        .copy_to_host(&mut got)
        .map_err(|e| format!("{}: download: {e:?}", c.tag))?;
    Ok(got)
}

/// The oracle's output for the prepared case.
fn oracle_output(c: &Case, p: &Prepared) -> TypedBuffer {
    let n = p.xs[0].len() as i64;
    let to_buf = |col: &Vec<f64>| {
        if c.dt == F32 || c.dt == F32Strict {
            let col32: Vec<f32> = col.iter().map(|v| *v as f32).collect();
            TypedBuffer::from_f32(&[n], &col32)
        } else {
            TypedBuffer::from_f64(&[n], col)
        }
    };
    let bufs: Vec<TypedBuffer> = p.xs.iter().map(to_buf).collect();
    let plan = build_plan(&p.op, &p.key);
    evaluate(&plan, &p.operands, &bufs, &[])
        .into_iter()
        .next()
        .expect("one output")
}

/// The little-endian `i`-th element of `bytes`, `w` bytes wide, as raw bits.
fn bits_of(bytes: &[u8], i: usize, w: usize) -> u128 {
    bytes[i * w..(i + 1) * w]
        .iter()
        .enumerate()
        .fold(0u128, |acc, (k, b)| acc | (u128::from(*b) << (8 * k)))
}

fn run_case(
    ctx: &Context,
    stream: &Stream,
    cc: &NvrtcCompiler,
    c: &Case,
) -> Result<Vec<String>, String> {
    let p = prepare(c)?;
    let got = run_on_device(ctx, stream, cc, c, &p)?;
    let want = oracle_output(c, &p);
    let w = bytes_of(c.out);
    let mut diffs = Vec::new();
    for i in 0..p.xs[0].len() {
        let (g, e) = (
            value(bits_of(&got, i, w), c.out),
            value(want.bits_at(i), c.out),
        );
        if !same(g, e) {
            let inputs: Vec<f64> = p.xs.iter().map(|col| col[i]).collect();
            diffs.push(format!(
                "  {}: x = {inputs:?} -> device {g:?}, oracle {e:?}",
                c.tag
            ));
        }
    }
    Ok(diffs)
}

fn run_cases(cases: Vec<Case>) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let cc = NvrtcCompiler::new(ArchSku::Sm89.into()).expect("Sm89 is a well-formed cuda: target");
    let n_cases = cases.len();
    let mut report = Vec::new();
    for c in &cases {
        match run_case(&ctx, &stream, &cc, c) {
            Ok(d) => report.extend(d),
            Err(e) => report.push(format!("  ERROR {e}")),
        }
    }
    assert!(
        report.is_empty(),
        "CUDA Fma/Cast/Bitcast disagrees with the oracle in {} place(s) over {n_cases} cases:\n{}",
        report.len(),
        report.join("\n")
    );
}

/// Edge inputs for an f32 compute dtype (each is rounded to f32 on both sides).
const EDGE_VALUES: &[f64] = &[
    0.0,
    -0.0,
    0.5,
    -0.5,
    1.0,
    -1.0,
    1.5,
    -1.5,
    2.5,
    -2.5,
    3.99,
    -3.99,
    126.5,
    127.0,
    127.5,
    128.0,
    -128.0,
    -128.5,
    -129.0,
    254.5,
    255.0,
    255.5,
    256.0,
    -1.0e-3,
    32767.5,
    32768.0,
    -32768.0,
    -32769.0,
    65535.0,
    65535.5,
    65536.0,
    16777216.0,
    16777217.0,
    2147483520.0,
    2147483648.0,
    -2147483648.0,
    -2147483904.0,
    4294967040.0,
    4294967296.0,
    9_223_372_036_854_775_808.0,
    -9_223_372_036_854_775_808.0,
    1.8446744073709552e19,
    3.4028234663852886e38,
    -3.4028234663852886e38,
    1.0e-30,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
];

/// [`EDGE_VALUES`] plus the two f32 neighbours of one half.
fn edges() -> Vec<f64> {
    let mut v = EDGE_VALUES.to_vec();
    v.push(f64::from(f32::from_bits(0x3eff_ffff))); // 0.49999997
    v.push(f64::from(f32::from_bits(0x3f00_0001))); // 0.50000006
    v
}

const MODES: [(&str, RoundMode); 4] = [
    ("rn", RoundMode::NearestEven),
    ("rz", RoundMode::TowardZero),
    ("rm", RoundMode::TowardNegInf),
    ("rp", RoundMode::TowardPosInf),
];

fn case(tag: String, body: Expr, dt: ElementKind, out: ElementKind, xs: Vec<f64>) -> Case {
    Case {
        tag,
        body,
        dt,
        out,
        xs: vec![xs],
    }
}

#[test]
fn float_to_integer_casts_agree_with_the_oracle_in_every_rounding_mode() {
    let mut cases = Vec::new();
    for dt in [F32, F64] {
        for &t in &[I8, I16, I32, I64, U8, U16, U32] {
            for (name, mode) in MODES {
                cases.push(case(
                    format!("f2i_{dt:?}_{t:?}_{name}"),
                    input(0).cast(t, mode, Saturation::Clamp),
                    dt,
                    t,
                    edges(),
                ));
            }
        }
    }
    run_cases(cases);
}

#[test]
fn integer_to_integer_casts_wrap_and_clamp_like_the_oracle() {
    let mut cases = Vec::new();
    for &t in &[I8, I16, U8, U16, U32, I64] {
        for (name, sat) in [("wrap", Saturation::Wrap), ("clamp", Saturation::Clamp)] {
            // float -> i32 (saturating) -> t: the second cast is the int-to-int under test.
            let body = input(0)
                .cast(I32, RoundMode::TowardZero, Saturation::Clamp)
                .cast(t, RoundMode::NearestEven, sat);
            cases.push(case(format!("i2i_{t:?}_{name}"), body, F32, t, edges()));
        }
    }
    run_cases(cases);
}

#[test]
fn integer_to_float_casts_round_like_the_oracle() {
    let mut cases = Vec::new();
    for (via, vname) in [
        (I32, "i32"),
        (I64, "i64"),
        (U32, "u32"),
        (I8, "i8"),
        (U16, "u16"),
    ] {
        for &t in &[F32, F64] {
            for (name, mode) in MODES {
                let body = input(0)
                    .cast(via, RoundMode::TowardZero, Saturation::Clamp)
                    .cast(t, mode, Saturation::Wrap);
                cases.push(case(
                    format!("i2f_{vname}_{t:?}_{name}"),
                    body,
                    F32,
                    t,
                    edges(),
                ));
            }
        }
    }
    run_cases(cases);
}

#[test]
fn float_to_float_casts_round_and_clamp_like_the_oracle() {
    let mut xs = edges();
    xs.extend([
        1.0e300,
        -1.0e300,
        3.4028235677973366e38, // just above f32::MAX, rounds to f32::MAX under RNE
        3.4028235677973367e38, // halfway to the next binade, rounds to inf
        1.0 + 2.0f64.powi(-24), // a tie between two f32s
        1.0 + 3.0 * 2.0f64.powi(-25),
        1.0e-46, // below the smallest f32 subnormal
    ]);
    let mut cases = Vec::new();
    for (mname, mode) in MODES {
        for (name, sat) in [
            ("wrap", Saturation::Wrap),
            ("clampfinite", Saturation::ClampFinite),
        ] {
            cases.push(case(
                format!("f2f_f64_f32_{name}_{mname}"),
                input(0).cast(F32, mode, sat),
                F64,
                F32,
                xs.clone(),
            ));
        }
    }
    cases.push(case(
        "f2f_f32_f64".to_string(),
        input(0).cast(F64, RoundMode::NearestEven, Saturation::Wrap),
        F32,
        F64,
        edges(),
    ));
    run_cases(cases);
}

#[test]
fn bitcast_moves_bits_exactly() {
    let cases = vec![
        case(
            "bc_f32_i32".to_string(),
            input(0).bitcast(I32),
            F32,
            I32,
            edges(),
        ),
        case(
            "bc_f32_u32".to_string(),
            input(0).bitcast(U32),
            F32,
            U32,
            edges(),
        ),
        case(
            "bc_f64_i64".to_string(),
            input(0).bitcast(I64),
            F64,
            I64,
            edges(),
        ),
        case(
            "bc_roundtrip".to_string(),
            input(0).bitcast(U32).bitcast(F32),
            F32,
            F32,
            edges(),
        ),
        case(
            "bc_i32_to_u32_to_i32".to_string(),
            input(0)
                .cast(I32, RoundMode::TowardZero, Saturation::Clamp)
                .bitcast(U32),
            F32,
            U32,
            edges(),
        ),
    ];
    run_cases(cases);
}

#[test]
fn fma_has_one_rounding_in_f32_and_f64() {
    // a*b+c where the exact product is not representable, so mul-then-add and fused differ.
    let big = 1.0 + 2.0f64.powi(-12);
    let a = vec![
        big,
        -big,
        1.0 + 2.0f64.powi(-23),
        3.0,
        0.1,
        1.0e30,
        f64::NAN,
        f64::INFINITY,
        0.0,
    ];
    let b = vec![
        big,
        big,
        1.0 - 2.0f64.powi(-23),
        0.1,
        10.0,
        1.0e10,
        1.0,
        0.0,
        5.0,
    ];
    let mut cases = Vec::new();
    for dt in [F32, F64] {
        // fma(a, b, -(a*b rounded)) exposes the rounding error of the product.
        cases.push(Case {
            tag: format!("fma_{dt:?}"),
            body: input(0).fma(input(1), konst(-1.0)),
            dt,
            out: dt,
            xs: vec![a.clone(), b.clone()],
        });
        cases.push(Case {
            tag: format!("fma_cc_{dt:?}"),
            body: input(0).fma(input(1), input(0)),
            dt,
            out: dt,
            xs: vec![a.clone(), b.clone()],
        });
    }
    run_cases(cases);
}
