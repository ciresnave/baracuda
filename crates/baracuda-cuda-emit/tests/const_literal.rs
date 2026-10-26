//! An `f32` kernel's constants are `float` literals (`1.0f`), so the arithmetic stays in
//! `float`.
//!
//! A suffix-less `1.0` is a C `double`: `g / (g + 1.0)` with a `float` `g` is evaluated in
//! double and rounded to `float` once more on the store, which is a different result from the
//! `float` arithmetic the IR means. This runs the generated kernel on the device and compares
//! every output bit with plain `f32` arithmetic on the host; it also counts how many inputs the
//! double form would have got wrong, so the test cannot pass by accident of the inputs.
//!
//! Run with `cargo gpu-test -p baracuda-cuda-emit --features nvrtc,device-tests --test const_literal`.
#![cfg(all(feature = "nvrtc", feature = "device-tests"))]

use baracuda_cuda_emit::{Cuda, NvrtcCompiler};
use baracuda_driver::{Context, Device, DeviceBuffer, Module, Stream, init};
use unpopped::ir::{OpDef, input, konst};
use unpopped::jit::Compiler;
use unpopped::try_generate;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

/// `g = a * b; out = g / (g + 1)`, the diamond whose shared interior is hoisted to a `tmp`.
fn diamond() -> OpDef {
    let g = input(0) * input(1);
    OpDef::elementwise(
        "const_diamond",
        2,
        &[ElementKind::F32],
        g.clone() / (g + konst(1.0)),
    )
}

fn generate(n: i64) -> unpopped::backend::GeneratedKernel {
    let t = OperandDesc::new(1, &[n], &[1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::BinaryElementwise, &[t, t, t], ArchSku::Sm89);
    try_generate(&diamond(), &key, &Cuda).expect("generate")
}

/// Deterministic inputs spread over a few binades (an xorshift, no external crate).
fn inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        // 2^-4 ..= 2^4 with a full 23-bit mantissa.
        let frac = (s >> 41) as f32 / (1u64 << 23) as f32;
        let exp = ((s >> 8) % 9) as i32 - 4;
        (1.0 + frac) * (exp as f32).exp2()
    };
    let a: Vec<f32> = (0..n).map(|_| next()).collect();
    let b: Vec<f32> = (0..n).map(|_| next()).collect();
    (a, b)
}

/// Run the generated kernel on the device over `a` and `b`.
fn run_on_device(kernel: &unpopped::backend::GeneratedKernel, a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    init().expect("cuInit");
    let device = Device::get(0).expect("device");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let compiler = NvrtcCompiler::new(ArchSku::Sm89.into()).expect("nvrtc target");
    let ptx = compiler
        .compile(&kernel.source, &kernel.name, 5000)
        .unwrap_or_else(|e| {
            panic!(
                "nvrtc: {e}
{}",
                kernel.source
            )
        });
    let module = Module::load_ptx(&ctx, &String::from_utf8(ptx).expect("ptx utf8")).expect("load");
    let func = module.get_function(&kernel.name).expect("function");

    let da = DeviceBuffer::from_slice(&ctx, a).expect("upload a");
    let db = DeviceBuffer::from_slice(&ctx, b).expect("upload b");
    let dout = DeviceBuffer::<f32>::zeros(&ctx, n).expect("alloc out");
    let (pa, pb, po, nn) = (da.as_raw(), db.as_raw(), dout.as_raw(), n as i64);
    let block = 256u32;
    let grid = (n as u32).div_ceil(block);
    // SAFETY: the kernel is `(const float* in0, const float* in1, float* out, long long n)`;
    // the arguments are pushed in that order with those types, and the buffers `da`, `db` and
    // `dout` outlive the launch because the stream is synchronized before they drop.
    unsafe {
        func.launch()
            .grid((grid, 1, 1))
            .block((block, 1, 1))
            .stream(&stream)
            .arg(&pa)
            .arg(&pb)
            .arg(&po)
            .arg(&nn)
            .launch()
            .expect("launch");
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; n];
    dout.copy_to_host(&mut got).expect("download");
    got
}

/// The oracle: `float` arithmetic, as the IR means, and the defect: the same expression with a
/// double constant evaluated in double.
fn host_forms(a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let float_form = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let g = x * y;
            g / (g + 1.0f32)
        })
        .collect();
    let double_form = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let g = f64::from(x * y);
            (g / (g + 1.0f64)) as f32
        })
        .collect();
    (float_form, double_form)
}

fn run(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, String) {
    let kernel = generate(n as i64);
    let (a, b) = inputs(n);
    let got = run_on_device(&kernel, &a, &b);
    let (float_form, double_form) = host_forms(&a, &b);
    (got, float_form, double_form, kernel.source)
}

fn check(n: usize) {
    let (got, float_form, double_form, source) = run(n);
    assert!(
        source.contains("1.0f"),
        "the f32 kernel must spell its constant `1.0f`:\n{source}"
    );
    let wrong_if_double = float_form
        .iter()
        .zip(&double_form)
        .filter(|(f, d)| f.to_bits() != d.to_bits())
        .count();
    eprintln!(
        "n={n}: {wrong_if_double} of {n} inputs differ between the float and the double form"
    );
    assert!(
        wrong_if_double * 100 >= n,
        "control: the double form should differ on at least 1% of inputs, saw {wrong_if_double}/{n}"
    );
    let bad: Vec<usize> = (0..n)
        .filter(|&i| got[i].to_bits() != float_form[i].to_bits())
        .collect();
    assert!(
        bad.is_empty(),
        "{} of {n} outputs differ from f32 arithmetic; first at {}: device {:e} vs host {:e}\n{source}",
        bad.len(),
        bad[0],
        got[bad[0]],
        float_form[bad[0]]
    );
}

#[test]
fn scalar_schedule_is_bit_exact_f32_arithmetic() {
    // An odd length keeps the planner on the scalar schedule.
    check(20_001);
}

#[test]
fn vectorized_schedule_is_bit_exact_f32_arithmetic() {
    // A multiple of the vector width takes the `float4` path.
    check(20_000);
}
