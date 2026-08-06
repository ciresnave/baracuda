//! Gate 5 (regression repro) + gate 6 (on-device output-CORRECTNESS) for the
//! alpha.78 all-zero bug: Fuel bisected a synthesized `relu(add(a,b))` f32-scalar
//! kernel writing all zeros (alpha.76/77 PASS, .78 FAIL; byte-identical symbol →
//! emitted body or launch ABI, not identity). This test EMITS the exact kernel,
//! NVRTC-compiles it, LAUNCHES it on the RTX 4070 with a mixed-sign input, and
//! asserts the output equals `relu(a+b)` element-for-element — the gap that let an
//! all-zero-no-error kernel ship (goldens prove bytes, on-device proves compile,
//! nothing proved the launched kernel writes the RIGHT answer).
//!
//! The output buffer is pre-seeded with a NaN sentinel so "kernel wrote zeros"
//! (the bug) is distinguishable from "kernel never wrote" (a launch/ABI failure).
//!
//! Ignored by default (needs a CUDA device + nvrtc); run with:
//! `cargo test -p baracuda-kernels-bench --test alpha78_relu_add_ondevice -- --ignored --nocapture`

use baracuda_cuda_emit::{Cuda, NvrtcCompiler};
use baracuda_driver::{DeviceBuffer, Module};
use baracuda_kernelgen::{Compiler, OpDef, contract, generate, input};
use baracuda_kernels_bench::setup_device;
use baracuda_kernels_types::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

/// Parse the `count_unit:` line of an FKC contract into the divisor `w` for the
/// launch count (`elements` -> 1, `vectors_x{w}` -> w). This is exactly the read
/// a launcher must do to marshal `n` — validating it end-to-end on device is the
/// gap the alpha.78 all-zero exposed (count_unit is the emit→launch contract).
fn count_unit_divisor(contract_text: &str) -> u32 {
    for line in contract_text.lines() {
        let l = line.trim();
        if let Some(rest) = l.strip_prefix("count_unit:") {
            let rest = rest.trim();
            if rest == "elements" {
                return 1;
            }
            if let Some(w) = rest.strip_prefix("vectors_x") {
                return w.parse().expect("count_unit width");
            }
            panic!("unrecognized count_unit: {rest:?}");
        }
    }
    panic!("no count_unit in contract:\n{contract_text}");
}

#[test]
#[ignore = "requires a CUDA device + nvrtc"]
fn relu_add_f32_scalar_writes_the_right_answer_on_device() {
    let (ctx, stream) = setup_device();

    // The exact op Fuel synthesizes: relu(add(in0, in1)), f32.
    let op = OpDef::elementwise(
        "relu_add",
        2,
        &[ElementKind::F32],
        (input(0) + input(1)).relu(),
    );
    // align = 4 → Schedule::Scalar (the general one-thread-per-element path, not
    // float4-vectorized) — the shape the synth seam lands on.
    let n: i64 = 4096;
    let a = OperandDesc::new(1, &[n], &[1], ElementKind::F32, 4);
    let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
    let k = generate(&op, &key, &Cuda);
    println!("=== NAME ===\n{}", k.name);
    println!("=== SOURCE ===\n{}", k.source);

    // Mixed-sign input so a correct relu output has BOTH zeros (a+b < 0) and
    // positive values — an all-zero output is then unambiguously wrong.
    let in0: Vec<f32> = (0..n).map(|i| (i as f32) - (n as f32) / 2.0).collect();
    let in1: Vec<f32> = vec![0.25f32; n as usize];
    let expected: Vec<f32> = in0
        .iter()
        .zip(&in1)
        .map(|(&a, &b)| {
            let s = a + b;
            if s < 0.0 { 0.0 } else { s }
        })
        .collect();

    let compiler = NvrtcCompiler::new(ArchSku::Sm89);
    let ptx = compiler
        .compile(&k.source, &k.name, 30_000)
        .unwrap_or_else(|e| panic!("nvrtc({}) failed: {e}", k.name));
    let ptx = String::from_utf8(ptx).expect("ptx is text");
    let module = Module::load_ptx(&ctx, &ptx).expect("module load");
    let f = module.get_function(&k.name).expect("get_function");

    let d_in0 = DeviceBuffer::from_slice(&ctx, &in0).expect("upload in0");
    let d_in1 = DeviceBuffer::from_slice(&ctx, &in1).expect("upload in1");
    // Seed output with a NaN sentinel: distinguishes "wrote zeros" from "never wrote".
    let sentinel = vec![f32::NAN; n as usize];
    let d_out = DeviceBuffer::from_slice(&ctx, &sentinel).expect("alloc out");

    let block = 256u32;
    let grid = ((n as u32) + block - 1) / block;
    // Scalar-elementwise ABI: (const float* in0, const float* in1, float* out, long long n).
    // SAFETY: the launch signature matches the emitted kernel; buffers outlive it.
    unsafe {
        f.launch()
            .grid(grid)
            .block(block)
            .stream(&stream)
            .arg(&d_in0)
            .arg(&d_in1)
            .arg(&d_out)
            .arg(&n)
            .launch()
            .unwrap_or_else(|e| panic!("launch failed: {e}"));
    }
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; n as usize];
    d_out.copy_to_host(&mut got).expect("download");

    // Diagnostics before asserting: is it the all-zero symptom, the never-wrote
    // symptom (sentinel survives), or correct?
    let n_nan = got.iter().filter(|x| x.is_nan()).count();
    let n_zero = got.iter().filter(|&&x| x == 0.0).count();
    let n_pos = got.iter().filter(|&&x| x > 0.0).count();
    let expected_pos = expected.iter().filter(|&&x| x > 0.0).count();
    println!(
        "output: {n_nan} NaN-sentinel (never-wrote), {n_zero} zero, {n_pos} positive; \
         expected {expected_pos} positive"
    );
    assert_eq!(
        n_nan, 0,
        "kernel never wrote {n_nan} elements (launch/ABI failure)"
    );
    assert!(
        n_pos > 0,
        "ALL-ZERO REGRESSION REPRODUCED: kernel wrote {n_zero} zeros / {n_pos} positive, \
         expected {expected_pos} positive relu outputs"
    );

    // Exact element-for-element correctness (relu of an exactly-representable sum
    // is exact — no tolerance needed).
    let mut mism = 0usize;
    for (idx, (&g, &e)) in got.iter().zip(&expected).enumerate() {
        if g != e {
            if mism < 5 {
                println!(
                    "  mismatch[{idx}]: got {g}, expected {e} (in0={}, in1={})",
                    in0[idx], in1[idx]
                );
            }
            mism += 1;
        }
    }
    assert_eq!(mism, 0, "{mism}/{n} elements differ from relu(a+b)");
    println!(
        "PASS: relu(add) f32 scalar writes the correct answer on the RTX 4070 ({n_pos} positive, {n_zero} clamped-to-zero)."
    );
}

#[test]
#[ignore = "requires a CUDA device + nvrtc"]
fn relu_add_f32_vectorized_launches_by_count_unit_on_device() {
    // The VECTORIZED (float4) path — align 16 → Schedule::Vectorized{4}, whose
    // contract advertises `count_unit: vectors_x4`. A launcher that reads count_unit
    // and passes n = elements/4 must get the right answer; passing n = elements is a
    // 4x OOB, and passing n ≈ 0 is the all-zero symptom. This launches by the
    // count_unit-derived count and asserts correctness — validating the emit→launch
    // contract Fuel bisected end-to-end on the RTX 4070.
    let (ctx, stream) = setup_device();

    let op = OpDef::elementwise(
        "relu_add",
        2,
        &[ElementKind::F32],
        (input(0) + input(1)).relu(),
    );
    let numel: i64 = 4096; // multiple of 4
    let a = OperandDesc::new(1, &[numel], &[1], ElementKind::F32, 16); // align 16 -> vectorized
    let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
    let k = generate(&op, &key, &Cuda);
    let c = contract(&op, &key, &k, &Cuda).expect("contract");
    let w = count_unit_divisor(&c) as i64;
    println!("=== VEC NAME ===\n{}\ncount_unit divisor w = {w}", k.name);
    assert_eq!(
        w, 4,
        "float4 elementwise must advertise count_unit vectors_x4"
    );
    let nv = numel / w; // the count the launcher marshals

    let in0: Vec<f32> = (0..numel)
        .map(|i| (i as f32) - (numel as f32) / 2.0)
        .collect();
    let in1: Vec<f32> = vec![0.25f32; numel as usize];
    let expected: Vec<f32> = in0
        .iter()
        .zip(&in1)
        .map(|(&a, &b)| {
            let s = a + b;
            if s < 0.0 { 0.0 } else { s }
        })
        .collect();

    let compiler = NvrtcCompiler::new(ArchSku::Sm89);
    let ptx = compiler
        .compile(&k.source, &k.name, 30_000)
        .unwrap_or_else(|e| panic!("nvrtc({}) failed: {e}", k.name));
    let ptx = String::from_utf8(ptx).expect("ptx");
    let module = Module::load_ptx(&ctx, &ptx).expect("load");
    let f = module.get_function(&k.name).expect("func");

    // f32 buffers passed as float4* (the device pointer is 16-byte aligned).
    let d_in0 = DeviceBuffer::from_slice(&ctx, &in0).expect("in0");
    let d_in1 = DeviceBuffer::from_slice(&ctx, &in1).expect("in1");
    let d_out = DeviceBuffer::from_slice(&ctx, &vec![f32::NAN; numel as usize]).expect("out");

    let block = 256u32;
    let grid = ((nv as u32) + block - 1) / block;
    // SAFETY: matches (float4* in0, float4* in1, float4* out, long long nv).
    unsafe {
        f.launch()
            .grid(grid)
            .block(block)
            .stream(&stream)
            .arg(&d_in0)
            .arg(&d_in1)
            .arg(&d_out)
            .arg(&nv)
            .launch()
            .unwrap_or_else(|e| panic!("launch: {e}"));
    }
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel as usize];
    d_out.copy_to_host(&mut got).expect("download");

    let n_nan = got.iter().filter(|x| x.is_nan()).count();
    let n_pos = got.iter().filter(|&&x| x > 0.0).count();
    println!(
        "vec output: {n_nan} never-wrote, {n_pos} positive; expected {}",
        expected.iter().filter(|&&x| x > 0.0).count()
    );
    assert_eq!(
        n_nan, 0,
        "count_unit-derived launch left {n_nan} elements unwritten"
    );
    let mism = got.iter().zip(&expected).filter(|(g, e)| g != e).count();
    assert_eq!(
        mism, 0,
        "{mism}/{numel} vectorized elements differ from relu(a+b)"
    );
    println!(
        "PASS: vectorized relu(add) launched by count_unit (n=elements/{w}) writes the correct answer on the RTX 4070."
    );
}
