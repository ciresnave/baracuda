//! On-device numerical two-sided validation of the precision-first
//! `MorePrecise` variants — proving the `_prec` kernel is MORE ACCURATE than
//! the base (not merely different) vs an f64 oracle, on an ill-conditioned
//! input, on real sm_89 hardware.
//!
//! The clean demonstration is the OUTER-AXIS reduction: the base folds each
//! column in a SERIAL `float` (the general one-thread-per-output path), and the
//! `_prec` variant folds the SAME serial order in `double`. Only the
//! accumulator width differs — so a summation whose exact result exceeds the
//! f32 mantissa's reach (`> 2^24`) SATURATES the float accumulator while the
//! double one stays exact. (The InnerContig / RowReduce bases fold in a
//! block-tree, which balances partials and stays accurate until sums exceed
//! 2^24 anyway — so the serial-base reduction is where the accuracy win is
//! dramatic and unambiguous; the RowReduce v1b variant's primary guarantee is
//! bitwise cross-hardware reproducibility, validated by construction.)
//!
//! Ignored by default (needs a CUDA device + nvrtc); run with:
//! `cargo test -p baracuda-kernels-bench --test precision_first_numerics -- --ignored --nocapture`

use baracuda_cuda_emit::{Cuda, NvrtcCompiler};
use baracuda_driver::{DeviceBuffer, Module};
use baracuda_kernelgen::{Compiler, OpDef, ReduceOp, VariantFidelity, generate_variants, input};
use baracuda_kernels_bench::setup_device;
use baracuda_kernels_types::{
    ArchSku, AxisMask, ElementKind, OpCategory, OperandDesc, structure_key,
};

#[test]
#[ignore = "requires a CUDA device + nvrtc"]
fn prec_reduction_beats_saturating_float_serial_sum() {
    let (ctx, stream) = setup_device();

    // Outer-axis (axis 0) f32 Sum over N rows into COLS outputs. N > 2^24 so a
    // serial float accumulator saturates at 16_777_216; the exact column sum is
    // N. COLS small so each of the N-long folds is a single serial thread — the
    // exact base structure the prec variant re-widens to double.
    const N: i64 = 20_000_000; // > 2^24 = 16_777_216
    const COLS: i64 = 4;
    const SAT: f64 = 16_777_216.0; // 2^24, where float-serial +1.0 stalls

    let op = OpDef::reduction_axes(
        "sum",
        1,
        &[ElementKind::F32],
        input(0),
        ReduceOp::Sum,
        AxisMask(0b1),
        false,
    );
    let a = OperandDesc::new(2, &[N, COLS], &[COLS, 1], ElementKind::F32, 256);
    let o = OperandDesc::new(1, &[COLS], &[1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);

    let variants = generate_variants(&op, &key, &Cuda);
    let base = variants.iter().find(|v| v.tag == "base").expect("base");
    let prec = variants
        .iter()
        .find(|v| v.tag == "prec")
        .expect("precision-first variant offered for outer-axis f32 Sum");
    assert_eq!(prec.fidelity, VariantFidelity::MorePrecise);
    assert_eq!(prec.fidelity.determinism_str(), "bitwise");

    // nvrtc-compile base + prec; load each via the driver.
    let compiler = NvrtcCompiler::new(ArchSku::Sm89);
    let mut modules = Vec::new();
    for v in [base, prec] {
        let k = &v.kernels[0];
        let ptx = compiler
            .compile(&k.source, &k.name, 30_000)
            .unwrap_or_else(|e| panic!("nvrtc({}) failed: {e}", k.name));
        let ptx = String::from_utf8(ptx).expect("ptx text");
        modules.push((k.name.clone(), Module::load_ptx(&ctx, &ptx).expect("load")));
    }
    let func = |name: &str| {
        let (_, m) = modules.iter().find(|(n, _)| n == name).unwrap();
        m.get_function(name).expect("get_function")
    };
    let f_base = func(&base.kernels[0].name);
    let f_prec = func(&prec.kernels[0].name);

    // All ones → exact column sum = N.
    let host = vec![1.0f32; (N * COLS) as usize];
    let d_in = DeviceBuffer::from_slice(&ctx, &host).expect("upload");
    let d_out = DeviceBuffer::<f32>::new(&ctx, COLS as usize).expect("out");
    let oracle = N as f64;

    // General-path ABI: (in0, out, shape0=N, shape1=COLS, s0_0=COLS, s0_1=1,
    // so_0=1, n_out=COLS). Base and prec share it (prec is the same general path
    // in double). One thread per output column.
    let run = |name: &str, f: &baracuda_driver::Function| -> Vec<f32> {
        let (shape0, shape1, s0_0, s0_1, so_0, n_out) = (N, COLS, COLS, 1i64, 1i64, COLS);
        // SAFETY: matches the general-path signature; buffers outlive the launch.
        unsafe {
            f.launch()
                .grid(1u32)
                .block(256u32)
                .stream(&stream)
                .arg(&d_in)
                .arg(&d_out)
                .arg(&shape0)
                .arg(&shape1)
                .arg(&s0_0)
                .arg(&s0_1)
                .arg(&so_0)
                .arg(&n_out)
                .launch()
                .unwrap_or_else(|e| panic!("{name} launch: {e}"));
        }
        stream.synchronize().expect("sync");
        let mut got = vec![0f32; COLS as usize];
        d_out.copy_to_host(&mut got).expect("download");
        got
    };
    let base_out = run("base", &f_base);
    let prec_out = run("prec", &f_prec);

    let relerr = |g: f32| ((g as f64) - oracle).abs() / oracle;
    for c in 0..COLS as usize {
        let (rb, rp) = (relerr(base_out[c]), relerr(prec_out[c]));
        println!(
            "col {c}: base={} (relerr {rb:.4})  prec={} (relerr {rp:.2e})  oracle={oracle}",
            base_out[c], prec_out[c]
        );
        // The base float-serial fold saturated at 2^24; the prec double fold is exact.
        assert!(
            (base_out[c] as f64 - SAT).abs() < 2.0,
            "base float-serial must saturate at 2^24, got {}",
            base_out[c]
        );
        assert!(
            rp < 1e-6,
            "prec (double serial) must be ~exact: relerr {rp}"
        );
        assert!(
            rb > 0.05,
            "base (float serial) must be far off: relerr {rb}"
        );
        assert!(
            rp < rb,
            "prec must be strictly MORE ACCURATE than base (rp {rp} < rb {rb})"
        );
    }
    println!(
        "PROVED: precision-first reduction is more accurate — base saturates at 2^24 ({SAT}), \
         prec (double serial) == oracle ({oracle}). MorePrecise = not just different."
    );
}
