//! Device check of `extern_asm_cuda` (ROADMAP U11 step 3, `docs/u11-ptx-leg-design.md`).
//!
//! Each case declares an Unpopped `ExternDecl` with a `cuda-ptx` asm body and a plain `reference` kernel. The
//! oracle runs a module that calls the declaration per element (it runs the reference, never the asm). The device
//! runs a `.cu` kernel whose one asm statement is spelled by `extern_asm_cuda`, compiled with NVRTC. The two outputs are
//! compared bit for bit (every NaN counts as one value: the payload a NaN result carries is not specified here).
//!
//! Cases: `fma.rn.f32` (reference: `ScalarExpr::Fma`) and `popc.b32` (reference: 32 shift-and-mask terms summed).
//!
//! Run with `cargo gpu-test -p baracuda-cuda-emit --features extern_asm_cuda,nvrtc,device-tests --test extern_ptx`.
#![cfg(all(
    feature = "extern_asm_cuda",
    feature = "nvrtc",
    feature = "device-tests"
))]

use baracuda_cuda_emit::{NvrtcCompiler, extern_asm_cuda};
use baracuda_driver::{Context, Device, DeviceBuffer, Module as DevModule, Stream, init};
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::jit::Compiler;
use unpopped::kernel::{
    Block, BodyText, BufferRole, Effects, ExternDecl, KernelFn, LocalDecl, Module, Param, Place,
    Stmt, TargetBody,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_module;
use unpopped_vocab::{ArchSku, ElementKind};

const F32: ElementKind = ElementKind::F32;
const U32: ElementKind = ElementKind::U32;
/// Oracle threads per launch (the oracle runs one frame stack per thread).
const CHUNK: usize = 256;

fn cu(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn bin(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}

fn c_type(k: ElementKind) -> &'static str {
    if k == F32 { "float" } else { "unsigned int" }
}

/// One declared instruction and everything the two legs need.
struct Case {
    tag: &'static str,
    decl: ExternDecl,
    /// The reference kernel (a plain function with the declared signature).
    reference: KernelFn,
    /// One column of raw 32-bit words per parameter, all the same length.
    inputs: Vec<Vec<u32>>,
}

fn asm_body(template: &str, constraints: &str) -> TargetBody {
    TargetBody::new(
        "cuda-ptx",
        BodyText::Asm {
            template: template.into(),
            constraints: constraints.into(),
        },
    )
}

/// `r = fma(x, y, z)` with one rounding.
fn fma_case(inputs: Vec<Vec<u32>>) -> Case {
    let scalar = |n: &str, i| Param::scalar(n, F32, i);
    let reference = KernelFn::new(
        "ref_fma",
        F32,
        vec![scalar("x", 0), scalar("y", 1), scalar("z", 2)],
        (0..4).map(|_| LocalDecl::scalar(F32)).collect(),
        Block(vec![Stmt::Let {
            local: 3,
            init: ScalarExpr::Fma(Box::new(local(0)), Box::new(local(1)), Box::new(local(2))),
        }]),
    )
    .with_result(3);
    Case {
        tag: "fma.rn.f32",
        decl: ExternDecl::new("fma", vec![F32; 3], vec![F32], Effects::pure())
            .with_body(asm_body("fma.rn.f32 {3}, {0}, {1}, {2};", "=f,f,f,f")),
        reference,
        inputs,
    }
}

/// `r = popcount(x)`: the sum of `(x >> i) & 1` over the 32 bit positions.
fn popc_case(inputs: Vec<Vec<u32>>) -> Case {
    let term = |i: i64| bin(BinaryOp::BitAnd, bin(BinaryOp::Shr, local(0), cu(i)), cu(1));
    let sum = (1..32).fold(term(0), |acc, i| {
        ScalarExpr::Add(Box::new(acc), Box::new(term(i)))
    });
    let reference = KernelFn::new(
        "ref_popc",
        U32,
        vec![Param::scalar("x", U32, 0)],
        vec![LocalDecl::scalar(U32), LocalDecl::scalar(U32)],
        Block(vec![Stmt::Let {
            local: 1,
            init: sum,
        }]),
    )
    .with_result(1);
    Case {
        tag: "popc.b32",
        decl: ExternDecl::new("popc", vec![U32], vec![U32], Effects::opaque())
            .with_body(asm_body("popc.b32 {1}, {0};", "=r,r")),
        reference,
        inputs,
    }
}

fn slot(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}

/// The module the oracle runs: `main(in0.., out)` loads each input at `tid`, calls the declaration, stores the result.
fn oracle_module(c: &Case) -> Module {
    let (np, ty) = (c.decl.params.len() as u32, c.decl.params[0]);
    let mut params: Vec<Param> = c
        .decl
        .params
        .iter()
        .enumerate()
        .map(|(i, k)| Param::buffer(format!("in{i}"), *k, BufferRole::Read, Some(1)))
        .collect();
    params.push(Param::buffer(
        "out",
        c.decl.results[0],
        BufferRole::Write,
        Some(1),
    ));
    let mut body: Vec<Stmt> = (0..np)
        .map(|i| Stmt::Load {
            local: i,
            place: slot(i),
        })
        .collect();
    body.push(Stmt::ExternCall {
        decl: 0,
        args: (0..np).map(local).collect(),
        results: vec![np],
    });
    body.push(Stmt::Store {
        place: slot(np),
        value: local(np),
    });
    let locals: Vec<LocalDecl> = (0..=np)
        .map(|i| LocalDecl::scalar(if i < np { ty } else { c.decl.results[0] }))
        .collect();
    let main = KernelFn::new("main", ty, params, locals, Block(body))
        .with_launch(LaunchShape::new([CHUNK as u32, 1, 1], [1, 1, 1]));
    Module::new(vec![main, c.reference.clone()])
        .with_externs(vec![c.decl.clone().with_reference(1)])
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn buf(kind: ElementKind, w: &[u32]) -> KernelArg {
    let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    KernelArg::Buffer(TypedBuffer::new(kind, vec![w.len() as i64], vec![1], bytes))
}

/// The oracle's output words, `CHUNK` elements per launch.
fn oracle_output(c: &Case) -> Vec<u32> {
    let m = oracle_module(c);
    let n = c.inputs[0].len();
    let mut out = Vec::with_capacity(n);
    for start in (0..n).step_by(CHUNK) {
        let end = (start + CHUNK).min(n);
        // A partial last chunk is padded with zeros so every thread of the launch has a slot.
        let pad = |col: &Vec<u32>| {
            let mut w = col[start..end].to_vec();
            w.resize(CHUNK, 0);
            w
        };
        let mut args: Vec<KernelArg> = c
            .decl
            .params
            .iter()
            .zip(&c.inputs)
            .map(|(k, col)| buf(*k, &pad(col)))
            .collect();
        args.push(buf(c.decl.results[0], &vec![0; CHUNK]));
        run_launch_module(&m, 0, &mut args).unwrap_or_else(|e| panic!("{}: oracle: {e:?}", c.tag));
        let KernelArg::Buffer(o) = args.last().unwrap() else {
            unreachable!("a buffer")
        };
        out.extend(&words(&o.bytes)[..end - start]);
    }
    out
}

/// The `.cu` kernel: `ext_kernel(const T0* in0.., R* out, long long n)`, with the asm statement from `extern_asm_cuda`.
fn device_source(c: &Case) -> String {
    let np = c.decl.params.len();
    let ins: Vec<String> = (0..np)
        .map(|i| format!("const {}* in{i}", c_type(c.decl.params[i])))
        .collect();
    let args: Vec<String> = (0..np).map(|i| format!("x{i}")).collect();
    let stmt = extern_asm_cuda(&c.decl, &args, &["r".to_string()]).expect("spell the asm body");
    let loads: String = (0..np)
        .map(|i| format!("    {} x{i} = in{i}[i];\n", c_type(c.decl.params[i])))
        .collect();
    format!(
        "extern \"C\" __global__ void ext_kernel({}, {}* out, long long n) {{\n    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n    if (i >= n) return;\n{loads}    {} r;\n    {stmt}\n    out[i] = r;\n}}\n",
        ins.join(", "),
        c_type(c.decl.results[0]),
        c_type(c.decl.results[0]),
    )
}

fn device_output(ctx: &Context, stream: &Stream, cc: &NvrtcCompiler, c: &Case) -> Vec<u32> {
    let src = device_source(c);
    let ptx = cc
        .compile(&src, "ext_kernel", 5000)
        .unwrap_or_else(|e| panic!("{}: nvrtc: {e}\n{src}", c.tag));
    let ptx = String::from_utf8(ptx).expect("ptx is utf8");
    let module = DevModule::load_ptx(ctx, &ptx).expect("load ptx");
    let func = module.get_function("ext_kernel").expect("function");
    let n = c.inputs[0].len();
    let d_in: Vec<DeviceBuffer<u32>> = c
        .inputs
        .iter()
        .map(|col| DeviceBuffer::from_slice(ctx, col).expect("upload"))
        .collect();
    let d_out: DeviceBuffer<u32> = DeviceBuffer::zeros(ctx, n).expect("alloc out");
    let ptrs: Vec<_> = d_in.iter().map(DeviceBuffer::as_raw).collect();
    let out_ptr = d_out.as_raw();
    let n64 = n as i64;
    // SAFETY: the kernel is `(const T* in0.., R* out, long long n)`; arguments are pushed in that order with those
    // types, each buffer holds `n` 4-byte words and outlives the launch (the stream is synchronised below).
    unsafe {
        let mut l = func
            .launch()
            .grid(((n as u32).div_ceil(256), 1, 1))
            .block((256, 1, 1))
            .stream(stream);
        for p in &ptrs {
            l = l.arg(p);
        }
        l.arg(&out_ptr).arg(&n64).launch().expect("launch");
    }
    stream.synchronize().expect("sync");
    let mut got = vec![0u32; n];
    d_out.copy_to_host(&mut got).expect("download");
    got
}

/// Equal bits, or both NaN.
fn same(kind: ElementKind, a: u32, b: u32) -> bool {
    a == b || (kind == F32 && f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan())
}

fn check(case: &Case) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let cc = NvrtcCompiler::new(ArchSku::Sm89.into()).expect("Sm89 is a well-formed cuda: target");
    let want = oracle_output(case);
    let got = device_output(&ctx, &stream, &cc, case);
    let kind = case.decl.results[0];
    let bad: Vec<String> = (0..want.len())
        .filter(|&i| !same(kind, got[i], want[i]))
        .take(10)
        .map(|i| {
            let xs: Vec<String> = case
                .inputs
                .iter()
                .map(|c| format!("{:#010x}", c[i]))
                .collect();
            format!(
                "  {} x={xs:?}: device {:#010x}, oracle {:#010x}",
                case.tag, got[i], want[i]
            )
        })
        .collect();
    assert!(
        bad.is_empty(),
        "{}: {} of {} elements differ; first:\n{}",
        case.tag,
        (0..want.len())
            .filter(|&i| !same(kind, got[i], want[i]))
            .count(),
        want.len(),
        bad.join("\n")
    );
    eprintln!("{}: {} elements, device == oracle", case.tag, want.len());
}

/// A fixed-seed xorshift, so a failure reproduces.
fn randoms(n: usize, mut s: u32) -> Vec<u32> {
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        })
        .collect()
}

#[test]
fn popc_b32_matches_its_reference() {
    let mut x = vec![
        0,
        1,
        u32::MAX,
        0x8000_0000,
        0x7fff_ffff,
        0xaaaa_aaaa,
        0x5555_5555,
        0x0000_ffff,
        0xffff_0000,
    ];
    x.extend(randoms(1000, 0x9e37_79b9));
    check(&popc_case(vec![x]));
}

#[test]
fn fma_rn_f32_matches_its_reference() {
    // Edge values: zeros, ones, subnormals, the smallest normal, the largest finite, infinities, NaNs (one with a payload).
    let edge: [u32; 16] = [
        0x0000_0000,
        0x8000_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x0000_0001,
        0x8000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0x7fa0_0001,
        0x3f00_0001,
        0x4b00_0001,
    ];
    let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
    for &x in &edge {
        for &y in &edge {
            for &z in &edge {
                a.push(x);
                b.push(y);
                c.push(z);
            }
        }
    }
    // Random finite-ish operands, where a separate multiply and add rounds twice and disagrees with a fused one.
    let n = 2000;
    a.extend(randoms(n, 0x1234_5679));
    b.extend(randoms(n, 0x2545_f491));
    c.extend(randoms(n, 0x6c07_8965));
    check(&fma_case(vec![a, b, c]));
}
