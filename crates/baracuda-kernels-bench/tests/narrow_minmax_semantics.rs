//! The two CUDA semantics the §6.16-0009 bit-move path for `RowReduce` rests
//! on, measured on an RTX 4070 rather than read out of the docs.
//!
//! # 1. The block reducer's NaN test SURVIVES a narrow accumulator
//!
//! `emit_block_reducers` spells its Max/Min combine as
//!
//! ```c
//! if (oh && (!has || ov != ov || ov > v)) { v = ov; has = 1; }
//! ```
//!
//! `ov != ov` is the NaN test, and the house convention is NaN-PROPAGATING
//! Max/Min. Wiring the bit-move path substitutes `__nv_bfloat16` / `__half`
//! for `{acc}` in that same string, where `!=` is an operator overload rather
//! than IEEE `!=`.
//!
//! ⚠️ I EXPECTED THAT TO BREAK IT — that `!=` lowers to `__hne`, which the CUDA
//! docs describe as producing false for NaN, silently deleting NaN propagation.
//! **MEASURED: it does not break. `ov != ov` returns 1 for bf16 and for f16**,
//! matching the float reference. So the substitution is safe and the reducers
//! need no narrow variant. The hypothesis was wrong and the test is kept
//! because a future reader will have exactly the same worry.
//!
//! # 2. The float round trip DESTROYS the NaN payload — which is the whole point
//!
//! ```text
//!            bit copy      via float
//!   bf16     0x7FC1        0x7FFF      source 0x7FC1
//!   f16      0x7E01        0x7FFF      source 0x7E01
//! ```
//!
//! ⚠️ **That is KISS-355 measured directly.** The RowReduce path today loads
//! with `__bfloat162float(in0[..])` and narrows at the store, so an all-move
//! Max fold canonicalizes any non-default NaN payload to `0x7FFF`. §6.16-0009
//! forbids exactly that round trip, and this is the evidence for why the
//! bit-move wiring is worth doing rather than a citation of a clause.
//!
//! # Sibling test, different question
//!
//! `max_nan_propagation.rs` asks whether the f32 NaN-propagating ternary
//! SURVIVES the compile chain — whether ptxas or the driver JIT contracts it
//! back into a NaN-suppressing hardware `max.f32`. That is a question about a
//! layer BELOW the emitted source.
//!
//! This asks a different one: whether the same source spelling still MEANS
//! NaN-propagation once `{acc}` is a narrow type, and what the promoted path
//! costs. Neither subsumes the other — a reader arriving at either should know
//! the other exists, because "NaN handling is already covered on device" is
//! true of both and answers neither.
//!
//! `#[ignore]`d — needs a CUDA device + nvrtc. Run with:
//! `cargo test -p baracuda-kernels-bench --test narrow_minmax_semantics -- --ignored`

use baracuda_cuda_emit::{Cuda as _Cuda, NvrtcCompiler};
use baracuda_driver::DeviceBuffer;
use baracuda_driver::{Context, Device, Module, Stream, require_optional};
use baracuda_kernels_bench::{current_hwstamp, setup_device};
use baracuda_kernels_types::ArchSku;
use unpopped::Compiler;

const _: Option<_Cuda> = None; // keep the emitter dep honest in this target

/// One probe per (type, spelling). Each writes 1 if the NaN branch fired.
const SRC: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// A NaN with a NON-CANONICAL payload, so a "quieted to 0x7FFF" result is
// distinguishable from a preserved one.
__device__ __forceinline__ __nv_bfloat16 bf16_nan_payload() {
    unsigned short bits = 0x7FC1u;          // qNaN, payload 0x41
    __nv_bfloat16 h; memcpy(&h, &bits, 2); return h;
}
__device__ __forceinline__ __half f16_nan_payload() {
    unsigned short bits = 0x7E01u;          // qNaN, payload 0x01
    __half h; memcpy(&h, &bits, 2); return h;
}

extern "C" __global__ void probe(int* out, unsigned short* bits) {
    // [0] float   `ov != ov`   -- the reference: IEEE unordered, must be 1
    { float ov = __int_as_float(0x7FC00001); out[0] = (ov != ov) ? 1 : 0; }

    // [1] bf16    `ov != ov`   -- THE QUESTION
    { __nv_bfloat16 ov = bf16_nan_payload(); out[1] = (ov != ov) ? 1 : 0; }

    // [2] f16     `ov != ov`   -- THE QUESTION
    { __half ov = f16_nan_payload(); out[2] = (ov != ov) ? 1 : 0; }

    // [3] bf16    `__hisnan`   -- the candidate replacement
    { __nv_bfloat16 ov = bf16_nan_payload(); out[3] = __hisnan(ov) ? 1 : 0; }

    // [4] f16     `__hisnan`   -- the candidate replacement
    { __half ov = f16_nan_payload(); out[4] = __hisnan(ov) ? 1 : 0; }

    // [5] bf16    `ov > v` with ov NaN -- must be 0 (ordered compare, NaN false)
    { __nv_bfloat16 ov = bf16_nan_payload(); __nv_bfloat16 v = __float2bfloat16(1.0f);
      out[5] = (ov > v) ? 1 : 0; }

    // Payload survival through a pure bit copy in the narrow type: this is what
    // the bit-move path relies on, and it is the half that must hold for the
    // whole exercise to be worth doing.
    { __nv_bfloat16 ov = bf16_nan_payload(); __nv_bfloat16 v = ov;
      unsigned short b; memcpy(&b, &v, 2); bits[0] = b; }
    { __half ov = f16_nan_payload(); __half v = ov;
      unsigned short b; memcpy(&b, &v, 2); bits[1] = b; }
    // And through a float round trip -- the path the CURRENT promoted reducer
    // takes. If this differs from the above, the round trip is the payload loss.
    { __nv_bfloat16 ov = bf16_nan_payload();
      float f = __bfloat162float(ov); __nv_bfloat16 v = __float2bfloat16(f);
      unsigned short b; memcpy(&b, &v, 2); bits[2] = b; }
    { __half ov = f16_nan_payload();
      float f = __half2float(ov); __half v = __float2half(f);
      unsigned short b; memcpy(&b, &v, 2); bits[3] = b; }
}
"#;

/// Compile and run the probe; return `(flags, bits)`.
///
/// Split out so the test body is only ASSERTIONS. Getting a measurement and
/// deciding what it means are different jobs, and interleaving them is how a
/// setup edit silently changes a conclusion.
fn probe_device(ctx: &Context, stream: &Stream) -> (Vec<i32>, Vec<u16>) {
    let compiler = NvrtcCompiler::new(ArchSku::Sm89);
    let ptx = compiler
        .compile(SRC, "probe", 30_000)
        .unwrap_or_else(|e| panic!("nvrtc failed: {e}"));
    let ptx = String::from_utf8(ptx).expect("ptx is text");
    let module = Module::load_ptx(ctx, &ptx).expect("module load");
    let f = module.get_function("probe").expect("get_function");

    let d_out = DeviceBuffer::<i32>::new(ctx, 6).expect("d_out");
    let d_bits = DeviceBuffer::<u16>::new(ctx, 4).expect("d_bits");
    // SAFETY: signature is (int*, unsigned short*); both buffers outlive the launch.
    unsafe {
        f.launch()
            .grid(1u32)
            .block(1u32)
            .stream(stream)
            .arg(&d_out)
            .arg(&d_bits)
            .launch()
            .expect("launch");
    }
    stream.synchronize().expect("sync");
    let mut out = vec![0i32; 6];
    let mut bits = vec![0u16; 4];
    d_out.copy_to_host(&mut out).expect("copy out");
    d_bits.copy_to_host(&mut bits).expect("copy bits");
    (out, bits)
}

#[test]
#[ignore = "requires a CUDA device + nvrtc"]
fn narrow_nan_test_and_payload_survival() {
    let _serial = baracuda_kernels_bench::DEVICE_TIMING_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (ctx, stream) = setup_device();
    let device = Device::get(0).expect("device");
    let stamp = current_hwstamp(&device).expect("hwstamp");
    // ⚠️ THE GATE LIVES HERE, NOT IN THE PROBE. `require_optional!` logs the
    // decline and returns, which needs a `-> ()` body. A first draft moved it
    // into `probe_device` and returned neutral data instead — and the control
    // assertion REJECTS that data, so an honest skip on legitimate sm80
    // hardware would have become a FAILURE. A silent skip and a false failure
    // are both instrument defects; this file is about not shipping either.
    require_optional!(
        (stamp.target == ArchSku::Sm89.into()).then_some(()),
        "an sm89 device (this probe compiles sm89 PTX)"
    );
    let (out, bits) = probe_device(&ctx, &stream);
    report(&out, &bits);

    // THE CONTROL. If float `!=` does not detect NaN, the probe itself is broken
    // and nothing else here means anything.
    assert_eq!(
        out[0], 1,
        "control failed: float `ov != ov` did not detect a NaN, so this probe is not measuring what it claims"
    );

    // A bit copy in the narrow type must preserve the payload — this is the
    // premise of the whole bit-move path.
    assert_eq!(
        bits[0], 0x7FC1,
        "bf16 bit copy did not preserve the NaN payload"
    );
    assert_eq!(
        bits[1], 0x7E01,
        "f16 bit copy did not preserve the NaN payload"
    );

    // FINDING 1 — the narrow `!=` DOES detect NaN, so `emit_block_reducers`
    // needs no narrow variant. Pinned so a CUDA release that changes it is a
    // red test here rather than a silent loss of NaN propagation in every
    // narrow Max/Min fold.
    assert_eq!(
        out[1], 1,
        "bf16 `ov != ov` no longer detects NaN — the block reducer's NaN test would be dead under a narrow accumulator"
    );
    assert_eq!(out[2], 1, "f16 `ov != ov` no longer detects NaN — same");
    assert_eq!(
        out[5], 0,
        "an ordered `>` against a NaN must be false, or the NaN branch is unreachable"
    );

    // `__hisnan` also works — kept as the fallback spelling if 1 ever flips.
    assert_eq!(out[3], 1, "bf16 __hisnan failed to detect a NaN");
    assert_eq!(out[4], 1, "f16 __hisnan failed to detect a NaN");

    // ⚠️ FINDING 2 — the motivation, asserted rather than cited. The float round
    // trip the CURRENT promoted path takes canonicalizes the payload; the bit
    // copy the bit-move path takes preserves it. If these ever agree, either
    // the hardware changed or the probe stopped measuring the round trip, and
    // §6.16-0009's cost here would be zero.
    assert_eq!(
        bits[2], 0x7FFF,
        "bf16 float round trip no longer canonicalizes the payload"
    );
    assert_eq!(
        bits[3], 0x7FFF,
        "f16 float round trip no longer canonicalizes the payload"
    );
    assert_ne!(
        bits[0], bits[2],
        "bit copy and float round trip agree — the payload loss §6.16-0009 forbids is not reproducible here, so the bit-move path would buy nothing"
    );
}

/// Print the measurement before interpreting it. Separate from the
/// assertions on purpose: these lines are the RECORD, and a reader who
/// distrusts a conclusion needs the numbers it was drawn from.
fn report(out: &[i32], bits: &[u16]) {
    eprintln!(
        "  [0] float  `ov != ov`   = {}   (reference: IEEE unordered)",
        out[0]
    );
    eprintln!("  [1] bf16   `ov != ov`   = {}", out[1]);
    eprintln!("  [2] f16    `ov != ov`   = {}", out[2]);
    eprintln!("  [3] bf16   `__hisnan`   = {}", out[3]);
    eprintln!("  [4] f16    `__hisnan`   = {}", out[4]);
    eprintln!(
        "  [5] bf16   NaN > 1.0    = {}   (ordered compare, expect 0)",
        out[5]
    );
    eprintln!(
        "  bf16 payload: bit-copy 0x{:04X}  float-round-trip 0x{:04X}  (source 0x7FC1)",
        bits[0], bits[2]
    );
    eprintln!(
        "  f16  payload: bit-copy 0x{:04X}  float-round-trip 0x{:04X}  (source 0x7E01)",
        bits[1], bits[3]
    );
}
