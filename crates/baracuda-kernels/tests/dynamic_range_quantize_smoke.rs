//! Real-GPU smoke test for `DynamicRangeQuantizePlan<f32, S8>` (Phase 8.3).
//!
//! `[N=2, D=4]` fixture; verifies the symmetric per-token DRQ recipe:
//!   scale[n] = max_d |x[n, d]| / 127
//!   q[n, d]  = clamp(round(x[n, d] / scale[n]), -127, 127)
//!
//! Tolerances:
//!   - `scale[n]`         : exact f32 equality of `max_abs / qmax`.
//!   - `q[n, d]`          : within ±1 of the CPU reference, matching
//!     the existing int-quantize smoke convention (the kernel uses
//!     `__float2int_rn` for round-half-to-even; Rust `round_ties_even`
//!     in the CPU ref produces the same result up to f32 path
//!     associativity).
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, DynamicRangeMode, DynamicRangeQuantizeArgs,
    DynamicRangeQuantizeDescriptor, DynamicRangeQuantizePlan, DynamicRangeScope, ElementKind,
    PlanPreference, TensorMut, TensorRef, Workspace, S8,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn cpu_dynamic_range_quantize_per_token_symmetric(
    n: usize,
    d: usize,
    x: &[f32],
    qmax: i32,
    qmin: i32,
) -> (Vec<f32>, Vec<i8>) {
    let mut scale = vec![0f32; n];
    let mut q = vec![0i8; n * d];
    let qmaxf = qmax as f32;
    for row in 0..n {
        let mut max_abs = 0f32;
        for col in 0..d {
            let v = x[row * d + col].abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        let s = if max_abs > 0.0 {
            max_abs / qmaxf
        } else {
            1.0
        };
        scale[row] = s;
        let inv_s = if s != 0.0 { 1.0 / s } else { 0.0 };
        for col in 0..d {
            let v = x[row * d + col];
            let qf = (v * inv_s).round_ties_even() as i32;
            let qclip = qf.clamp(qmin, qmax);
            q[row * d + col] = qclip as i8;
        }
    }
    (scale, q)
}

#[test]
#[ignore]
fn dynamic_range_quantize_f32_s8_per_token_symmetric_basic() {
    let (ctx, stream) = setup();
    let n: i32 = 2;
    let d: i32 = 4;
    // Row 0: dense around small range; max_abs = 0.30 → scale ≈ 0.30/127.
    // Row 1: includes a large value 12.5 → scale ≈ 12.5/127 ≈ 0.0984.
    let host_x: Vec<f32> = vec![
        0.05, 0.15, -0.07, 0.30,
        12.5, 1.0, -0.25, 4.0,
    ];
    let qmin: i32 = -127;
    let qmax: i32 = 127;
    let (expected_scale, expected_q) =
        cpu_dynamic_range_quantize_per_token_symmetric(n as usize, d as usize, &host_x, qmax, qmin);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_scale: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n as usize).expect("alloc s");
    let mut dev_q: DeviceBuffer<S8> = DeviceBuffer::zeros(&ctx, (n * d) as usize).expect("alloc q");

    let desc = DynamicRangeQuantizeDescriptor {
        n,
        d,
        q_min: qmin,
        q_max: qmax,
        mode: DynamicRangeMode::Symmetric,
        scope: DynamicRangeScope::Token,
        input_element: ElementKind::F32,
        output_element: ElementKind::S8,
    };
    let plan =
        DynamicRangeQuantizePlan::<f32, S8>::select(&stream, &desc, PlanPreference::default())
            .expect("select");
    let args = DynamicRangeQuantizeArgs::<f32, S8> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [n, d],
            stride: contiguous_stride([n, d]),
        },
        scale_out: TensorMut {
            data: dev_scale.as_slice_mut(),
            shape: [n],
            stride: contiguous_stride([n]),
        },
        output: TensorMut {
            data: dev_q.as_slice_mut(),
            shape: [n, d],
            stride: contiguous_stride([n, d]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_scale = vec![0f32; n as usize];
    dev_scale.copy_to_host(&mut got_scale).expect("dl scale");
    let mut got_q = vec![S8(0); (n * d) as usize];
    dev_q.copy_to_host(&mut got_q).expect("dl q");

    // Scale is exact f32: kernel computes `max / qmax` identically to ref.
    for row in 0..(n as usize) {
        let g = got_scale[row];
        let e = expected_scale[row];
        let rel = if e != 0.0 { (g - e).abs() / e.abs() } else { (g - e).abs() };
        assert!(
            rel < 1e-6,
            "scale row {row} mismatch: got {g} expected {e} (rel {rel})"
        );
    }

    // Quantized output: ±1 tolerance to match the int-GEMM smoke convention.
    for (i, (g, e)) in got_q.iter().zip(expected_q.iter()).enumerate() {
        assert!(
            (g.0 as i32 - *e as i32).abs() <= 1,
            "q mismatch at idx {i}: got {} expected {}",
            g.0,
            e
        );
    }
}
