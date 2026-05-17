//! Real-GPU smoke test for `QuantizePerChannelPlan<TIn, TOut>` —
//! Phase 8 Milestone 8.1.
//!
//! Verifies a rank-3 input with per-channel quantization along axis 1
//! (the typical per-output-channel weight quant pattern). The input
//! tensor is shape `[B, C, D]`; the caller pads to rank 4 with a
//! trailing `1` extent.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, QuantizePerChannelArgs,
    QuantizePerChannelDescriptor, QuantizePerChannelPlan, S8, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn quantize_per_channel_f32_s8_rank3_axis1() {
    let (ctx, stream) = setup();
    // Logical shape [B=2, C=3, D=4]. Pad to rank-4 with trailing 1.
    let shape4: [i32; 4] = [2, 3, 4, 1];
    let rank: u8 = 3;
    let axis: u8 = 1; // C
    let c: usize = 3;
    let numel: usize = 2 * 3 * 4;

    // Per-channel scales / zps.
    let host_scale: Vec<f32> = vec![0.1, 0.2, 0.05];
    let host_zp: Vec<i32> = vec![0, 5, -10];

    // Build an input pattern that, when quantized with the above
    // per-channel scales / zps, yields a known reference.
    //
    // x[b, c, d] = (c+1) * (d + 0.5) / 10.0
    let mut host_x: Vec<f32> = vec![0.0; numel];
    for b in 0..2usize {
        for ci in 0..c {
            for d in 0..4usize {
                let idx = (b * 3 + ci) * 4 + d;
                host_x[idx] = ((ci + 1) as f32) * ((d as f32 + 0.5) / 10.0);
            }
        }
    }

    // CPU reference using round-half-to-even semantics (matching
    // `__float2int_rn` on the device).
    let mut host_expected: Vec<i8> = vec![0; numel];
    for b in 0..2usize {
        for ci in 0..c {
            let s = host_scale[ci];
            let zp = host_zp[ci];
            for d in 0..4usize {
                let idx = (b * 3 + ci) * 4 + d;
                let r = (host_x[idx] / s).round_ties_even() as i32 + zp;
                let clipped = r.clamp(-128, 127);
                host_expected[idx] = clipped as i8;
            }
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_scale = DeviceBuffer::from_slice(&ctx, &host_scale).expect("up scale");
    let dev_zp = DeviceBuffer::from_slice(&ctx, &host_zp).expect("up zp");
    let mut dev_q: DeviceBuffer<S8> = DeviceBuffer::zeros(&ctx, numel).expect("alloc q");

    let desc = QuantizePerChannelDescriptor {
        shape: shape4,
        rank,
        axis,
        q_min: -128,
        q_max: 127,
        input_element: ElementKind::F32,
        output_element: ElementKind::S8,
    };
    let plan = QuantizePerChannelPlan::<f32, S8>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .expect("select");
    let args = QuantizePerChannelArgs::<f32, S8> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: shape4,
            stride: contiguous_stride(shape4),
        },
        scale: TensorRef {
            data: dev_scale.as_slice(),
            shape: [c as i32],
            stride: contiguous_stride([c as i32]),
        },
        zero_point: TensorRef {
            data: dev_zp.as_slice(),
            shape: [c as i32],
            stride: contiguous_stride([c as i32]),
        },
        output: TensorMut {
            data: dev_q.as_slice_mut(),
            shape: shape4,
            stride: contiguous_stride(shape4),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_storage = vec![S8(0); numel];
    dev_q.copy_to_host(&mut got_storage).expect("dl");
    let got: Vec<i8> = got_storage.iter().map(|s| s.0).collect();
    assert_eq!(got, host_expected, "per-channel quant f32→s8 mismatch");
}
