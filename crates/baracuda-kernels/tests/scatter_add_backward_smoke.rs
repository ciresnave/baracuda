//! Real-GPU smoke test documenting the `scatter_add` backward routing.
//!
//! `scatter_add` BW reuses the FW `gather` kernel — `dupdates = gather(dout,
//! scatter_dim, index)`. There is no dedicated `ScatterAddBackwardPlan`;
//! callers wire through `GatherPlan` directly. This test verifies that
//! the natural routing produces the correct adjoint for a small fixture.
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, GatherArgs, GatherDescriptor, GatherPlan, PlanPreference,
    TensorMut, TensorRef, Workspace,
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
fn scatter_add_backward_via_gather_f32_2d_dim1() {
    let (ctx, stream) = setup();
    // FW: scatter_add(out: [3, 6], dim=1, idx: [3, 4], updates: [3, 4]).
    // BW: dupdates = gather(dout, dim=1, idx).
    let out_shape = [3i32, 6];
    let upd_shape = [3i32, 4];
    let host_idx: Vec<i32> = vec![0, 1, 1, 5,  2, 0, 3, 4,  5, 5, 5, 0];
    // dout is the upstream gradient w.r.t. the FW scatter_add output.
    let host_dout: Vec<f32> =
        (0..(3 * 6)).map(|i| (i as f32) * 0.5 - 4.0).collect();
    // Reference: dupdates[i, j] = dout[i, idx[i, j]].
    let mut expected = vec![0f32; 3 * 4];
    for i in 0..3usize {
        for j in 0..4usize {
            let k = host_idx[i * 4 + j] as usize;
            expected[i * 4 + j] = host_dout[i * 6 + k];
        }
    }

    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("up dout");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_dupd: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, 3 * 4).expect("alloc dupd");

    let desc = GatherDescriptor {
        out_shape: upd_shape,
        gather_dim: 1,
        src_dim_size: 6,
        element: ElementKind::F32,
    };
    let plan = GatherPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select gather (acts as scatter_add BW)");
    let args = GatherArgs::<f32, 2> {
        src: TensorRef {
            data: dev_dout.as_slice(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
        index: TensorRef {
            data: dev_idx.as_slice(),
            shape: upd_shape,
            stride: contiguous_stride(upd_shape),
        },
        out: TensorMut {
            data: dev_dupd.as_slice_mut(),
            shape: upd_shape,
            stride: contiguous_stride(upd_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 3 * 4];
    dev_dupd.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "scatter_add BW via gather mismatch @ {i}: got {g} expected {e}"
        );
    }
}
