//! Real-GPU smoke test for `TopkPlan<T>` (Phase 9 Category O).
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, TopkArgs,
    TopkDescriptor, TopkPlan, Workspace,
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
fn topk_f32_largest_basic() {
    let (ctx, stream) = setup();
    let batch: i32 = 2;
    let row_len: i32 = 16;
    let k: i32 = 4;
    let input: Vec<f32> = (0..(batch * row_len) as usize)
        .map(|i| {
            // pseudo-random-ish but deterministic
            let x = (i as f32) * 1.7;
            (x.sin() * 10.0).round()
        })
        .collect();
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_vals: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (batch * k) as usize).expect("alloc v");
    let mut dev_idx: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, (batch * k) as usize).expect("alloc i");

    let desc = TopkDescriptor {
        batch,
        row_len,
        k,
        largest: true,
        element: ElementKind::F32,
    };
    let plan = TopkPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("select");
    plan.run(
        &stream,
        Workspace::None,
        TopkArgs::<f32> {
            input: TensorRef {
                data: dev_in.as_slice(),
                shape: [batch, row_len],
                stride: contiguous_stride([batch, row_len]),
            },
            values: TensorMut {
                data: dev_vals.as_slice_mut(),
                shape: [batch, k],
                stride: contiguous_stride([batch, k]),
            },
            indices: TensorMut {
                data: dev_idx.as_slice_mut(),
                shape: [batch, k],
                stride: contiguous_stride([batch, k]),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got_v = vec![0f32; (batch * k) as usize];
    dev_vals.copy_to_host(&mut got_v).expect("dl v");

    for row in 0..batch as usize {
        let off_in = row * row_len as usize;
        let off_out = row * k as usize;
        let mut sorted: Vec<f32> = input[off_in..off_in + row_len as usize].to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        for i in 0..k as usize {
            assert!(
                (got_v[off_out + i] - sorted[i]).abs() < 1e-6,
                "topk largest row {row} cell {i}: got {} expected {}",
                got_v[off_out + i],
                sorted[i]
            );
        }
    }
}
