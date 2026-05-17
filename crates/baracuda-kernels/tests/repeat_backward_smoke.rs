//! Real-GPU smoke test for `RepeatBackwardPlan<T, N>` — backward of
//! `torch.repeat(x, *repeats)`. Each `dx[c_in]` is the sum of
//! `prod(repeats[d])` cells of `dy` (one per coord in the per-axis
//! repeats grid). f16 / bf16 accumulate in f32 inside the kernel; f32 /
//! f64 accumulate in their native dtype.
//!
//! Summation order matters in FP — compare with a weighted-relative-eps
//! bound `K * eps * sum_of_|summands|` (K = number of dy cells summed
//! for this dx cell = `prod(repeats[d])`). CPU reference accumulates in
//! f64 for headroom against the half / bf16 kernel's f32 acc.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test repeat_backward_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, RepeatBackwardArgs,
    RepeatBackwardDescriptor, RepeatBackwardPlan, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference: brute-force sum over the per-axis repeats grid. Accum
/// in f64 to guarantee headroom vs the GPU's f32 acc on half / bf16.
/// Returns (expected_values, per_cell_sum_of_absvals).
fn cpu_ref<const N: usize, F: Fn(usize) -> f64>(
    input_shape: [i32; N],
    repeats: [i32; N],
    dy_shape: [i32; N],
    dy_at: F,
) -> (Vec<f64>, Vec<f64>) {
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_stride = [1usize; N];
    for d in (0..N).rev().skip(1) {
        dy_stride[d] = dy_stride[d + 1] * dy_shape[d + 1] as usize;
    }
    let mut values = Vec::with_capacity(in_numel);
    let mut sum_abs = Vec::with_capacity(in_numel);
    let total_reps: i64 = repeats.iter().map(|&r| r as i64).product();
    for linear in 0..in_numel {
        let mut coord = [0i32; N];
        let mut rem = linear;
        for d in (0..N).rev() {
            coord[d] = (rem % input_shape[d] as usize) as i32;
            rem /= input_shape[d] as usize;
        }
        let mut acc = 0.0f64;
        let mut abs_acc = 0.0f64;
        for t in 0..total_reps {
            let mut tr = t;
            let mut dy_idx = 0usize;
            for d in (0..N).rev() {
                let r = repeats[d] as i64;
                let k = if r == 0 { 0 } else { tr % r };
                if r != 0 {
                    tr /= r;
                }
                let dy_coord = coord[d] as i64 + k * input_shape[d] as i64;
                dy_idx += (dy_coord as usize) * dy_stride[d];
            }
            let v = dy_at(dy_idx);
            acc += v;
            abs_acc += v.abs();
        }
        values.push(acc);
        sum_abs.push(abs_acc);
    }
    (values, sum_abs)
}

// -------- f32 --------

#[test]
#[ignore]
fn repeat_backward_f32_2d() {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 4];
    let repeats = [2i32, 3];
    let dy_shape = [input_shape[0] * repeats[0], input_shape[1] * repeats[1]];
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f32> = (0..dy_numel).map(|i| (i as f32) * 0.5 - 10.0).collect();
    let (exp_values, sum_abs) =
        cpu_ref(input_shape, repeats, dy_shape, |i| host_dy[i] as f64);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc");
    let desc = RepeatBackwardDescriptor {
        input_shape,
        repeats,
        element: ElementKind::F32,
    };
    let plan = RepeatBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = RepeatBackwardArgs::<f32, 2> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: dy_shape,
            stride: contiguous_stride(dy_shape),
        },
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    let k: f64 = (repeats[0] as f64) * (repeats[1] as f64);
    for (i, (&g, (&exp, &sabs))) in got.iter().zip(exp_values.iter().zip(sum_abs.iter())).enumerate() {
        let tol = (k * (f32::EPSILON as f64) * sabs.max(1.0)) as f32;
        let diff = (g - exp as f32).abs();
        assert!(
            diff <= tol,
            "f32 repeat BW @ {i}: got {g} exp {exp} diff {diff} tol {tol}"
        );
    }
}

// -------- f16 --------

#[test]
#[ignore]
fn repeat_backward_f16_3d() {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 3, 4];
    let repeats = [2i32, 1, 2];
    let dy_shape = [
        input_shape[0] * repeats[0],
        input_shape[1] * repeats[1],
        input_shape[2] * repeats[2],
    ];
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f16> = (0..dy_numel)
        .map(|i| f16::from_f32(((i as f32) * 0.25 - 5.0) * 0.1))
        .collect();
    let (exp_values, sum_abs) =
        cpu_ref(input_shape, repeats, dy_shape, |i| host_dy[i].to_f32() as f64);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc");
    let desc = RepeatBackwardDescriptor {
        input_shape,
        repeats,
        element: ElementKind::F16,
    };
    let plan = RepeatBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = RepeatBackwardArgs::<f16, 3> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: dy_shape,
            stride: contiguous_stride(dy_shape),
        },
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::ZERO; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    // f16 eps ~= 9.77e-4. K * eps * sum_abs, rounded once at store.
    let k: f64 = repeats.iter().map(|&r| r as f64).product();
    let f16_eps: f64 = 9.765625e-4;
    for (i, (&g, (&exp, &sabs))) in got.iter().zip(exp_values.iter().zip(sum_abs.iter())).enumerate() {
        let g32 = g.to_f32() as f64;
        // Allow at least one ULP at the magnitude of |exp| (half ULP at
        // store), plus the in-kernel accumulation error scaled by K.
        let tol = (k * f16_eps * sabs.max(1.0)).max(f16_eps * exp.abs().max(1.0));
        let diff = (g32 - exp).abs();
        assert!(
            diff <= tol,
            "f16 repeat BW @ {i}: got {g32} exp {exp} diff {diff} tol {tol}"
        );
    }
}

// -------- bf16 --------

#[test]
#[ignore]
fn repeat_backward_bf16_2d() {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 5];
    let repeats = [3i32, 2];
    let dy_shape = [input_shape[0] * repeats[0], input_shape[1] * repeats[1]];
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<bf16> = (0..dy_numel)
        .map(|i| bf16::from_f32(((i as f32) * 0.5 - 3.0) * 0.05))
        .collect();
    let (exp_values, sum_abs) =
        cpu_ref(input_shape, repeats, dy_shape, |i| host_dy[i].to_f32() as f64);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc");
    let desc = RepeatBackwardDescriptor {
        input_shape,
        repeats,
        element: ElementKind::Bf16,
    };
    let plan = RepeatBackwardPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = RepeatBackwardArgs::<bf16, 2> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: dy_shape,
            stride: contiguous_stride(dy_shape),
        },
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::ZERO; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    // bf16 eps ~= 7.81e-3.
    let k: f64 = repeats.iter().map(|&r| r as f64).product();
    let bf16_eps: f64 = 7.8125e-3;
    for (i, (&g, (&exp, &sabs))) in got.iter().zip(exp_values.iter().zip(sum_abs.iter())).enumerate() {
        let g32 = g.to_f32() as f64;
        let tol = (k * bf16_eps * sabs.max(1.0)).max(bf16_eps * exp.abs().max(1.0));
        let diff = (g32 - exp).abs();
        assert!(
            diff <= tol,
            "bf16 repeat BW @ {i}: got {g32} exp {exp} diff {diff} tol {tol}"
        );
    }
}

// -------- f64 --------

#[test]
#[ignore]
fn repeat_backward_f64_3d() {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 2, 3];
    let repeats = [3i32, 2, 2];
    let dy_shape = [
        input_shape[0] * repeats[0],
        input_shape[1] * repeats[1],
        input_shape[2] * repeats[2],
    ];
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f64> = (0..dy_numel).map(|i| (i as f64) * 0.125 - 1.0).collect();
    let (exp_values, sum_abs) =
        cpu_ref(input_shape, repeats, dy_shape, |i| host_dy[i]);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc");
    let desc = RepeatBackwardDescriptor {
        input_shape,
        repeats,
        element: ElementKind::F64,
    };
    let plan = RepeatBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = RepeatBackwardArgs::<f64, 3> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: dy_shape,
            stride: contiguous_stride(dy_shape),
        },
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    let k: f64 = repeats.iter().map(|&r| r as f64).product();
    for (i, (&g, (&exp, &sabs))) in got.iter().zip(exp_values.iter().zip(sum_abs.iter())).enumerate() {
        let tol = k * f64::EPSILON * sabs.max(1.0);
        let diff = (g - exp).abs();
        assert!(
            diff <= tol,
            "f64 repeat BW @ {i}: got {g} exp {exp} diff {diff} tol {tol}"
        );
    }
}
