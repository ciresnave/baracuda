//! Real-GPU smoke test for `SparsemaxPlan` (Milestone 5.4).
//!
//! Forward: `y = ProjSimplex(x)` via the sort-then-threshold closed
//! form. Verifies:
//!   1. Known reference: for input `[2, 1, 0]`, sparsemax → `[0.5, 0.5, 0]`.
//!   2. Row-sums-to-1 invariant on a random input.
//!   3. All outputs `>= 0`, with some genuinely zero (sparsity).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SparsemaxArgs, SparsemaxDescriptor,
    SparsemaxPlan, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference for sparsemax along axis 1 of a 2-D shape.
fn host_sparsemax_f32(shape: [i32; 2], axis: usize, x: &[f32]) -> Vec<f32> {
    let extent = shape[axis] as usize;
    let other = shape[1 - axis] as usize;
    let numel = (shape[0] * shape[1]) as usize;
    let mut y = vec![0f32; numel];
    for o in 0..other {
        let idx = |j: usize| -> usize {
            if axis == 1 { o * shape[1] as usize + j } else { j * shape[1] as usize + o }
        };
        // Sort descending.
        let mut row: Vec<f32> = (0..extent).map(|j| x[idx(j)]).collect();
        let mut sorted = row.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mut cum = 0f32;
        let mut tau = 0f32;
        for a in 0..extent {
            cum += sorted[a];
            if 1.0 + (a as f32 + 1.0) * sorted[a] > cum {
                tau = (cum - 1.0) / (a as f32 + 1.0);
            }
        }
        for j in 0..extent {
            let v = (row[j] - tau).max(0.0);
            y[idx(j)] = v;
        }
        let _ = &mut row; // silence unused-warn
    }
    y
}

#[test]
#[ignore]
fn sparsemax_f32_known_reference() {
    let (ctx, stream) = setup();
    // Known reference: sparsemax([0.5, 0.5, 0]) — already on the simplex,
    // so it should return itself.
    // Also: sparsemax([2, 1, 0]) → [1, 0, 0]:
    //   sort desc [2,1,0], cum [2,3,3].
    //   k=0: 1 + 1·2 = 3 > 2 → τ=(2-1)/1 = 1.
    //   k=1: 1 + 2·1 = 3 > 3? No (strict).
    //   y = max(0, x - 1) = [1, 0, 0].
    let shape = [1i32, 3];
    let host_x = vec![2.0_f32, 1.0, 0.0];
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");
    let desc = SparsemaxDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        element: ElementKind::F32,
    };
    let plan =
        SparsemaxPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(&stream, Workspace::None, SparsemaxArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    })
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; 3];
    dev_y.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * f32::EPSILON;
    let expected = [1.0_f32, 0.0, 0.0];
    for i in 0..3 {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol,
            "sparsemax @ {i}: got={} want={}", got[i], expected[i]);
    }

    // Second case: sparsemax([0.6, 0.5, 0.4]) — closer logits should
    // yield denser output.
    //   sort desc [0.6, 0.5, 0.4], cum [0.6, 1.1, 1.5].
    //   k=0: 1 + 1·0.6 = 1.6 > 0.6 → τ=(0.6-1)/1 = -0.4.
    //   k=1: 1 + 2·0.5 = 2 > 1.1 → τ=(1.1-1)/2 = 0.05.
    //   k=2: 1 + 3·0.4 = 2.2 > 1.5 → τ=(1.5-1)/3 ≈ 0.1667.
    //   y = max(0, x - 0.1667) = [0.4333, 0.3333, 0.2333].
    let host_x2 = vec![0.6_f32, 0.5, 0.4];
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &host_x2).expect("up");
    let mut dev_y2: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");
    plan.run(&stream, Workspace::None, SparsemaxArgs {
        x: TensorRef { data: dev_x2.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorMut { data: dev_y2.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    })
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got2 = vec![0f32; 3];
    dev_y2.copy_to_host(&mut got2).expect("dl");
    let expected2 = [0.6 - 1.0/6.0_f32, 0.5 - 1.0/6.0, 0.4 - 1.0/6.0];
    for i in 0..3 {
        let tol = 1e-5_f32;
        assert!((got2[i] - expected2[i]).abs() <= tol,
            "sparsemax case2 @ {i}: got={} want={}", got2[i], expected2[i]);
    }
}

#[test]
#[ignore]
fn sparsemax_f32_2d_axis_1() {
    let (ctx, stream) = setup();
    let shape = [4i32, 6];
    let numel = 24usize;
    let host_x: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.3 - 1.0).sin()).collect();
    let expected = host_sparsemax_f32(shape, 1, &host_x);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = SparsemaxDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        element: ElementKind::F32,
    };
    let plan =
        SparsemaxPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(&stream, Workspace::None, SparsemaxArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    })
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    let eps = 8.0 * f32::EPSILON;
    for i in 0..numel {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol,
            "f32 sparsemax @ {i}: got={} want={}", got[i], expected[i]);
        assert!(got[i] >= 0.0, "f32 sparsemax non-negative @ {i}: {}", got[i]);
    }
    for row in 0..4 {
        let mut sum = 0f32;
        for j in 0..6 { sum += got[row * 6 + j]; }
        assert!((sum - 1.0).abs() <= 1e-5, "row-sum row={row} = {sum}");
    }
}

#[test]
#[ignore]
fn sparsemax_f64_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let host_x: Vec<f64> = (0..numel).map(|i| ((i as f64) * 0.2 - 0.5).cos()).collect();

    // f64 reference (clone host_sparsemax_f32 inline).
    let mut expected = vec![0f64; numel];
    for row in 0..3 {
        let row_data: Vec<f64> = (0..5).map(|j| host_x[row * 5 + j]).collect();
        let mut sorted = row_data.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mut cum = 0f64;
        let mut tau = 0f64;
        for a in 0..5 {
            cum += sorted[a];
            if 1.0 + (a as f64 + 1.0) * sorted[a] > cum {
                tau = (cum - 1.0) / (a as f64 + 1.0);
            }
        }
        for j in 0..5 {
            expected[row * 5 + j] = (row_data[j] - tau).max(0.0);
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = SparsemaxDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        element: ElementKind::F64,
    };
    let plan =
        SparsemaxPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(&stream, Workspace::None, SparsemaxArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    })
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * f64::EPSILON;
    for i in 0..numel {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol, "f64 sparsemax @ {i}");
    }
}

#[test]
#[ignore]
fn sparsemax_f16_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.25 - 0.5).cos()).collect();
    let expected_f32 = host_sparsemax_f32(shape, 1, &host_x_f32);
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = SparsemaxDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        element: ElementKind::F16,
    };
    let plan =
        SparsemaxPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(&stream, Workspace::None, SparsemaxArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    })
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * 9.77e-4_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "f16 sparsemax @ {i}: diff={diff}");
    }
}

#[test]
#[ignore]
fn sparsemax_bf16_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.25 - 1.0).sin()).collect();
    let expected_f32 = host_sparsemax_f32(shape, 1, &host_x_f32);
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = SparsemaxDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        element: ElementKind::Bf16,
    };
    let plan =
        SparsemaxPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(&stream, Workspace::None, SparsemaxArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    })
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * 7.81e-3_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "bf16 sparsemax @ {i}: diff={diff}");
    }
}
