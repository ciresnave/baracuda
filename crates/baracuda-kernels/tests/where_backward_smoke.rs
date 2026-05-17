//! Real-GPU smoke test for the where backward kernel
//! (`WhereBackwardPlan<T, N>`).
//!
//! Forward: `y = where(cond, a, b)`. Backward (cond non-differentiable):
//!   da = where(cond, dy, 0),  db = where(cond, 0, dy).
//!
//! Pure mask + copy — no arithmetic — so output is bit-exact against
//! host reference at every dtype. Each test compares `.to_bits()`
//! directly, mirroring the FW Where smoke pattern.
//!
//! Cond alternates every cell so both branches are exercised in every
//! test.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test where_backward_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, WhereBackwardArgs,
    WhereBackwardDescriptor, WhereBackwardPlan, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn where_backward_f32_3d() {
    let (ctx, stream) = setup();
    let shape: [i32; 3] = [8, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    // Cond alternates every cell so we exercise both branches.
    let host_cond: Vec<u8> = (0..numel).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_da: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");

    let stride = contiguous_stride(shape);
    let desc = WhereBackwardDescriptor {
        shape,
        element: ElementKind::F32,
    };
    let plan = WhereBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    let args = WhereBackwardArgs::<f32, 3> {
        cond: TensorRef {
            data: dev_cond.as_slice(),
            shape,
            stride,
        },
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape,
            stride,
        },
        da: TensorMut {
            data: dev_da.as_slice_mut(),
            shape,
            stride,
        },
        db: TensorMut {
            data: dev_db.as_slice_mut(),
            shape,
            stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_da = vec![0f32; numel];
    let mut got_db = vec![0f32; numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");

    for i in 0..numel {
        let (exp_da, exp_db) = if host_cond[i] != 0 {
            (host_dy[i], 0.0f32)
        } else {
            (0.0f32, host_dy[i])
        };
        assert_eq!(
            got_da[i].to_bits(),
            exp_da.to_bits(),
            "where backward f32 da @ {i}: got {} exp {}",
            got_da[i],
            exp_da
        );
        assert_eq!(
            got_db[i].to_bits(),
            exp_db.to_bits(),
            "where backward f32 db @ {i}: got {} exp {}",
            got_db[i],
            exp_db
        );
    }
}

#[test]
#[ignore]
fn where_backward_f16_3d() {
    let (ctx, stream) = setup();
    let shape: [i32; 3] = [8, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_cond: Vec<u8> = (0..numel).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
    // Magnitudes in f16's safe range to avoid subnormal corner cases.
    let host_dy: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(((i as f32) * 0.0125) - 10.0))
        .collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_da: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");

    let stride = contiguous_stride(shape);
    let desc = WhereBackwardDescriptor {
        shape,
        element: ElementKind::F16,
    };
    let plan = WhereBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    let args = WhereBackwardArgs::<f16, 3> {
        cond: TensorRef {
            data: dev_cond.as_slice(),
            shape,
            stride,
        },
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape,
            stride,
        },
        da: TensorMut {
            data: dev_da.as_slice_mut(),
            shape,
            stride,
        },
        db: TensorMut {
            data: dev_db.as_slice_mut(),
            shape,
            stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_da = vec![f16::from_f32(0.0); numel];
    let mut got_db = vec![f16::from_f32(0.0); numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");

    let zero = f16::from_f32(0.0);
    for i in 0..numel {
        let (exp_da, exp_db) = if host_cond[i] != 0 {
            (host_dy[i], zero)
        } else {
            (zero, host_dy[i])
        };
        assert_eq!(
            got_da[i].to_bits(),
            exp_da.to_bits(),
            "where backward f16 da @ {i}: got {} exp {}",
            got_da[i],
            exp_da
        );
        assert_eq!(
            got_db[i].to_bits(),
            exp_db.to_bits(),
            "where backward f16 db @ {i}: got {} exp {}",
            got_db[i],
            exp_db
        );
    }
}

#[test]
#[ignore]
fn where_backward_bf16_3d() {
    let (ctx, stream) = setup();
    let shape: [i32; 3] = [8, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_cond: Vec<u8> = (0..numel).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
    let host_dy: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(((i as f32) * 0.0125) - 10.0))
        .collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_da: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");

    let stride = contiguous_stride(shape);
    let desc = WhereBackwardDescriptor {
        shape,
        element: ElementKind::Bf16,
    };
    let plan = WhereBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    let args = WhereBackwardArgs::<bf16, 3> {
        cond: TensorRef {
            data: dev_cond.as_slice(),
            shape,
            stride,
        },
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape,
            stride,
        },
        da: TensorMut {
            data: dev_da.as_slice_mut(),
            shape,
            stride,
        },
        db: TensorMut {
            data: dev_db.as_slice_mut(),
            shape,
            stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_da = vec![bf16::from_f32(0.0); numel];
    let mut got_db = vec![bf16::from_f32(0.0); numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");

    let zero = bf16::from_f32(0.0);
    for i in 0..numel {
        let (exp_da, exp_db) = if host_cond[i] != 0 {
            (host_dy[i], zero)
        } else {
            (zero, host_dy[i])
        };
        assert_eq!(
            got_da[i].to_bits(),
            exp_da.to_bits(),
            "where backward bf16 da @ {i}: got {} exp {}",
            got_da[i],
            exp_da
        );
        assert_eq!(
            got_db[i].to_bits(),
            exp_db.to_bits(),
            "where backward bf16 db @ {i}: got {} exp {}",
            got_db[i],
            exp_db
        );
    }
}

#[test]
#[ignore]
fn where_backward_f64_3d() {
    let (ctx, stream) = setup();
    let shape: [i32; 3] = [8, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_cond: Vec<u8> = (0..numel).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
    let host_dy: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.5 - 17.25).collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_da: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");

    let stride = contiguous_stride(shape);
    let desc = WhereBackwardDescriptor {
        shape,
        element: ElementKind::F64,
    };
    let plan = WhereBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    let args = WhereBackwardArgs::<f64, 3> {
        cond: TensorRef {
            data: dev_cond.as_slice(),
            shape,
            stride,
        },
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape,
            stride,
        },
        da: TensorMut {
            data: dev_da.as_slice_mut(),
            shape,
            stride,
        },
        db: TensorMut {
            data: dev_db.as_slice_mut(),
            shape,
            stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_da = vec![0f64; numel];
    let mut got_db = vec![0f64; numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");

    for i in 0..numel {
        let (exp_da, exp_db) = if host_cond[i] != 0 {
            (host_dy[i], 0.0f64)
        } else {
            (0.0f64, host_dy[i])
        };
        assert_eq!(
            got_da[i].to_bits(),
            exp_da.to_bits(),
            "where backward f64 da @ {i}: got {} exp {}",
            got_da[i],
            exp_da
        );
        assert_eq!(
            got_db[i].to_bits(),
            exp_db.to_bits(),
            "where backward f64 db @ {i}: got {} exp {}",
            got_db[i],
            exp_db
        );
    }
}
