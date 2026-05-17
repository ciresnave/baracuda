//! Real-GPU smoke test for `ConcatBackwardPlan<T, N>` — backward of
//! `y = cat(a, b, dim=k)` is the pure inverse routing
//! (`da = dy[..., :split, ...]`, `db = dy[..., split:, ...]`).
//! Bit-exact, no math.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test concat_backward_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ConcatBackwardArgs, ConcatBackwardDescriptor, ConcatBackwardPlan,
    ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// CPU reference: for each dy coord c, if c[concat_dim] < split_offset
// it goes to da at the same coord; else to db at the coord with
// concat_dim shifted by `-split_offset`. Returns (da, db).
fn cpu_split_ref<const N: usize, T: Copy + Default>(
    dy: &[T],
    output_shape: [i32; N],
    concat_dim: usize,
    split_offset: i32,
) -> (Vec<T>, Vec<T>) {
    let mut da_shape = output_shape;
    da_shape[concat_dim] = split_offset;
    let mut db_shape = output_shape;
    db_shape[concat_dim] = output_shape[concat_dim] - split_offset;

    let da_numel: usize = da_shape.iter().map(|&d| d as usize).product();
    let db_numel: usize = db_shape.iter().map(|&d| d as usize).product();
    let dy_numel: usize = output_shape.iter().map(|&d| d as usize).product();

    let mut dy_stride = [1usize; N];
    let mut da_stride = [1usize; N];
    let mut db_stride = [1usize; N];
    for d in (0..N).rev().skip(1) {
        dy_stride[d] = dy_stride[d + 1] * output_shape[d + 1] as usize;
        da_stride[d] = da_stride[d + 1] * da_shape[d + 1] as usize;
        db_stride[d] = db_stride[d + 1] * db_shape[d + 1] as usize;
    }

    let mut da = vec![T::default(); da_numel];
    let mut db = vec![T::default(); db_numel];

    for linear in 0..dy_numel {
        let mut coord = [0i32; N];
        let mut rem = linear;
        for d in (0..N).rev() {
            coord[d] = (rem % output_shape[d] as usize) as i32;
            rem /= output_shape[d] as usize;
        }
        let dy_idx: usize = (0..N).map(|d| coord[d] as usize * dy_stride[d]).sum();
        if coord[concat_dim] < split_offset {
            let out_idx: usize = (0..N).map(|d| coord[d] as usize * da_stride[d]).sum();
            da[out_idx] = dy[dy_idx];
        } else {
            let mut adj = coord;
            adj[concat_dim] = coord[concat_dim] - split_offset;
            let out_idx: usize = (0..N).map(|d| adj[d] as usize * db_stride[d]).sum();
            db[out_idx] = dy[dy_idx];
        }
    }
    (da, db)
}

// -------- f32 --------

#[test]
#[ignore]
fn concat_backward_f32_2d_dim0() {
    let (ctx, stream) = setup();
    let output_shape = [7i32, 5];
    let concat_dim = 0u8;
    let split_offset = 3i32;
    let dy_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f32> = (0..dy_numel).map(|i| (i as f32) * 0.5 - 9.0).collect();
    let (expected_da, expected_db) =
        cpu_split_ref(&host_dy, output_shape, concat_dim as usize, split_offset);

    let da_shape = {
        let mut s = output_shape;
        s[concat_dim as usize] = split_offset;
        s
    };
    let db_shape = {
        let mut s = output_shape;
        s[concat_dim as usize] = output_shape[concat_dim as usize] - split_offset;
        s
    };
    let da_numel: usize = da_shape.iter().map(|&d| d as usize).product();
    let db_numel: usize = db_shape.iter().map(|&d| d as usize).product();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_da: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, da_numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, db_numel).expect("alloc db");

    let desc = ConcatBackwardDescriptor {
        output_shape,
        concat_dim,
        split_offset,
        element: ElementKind::F32,
    };
    let plan = ConcatBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ConcatBackwardArgs::<f32, 2> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
        da: TensorMut {
            data: dev_da.as_slice_mut(),
            shape: da_shape,
            stride: contiguous_stride(da_shape),
        },
        db: TensorMut {
            data: dev_db.as_slice_mut(),
            shape: db_shape,
            stride: contiguous_stride(db_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_da = vec![0f32; da_numel];
    let mut got_db = vec![0f32; db_numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");
    for (i, (g, e)) in got_da.iter().zip(expected_da.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f32 concat BW da mismatch @ {i}");
    }
    for (i, (g, e)) in got_db.iter().zip(expected_db.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f32 concat BW db mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn concat_backward_f32_3d_dim1() {
    let (ctx, stream) = setup();
    let output_shape = [2i32, 9, 4];
    let concat_dim = 1u8;
    let split_offset = 5i32;
    let dy_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f32> = (0..dy_numel).map(|i| (i as f32) * 0.25 - 4.0).collect();
    let (expected_da, expected_db) =
        cpu_split_ref(&host_dy, output_shape, concat_dim as usize, split_offset);

    let da_shape = {
        let mut s = output_shape;
        s[concat_dim as usize] = split_offset;
        s
    };
    let db_shape = {
        let mut s = output_shape;
        s[concat_dim as usize] = output_shape[concat_dim as usize] - split_offset;
        s
    };
    let da_numel: usize = da_shape.iter().map(|&d| d as usize).product();
    let db_numel: usize = db_shape.iter().map(|&d| d as usize).product();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_da: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, da_numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, db_numel).expect("alloc db");

    let desc = ConcatBackwardDescriptor {
        output_shape,
        concat_dim,
        split_offset,
        element: ElementKind::F32,
    };
    let plan = ConcatBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ConcatBackwardArgs::<f32, 3> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
        da: TensorMut {
            data: dev_da.as_slice_mut(),
            shape: da_shape,
            stride: contiguous_stride(da_shape),
        },
        db: TensorMut {
            data: dev_db.as_slice_mut(),
            shape: db_shape,
            stride: contiguous_stride(db_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_da = vec![0f32; da_numel];
    let mut got_db = vec![0f32; db_numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");
    for (i, (g, e)) in got_da.iter().zip(expected_da.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f32 concat BW 3d da mismatch @ {i}");
    }
    for (i, (g, e)) in got_db.iter().zip(expected_db.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f32 concat BW 3d db mismatch @ {i}");
    }
}

// -------- f16 --------

#[test]
#[ignore]
fn concat_backward_f16_2d_dim1() {
    let (ctx, stream) = setup();
    let output_shape = [4i32, 8];
    let concat_dim = 1u8;
    let split_offset = 3i32;
    let dy_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f16> = (0..dy_numel)
        .map(|i| f16::from_f32((i as f32) * 0.125 - 2.0))
        .collect();
    let (expected_da, expected_db) =
        cpu_split_ref(&host_dy, output_shape, concat_dim as usize, split_offset);

    let da_shape = {
        let mut s = output_shape;
        s[concat_dim as usize] = split_offset;
        s
    };
    let db_shape = {
        let mut s = output_shape;
        s[concat_dim as usize] = output_shape[concat_dim as usize] - split_offset;
        s
    };
    let da_numel: usize = da_shape.iter().map(|&d| d as usize).product();
    let db_numel: usize = db_shape.iter().map(|&d| d as usize).product();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_da: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, da_numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, db_numel).expect("alloc db");

    let desc = ConcatBackwardDescriptor {
        output_shape,
        concat_dim,
        split_offset,
        element: ElementKind::F16,
    };
    let plan = ConcatBackwardPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ConcatBackwardArgs::<f16, 2> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
        da: TensorMut {
            data: dev_da.as_slice_mut(),
            shape: da_shape,
            stride: contiguous_stride(da_shape),
        },
        db: TensorMut {
            data: dev_db.as_slice_mut(),
            shape: db_shape,
            stride: contiguous_stride(db_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_da = vec![f16::ZERO; da_numel];
    let mut got_db = vec![f16::ZERO; db_numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");
    for (i, (g, e)) in got_da.iter().zip(expected_da.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f16 concat BW da mismatch @ {i}");
    }
    for (i, (g, e)) in got_db.iter().zip(expected_db.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f16 concat BW db mismatch @ {i}");
    }
}

// -------- bf16 --------

#[test]
#[ignore]
fn concat_backward_bf16_3d_dim2() {
    let (ctx, stream) = setup();
    let output_shape = [2i32, 3, 7];
    let concat_dim = 2u8;
    let split_offset = 4i32;
    let dy_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<bf16> = (0..dy_numel)
        .map(|i| bf16::from_f32((i as f32) * 0.5 - 5.0))
        .collect();
    let (expected_da, expected_db) =
        cpu_split_ref(&host_dy, output_shape, concat_dim as usize, split_offset);

    let da_shape = {
        let mut s = output_shape;
        s[concat_dim as usize] = split_offset;
        s
    };
    let db_shape = {
        let mut s = output_shape;
        s[concat_dim as usize] = output_shape[concat_dim as usize] - split_offset;
        s
    };
    let da_numel: usize = da_shape.iter().map(|&d| d as usize).product();
    let db_numel: usize = db_shape.iter().map(|&d| d as usize).product();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_da: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, da_numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, db_numel).expect("alloc db");

    let desc = ConcatBackwardDescriptor {
        output_shape,
        concat_dim,
        split_offset,
        element: ElementKind::Bf16,
    };
    let plan = ConcatBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ConcatBackwardArgs::<bf16, 3> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
        da: TensorMut {
            data: dev_da.as_slice_mut(),
            shape: da_shape,
            stride: contiguous_stride(da_shape),
        },
        db: TensorMut {
            data: dev_db.as_slice_mut(),
            shape: db_shape,
            stride: contiguous_stride(db_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_da = vec![bf16::ZERO; da_numel];
    let mut got_db = vec![bf16::ZERO; db_numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");
    for (i, (g, e)) in got_da.iter().zip(expected_da.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "bf16 concat BW da mismatch @ {i}");
    }
    for (i, (g, e)) in got_db.iter().zip(expected_db.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "bf16 concat BW db mismatch @ {i}");
    }
}

// -------- f64 --------

#[test]
#[ignore]
fn concat_backward_f64_2d_dim0() {
    let (ctx, stream) = setup();
    let output_shape = [6i32, 4];
    let concat_dim = 0u8;
    let split_offset = 2i32;
    let dy_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f64> = (0..dy_numel).map(|i| (i as f64) * 0.125 - 1.5).collect();
    let (expected_da, expected_db) =
        cpu_split_ref(&host_dy, output_shape, concat_dim as usize, split_offset);

    let da_shape = {
        let mut s = output_shape;
        s[concat_dim as usize] = split_offset;
        s
    };
    let db_shape = {
        let mut s = output_shape;
        s[concat_dim as usize] = output_shape[concat_dim as usize] - split_offset;
        s
    };
    let da_numel: usize = da_shape.iter().map(|&d| d as usize).product();
    let db_numel: usize = db_shape.iter().map(|&d| d as usize).product();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_da: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, da_numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, db_numel).expect("alloc db");

    let desc = ConcatBackwardDescriptor {
        output_shape,
        concat_dim,
        split_offset,
        element: ElementKind::F64,
    };
    let plan = ConcatBackwardPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ConcatBackwardArgs::<f64, 2> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
        da: TensorMut {
            data: dev_da.as_slice_mut(),
            shape: da_shape,
            stride: contiguous_stride(da_shape),
        },
        db: TensorMut {
            data: dev_db.as_slice_mut(),
            shape: db_shape,
            stride: contiguous_stride(db_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_da = vec![0f64; da_numel];
    let mut got_db = vec![0f64; db_numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");
    for (i, (g, e)) in got_da.iter().zip(expected_da.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f64 concat BW da mismatch @ {i}");
    }
    for (i, (g, e)) in got_db.iter().zip(expected_db.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f64 concat BW db mismatch @ {i}");
    }
}

// Edge case: split_offset == 0 (entire dy goes to db; da is empty).
#[test]
#[ignore]
fn concat_backward_f32_split_zero_all_to_b() {
    let (ctx, stream) = setup();
    let output_shape = [3i32, 4];
    let concat_dim = 0u8;
    let split_offset = 0i32;
    let dy_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f32> = (0..dy_numel).map(|i| (i as f32) + 1.0).collect();

    let da_shape = [0i32, 4];
    let db_shape = output_shape;
    let db_numel: usize = db_shape.iter().map(|&d| d as usize).product();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_da: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc da");
    let mut dev_db: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, db_numel).expect("alloc db");

    let desc = ConcatBackwardDescriptor {
        output_shape,
        concat_dim,
        split_offset,
        element: ElementKind::F32,
    };
    let plan = ConcatBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ConcatBackwardArgs::<f32, 2> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
        da: TensorMut {
            data: dev_da.as_slice_mut(),
            shape: da_shape,
            stride: contiguous_stride(da_shape),
        },
        db: TensorMut {
            data: dev_db.as_slice_mut(),
            shape: db_shape,
            stride: contiguous_stride(db_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_db = vec![0f32; db_numel];
    dev_db.copy_to_host(&mut got_db).expect("download db");
    for (i, (g, e)) in got_db.iter().zip(host_dy.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f32 concat BW split=0 db mismatch @ {i}");
    }
}

// Validation guard — reject out-of-range concat_dim.
#[test]
#[ignore]
fn concat_backward_rejects_bad_concat_dim() {
    let (_ctx, stream) = setup();
    let desc = ConcatBackwardDescriptor::<2> {
        output_shape: [4i32, 4],
        concat_dim: 2,
        split_offset: 2,
        element: ElementKind::F32,
    };
    let res = ConcatBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default());
    assert!(res.is_err(), "bad concat_dim should fail select");
}
