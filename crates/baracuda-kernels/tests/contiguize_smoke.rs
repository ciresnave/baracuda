//! Real-GPU smoke tests for `ContiguizePlan<T, N>` — strided→contiguous
//! materialization (Phase 13.2).
//!
//! Coverage:
//!   - Already-contiguous fast path (rank-3 f32).
//!   - Innermost-stride-1 fast path (NHWC view of NCHW data, f32).
//!   - Transposed source (innermost stride > 1, rank-2 f32).
//!   - Negative strides (Flip on axis 0, rank-2 f32).
//!   - Zero-stride / BroadcastTo (rank-3 f32, axis-1 broadcast).
//!   - Nibble-packed S4 contiguize on a contiguous source.
//!   - Quick correctness on f16, bf16, f64 with a transposed source.
//!
//! Bit-exact compare via `to_bits()` (contiguize is pure copy — no math).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ContiguizeArgs, ContiguizeDescriptor, ContiguizePlan, ElementKind,
    PlanPreference, S4, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// =============================================================================
// Fast path #1 — source already contiguous + zero offset → cudaMemcpyAsync.
// =============================================================================

#[test]
#[ignore]
fn contiguize_already_contiguous_rank3_f32() {
    let (ctx, stream) = setup();
    let shape = [2i32, 3, 4];
    let numel = 24usize;
    let host_x: Vec<f32> = (0..numel).map(|i| i as f32 * 0.125).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let strides = contiguous_stride(shape);
    let desc = ContiguizeDescriptor::<3> {
        shape,
        source_strides: strides,
        source_offset: 0,
        element: ElementKind::F32,
    };
    let plan = ContiguizePlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ContiguizeArgs::<f32, 3> {
        source: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride: strides,
        },
        dest: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_x.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "already-contig mismatch @ {i}");
    }
}

// =============================================================================
// Fast path #2 — innermost stride 1.
// Build a `permute(0,2,3,1)`-like view that keeps the innermost W axis
// contiguous in the SOURCE layout.
//
// Concretely: source data is a 1-axis run [0..numel) materialized into
// device memory; the descriptor describes a view with shape [N, H, W]
// where outer strides are non-canonical but stride[2] == 1.
// =============================================================================

#[test]
#[ignore]
fn contiguize_inner_stride1_fastpath_f32() {
    let (ctx, stream) = setup();
    // Underlying buffer of shape [N=2, H=3, W=4], canonical layout.
    let phys_shape = [2i32, 3, 4];
    let _phys_strides = contiguous_stride(phys_shape); // [12, 4, 1] — documents the underlying layout.
    let numel = 24usize;
    let host_phys: Vec<f32> = (0..numel).map(|i| i as f32 + 100.0).collect();
    let dev_phys = DeviceBuffer::from_slice(&ctx, &host_phys).expect("upload");

    // Now describe a view that "reverses" axis ordering of the outer
    // axes — pick shape [3, 2, 4] (H, N, W) with strides [4, 12, 1].
    // Innermost stride is 1 → fast path #2 fires.
    let view_shape = [3i32, 2, 4];
    let view_strides = [4i64, 12, 1];
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = ContiguizeDescriptor::<3> {
        shape: view_shape,
        source_strides: view_strides,
        source_offset: 0,
        element: ElementKind::F32,
    };
    let plan = ContiguizePlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ContiguizeArgs::<f32, 3> {
        source: TensorRef {
            data: dev_phys.as_slice(),
            shape: view_shape,
            stride: view_strides,
        },
        dest: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: view_shape,
            stride: contiguous_stride(view_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");

    // Reference: for each (h, n, w), out_lin = h*8 + n*4 + w. The
    // source element is at phys offset h*4 + n*12 + w.
    let mut expected = vec![0f32; numel];
    for h in 0..3 {
        for n in 0..2 {
            for w in 0..4 {
                let dst = h * 8 + n * 4 + w;
                let src = h * 4 + n * 12 + w;
                expected[dst] = host_phys[src];
            }
        }
    }
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "inner-stride-1 fastpath mismatch @ {i}: got {g} expected {e}"
        );
    }
}

// =============================================================================
// Generic path — transposed source (innermost stride > 1).
// Rank-2 transpose: source shape [M, N] with strides [1, M] over a
// physically [N, M]-laid-out buffer.
// =============================================================================

#[test]
#[ignore]
fn contiguize_transposed_rank2_f32() {
    let (ctx, stream) = setup();
    // Physical layout: shape [N=3, M=4] row-major.
    let phys_rows = 3usize;
    let phys_cols = 4usize;
    let numel = phys_rows * phys_cols; // 12
    let host_phys: Vec<f32> = (0..numel).map(|i| (i as f32) - 5.5).collect();
    let dev_phys = DeviceBuffer::from_slice(&ctx, &host_phys).expect("upload");

    // Logical (transposed) view: shape [M=4, N=3] with strides [1, M=4].
    let view_shape = [4i32, 3];
    let view_strides = [1i64, 4];
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = ContiguizeDescriptor::<2> {
        shape: view_shape,
        source_strides: view_strides,
        source_offset: 0,
        element: ElementKind::F32,
    };
    let plan = ContiguizePlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ContiguizeArgs::<f32, 2> {
        source: TensorRef {
            data: dev_phys.as_slice(),
            shape: view_shape,
            stride: view_strides,
        },
        dest: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: view_shape,
            stride: contiguous_stride(view_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    // Reference: out[i, j] = phys[i*1 + j*4]
    let mut expected = vec![0f32; numel];
    for i in 0..4 {
        for j in 0..3 {
            expected[i * 3 + j] = host_phys[i + j * 4];
        }
    }
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "transpose mismatch @ {i}");
    }
}

// =============================================================================
// Negative strides — Flip on axis 0.
// Physical buffer shape [4, 3]; view with strides [-3, 1] starting at
// element offset (4-1)*3 == 9 reverses rows.
// =============================================================================

#[test]
#[ignore]
fn contiguize_negative_stride_rank2_f32() {
    let (ctx, stream) = setup();
    let shape = [4i32, 3];
    let numel = 12usize;
    let host_phys: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5).collect();
    let dev_phys = DeviceBuffer::from_slice(&ctx, &host_phys).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    // Flip axis 0: strides [-3, 1], offset = (4-1)*3 = 9.
    let view_strides = [-3i64, 1];
    let view_offset: i64 = 9;

    let desc = ContiguizeDescriptor::<2> {
        shape,
        source_strides: view_strides,
        source_offset: view_offset,
        element: ElementKind::F32,
    };
    let plan = ContiguizePlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ContiguizeArgs::<f32, 2> {
        source: TensorRef {
            data: dev_phys.as_slice(),
            shape,
            stride: view_strides,
        },
        dest: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    // Reference: out[i, j] = phys[9 + i*(-3) + j*1] = phys[(3-i)*3 + j].
    let mut expected = vec![0f32; numel];
    for i in 0..4 {
        for j in 0..3 {
            expected[i * 3 + j] = host_phys[(3 - i) * 3 + j];
        }
    }
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "neg-stride mismatch @ {i}");
    }
}

// =============================================================================
// Zero stride — BroadcastTo on axis 1.
// Physical buffer: shape [2, 4] (broadcast source). View: shape
// [2, 3, 4] with stride[1] == 0 — every step along axis 1 reads the
// same source row.
// =============================================================================

#[test]
#[ignore]
fn contiguize_zero_stride_rank3_f32() {
    let (ctx, stream) = setup();
    let phys_shape_outer = 2usize;
    let phys_shape_inner = 4usize;
    let phys_numel = phys_shape_outer * phys_shape_inner; // 8
    let host_phys: Vec<f32> = (0..phys_numel).map(|i| (i as f32) + 10.0).collect();
    let dev_phys = DeviceBuffer::from_slice(&ctx, &host_phys).expect("upload");

    // View shape [2, 3, 4]: axis 1 is broadcast (stride 0).
    let view_shape = [2i32, 3, 4];
    let view_strides = [4i64, 0, 1];
    let view_numel = 24usize;
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, view_numel).expect("alloc");

    let desc = ContiguizeDescriptor::<3> {
        shape: view_shape,
        source_strides: view_strides,
        source_offset: 0,
        element: ElementKind::F32,
    };
    let plan = ContiguizePlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ContiguizeArgs::<f32, 3> {
        source: TensorRef {
            data: dev_phys.as_slice(),
            shape: view_shape,
            stride: view_strides,
        },
        dest: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: view_shape,
            stride: contiguous_stride(view_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; view_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    // Reference: out[i, j, k] = phys[i*4 + k]  (j doesn't index source).
    let mut expected = vec![0f32; view_numel];
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                expected[i * 12 + j * 4 + k] = host_phys[i * 4 + k];
            }
        }
    }
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "zero-stride mismatch @ {i}");
    }
}

// =============================================================================
// Nibble (S4) — contiguous source. The most common nibble case and a
// smoke test of the nibble kernel path.
// =============================================================================

#[test]
#[ignore]
fn contiguize_s4_contiguous_rank1() {
    let (ctx, stream) = setup();
    // 8 s4 elements packed into 4 bytes. Values in [-8, +7].
    let elems: Vec<i8> = vec![1, -2, 3, -4, 5, -6, 7, -8];
    let packed: Vec<S4> = (0..4).map(|i| S4::pack(elems[2 * i], elems[2 * i + 1])).collect();
    let dev_src = DeviceBuffer::from_slice(&ctx, &packed).expect("upload s4");
    // Dest: 4 storage slots == 8 nibbles.
    let mut dev_dst: DeviceBuffer<S4> = DeviceBuffer::zeros(&ctx, 4).expect("alloc s4");

    let shape = [8i32]; // 8 NIBBLES (logical s4 elements).
    let desc = ContiguizeDescriptor::<1> {
        shape,
        source_strides: [1i64],
        source_offset: 0,
        element: ElementKind::S4,
    };
    let plan = ContiguizePlan::<S4, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select s4");
    let args = ContiguizeArgs::<S4, 1> {
        source: TensorRef {
            data: dev_src.as_slice(),
            shape,
            stride: [1i64],
        },
        dest: TensorMut {
            data: dev_dst.as_slice_mut(),
            shape,
            stride: [1i64],
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run s4");
    stream.synchronize().expect("sync");

    let mut got = vec![S4(0); 4];
    dev_dst.copy_to_host(&mut got).expect("download s4");
    // Source was contiguous + zero offset → should be a bit-exact byte
    // copy of the packed bytes.
    for i in 0..4 {
        assert_eq!(
            got[i].0,
            packed[i].0,
            "s4 contig copy mismatch @ pack slot {i}"
        );
    }
}

// =============================================================================
// Dtype fanout — transposed rank-2 source in f16 / bf16 / f64.
// =============================================================================

fn contiguize_transposed_dtype<T>(kind: ElementKind, make: impl Fn(usize) -> T, to_bits: impl Fn(T) -> u128)
where
    T: baracuda_types::DeviceRepr + Copy + 'static + core::fmt::Debug,
{
    let (ctx, stream) = setup();
    let phys_rows = 3usize;
    let phys_cols = 4usize;
    let numel = phys_rows * phys_cols;
    let host_phys: Vec<T> = (0..numel).map(&make).collect();
    let dev_phys = DeviceBuffer::from_slice(&ctx, &host_phys).expect("upload");

    let view_shape = [4i32, 3];
    let view_strides = [1i64, 4];
    let mut dev_y: DeviceBuffer<T> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = ContiguizeDescriptor::<2> {
        shape: view_shape,
        source_strides: view_strides,
        source_offset: 0,
        element: kind,
    };
    let plan = ContiguizePlan::<T, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ContiguizeArgs::<T, 2> {
        source: TensorRef {
            data: dev_phys.as_slice(),
            shape: view_shape,
            stride: view_strides,
        },
        dest: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: view_shape,
            stride: contiguous_stride(view_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got: Vec<T> = (0..numel).map(&make).collect(); // placeholder buffer
    dev_y.copy_to_host(&mut got).expect("download");
    for i in 0..4 {
        for j in 0..3 {
            let expected = host_phys[i + j * 4];
            let g = got[i * 3 + j];
            assert_eq!(
                to_bits(g),
                to_bits(expected),
                "transpose dtype {:?} mismatch @ ({i},{j})",
                kind,
            );
        }
    }
}

#[test]
#[ignore]
fn contiguize_transposed_f16() {
    contiguize_transposed_dtype::<f16>(
        ElementKind::F16,
        |i| f16::from_f32((i as f32) - 5.0),
        |x| x.to_bits() as u128,
    );
}

#[test]
#[ignore]
fn contiguize_transposed_bf16() {
    contiguize_transposed_dtype::<bf16>(
        ElementKind::Bf16,
        |i| bf16::from_f32((i as f32) - 5.0),
        |x| x.to_bits() as u128,
    );
}

#[test]
#[ignore]
fn contiguize_transposed_f64() {
    contiguize_transposed_dtype::<f64>(
        ElementKind::F64,
        |i| (i as f64) * 0.25 - 1.5,
        |x| x.to_bits() as u128,
    );
}
