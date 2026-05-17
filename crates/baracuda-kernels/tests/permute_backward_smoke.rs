//! Real-GPU smoke test for `PermuteBackwardPlan<T, N>` — backward of
//! `permute` is `permute` with the inverse permutation. Bit-exact, no
//! math.
//!
//! Run: `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test permute_backward_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PermuteBackwardArgs, PermuteBackwardDescriptor,
    PermuteBackwardPlan, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// CPU reference for the BW: given dy of shape dy_shape and the forward
// `dims`, compute dx of shape input_shape such that
// `dx[c_in] = dy[c_out]` where `c_out[d] = c_in[dims[d]]`. Equivalently:
// dx[c_in] = dy at the coord obtained by permuting c_in.
fn cpu_permute_bw_ref<const N: usize, T: Copy + Default>(
    dy: &[T],
    input_shape: [i32; N],
    dy_shape: [i32; N],
    dims: [i32; N],
) -> Vec<T> {
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_stride = [1usize; N];
    for d in (0..N).rev().skip(1) {
        dy_stride[d] = dy_stride[d + 1] * dy_shape[d + 1] as usize;
    }
    let mut out = vec![T::default(); in_numel];
    // Walk dx in row-major order.
    for linear in 0..in_numel {
        let mut c_in = [0i32; N];
        let mut rem = linear;
        for d in (0..N).rev() {
            c_in[d] = (rem % input_shape[d] as usize) as i32;
            rem /= input_shape[d] as usize;
        }
        let mut dy_idx = 0usize;
        for d_out in 0..N {
            let in_axis = dims[d_out] as usize;
            dy_idx += c_in[in_axis] as usize * dy_stride[d_out];
        }
        out[linear] = dy[dy_idx];
    }
    out
}

fn run_case<T, const N: usize>(
    ctx: &Context,
    stream: &Stream,
    input_shape: [i32; N],
    dims: [i32; N],
    kind: ElementKind,
    host_dy: Vec<T>,
) where
    T: Copy + Default + baracuda_types::DeviceRepr + baracuda_kernels::Element + 'static,
{
    let dy_shape = {
        let mut s = [0i32; N];
        for d in 0..N {
            s[d] = input_shape[dims[d] as usize];
        }
        s
    };
    let expected: Vec<T> = cpu_permute_bw_ref(&host_dy, input_shape, dy_shape, dims);

    let dev_dy = DeviceBuffer::from_slice(ctx, &host_dy).expect("upload");
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dev_dx: DeviceBuffer<T> = DeviceBuffer::zeros(ctx, in_numel).expect("alloc");
    let desc = PermuteBackwardDescriptor {
        input_shape,
        dims,
        element: kind,
    };
    let plan = PermuteBackwardPlan::<T, N>::select(stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = PermuteBackwardArgs::<T, N> {
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
    plan.run(stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![T::default(); in_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        // Bit-exact compare via raw bytes (works for all wired dtypes).
        let g_bytes = unsafe {
            core::slice::from_raw_parts(
                (g as *const T) as *const u8,
                core::mem::size_of::<T>(),
            )
        };
        let e_bytes = unsafe {
            core::slice::from_raw_parts(
                (e as *const T) as *const u8,
                core::mem::size_of::<T>(),
            )
        };
        assert_eq!(g_bytes, e_bytes, "permute BW mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn permute_backward_f32_2d_transpose() {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 5];
    let dims = [1i32, 0];
    let host_dy: Vec<f32> = (0..15).map(|i| (i as f32) * 0.5 - 3.0).collect();
    run_case::<f32, 2>(&ctx, &stream, input_shape, dims, ElementKind::F32, host_dy);
}

#[test]
#[ignore]
fn permute_backward_f32_3d_perm_201() {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 3, 4];
    let dims = [2i32, 0, 1];
    let host_dy: Vec<f32> = (0..24).map(|i| (i as f32) * 0.25 - 2.0).collect();
    run_case::<f32, 3>(&ctx, &stream, input_shape, dims, ElementKind::F32, host_dy);
}

#[test]
#[ignore]
fn permute_backward_f32_identity_is_copy() {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 3, 4];
    let dims = [0i32, 1, 2];
    let host_dy: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1).collect();
    run_case::<f32, 3>(&ctx, &stream, input_shape, dims, ElementKind::F32, host_dy);
}

#[test]
#[ignore]
fn permute_backward_f16_3d_perm_120() {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 3, 4];
    let dims = [1i32, 2, 0];
    let host_dy: Vec<f16> = (0..24).map(|i| f16::from_f32((i as f32) * 0.125 - 1.0)).collect();
    run_case::<f16, 3>(&ctx, &stream, input_shape, dims, ElementKind::F16, host_dy);
}

#[test]
#[ignore]
fn permute_backward_bf16_2d_transpose() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 6];
    let dims = [1i32, 0];
    let host_dy: Vec<bf16> = (0..24).map(|i| bf16::from_f32((i as f32) * 0.25)).collect();
    run_case::<bf16, 2>(&ctx, &stream, input_shape, dims, ElementKind::Bf16, host_dy);
}

#[test]
#[ignore]
fn permute_backward_f64_3d_perm_201() {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 3, 4];
    let dims = [2i32, 0, 1];
    let host_dy: Vec<f64> = (0..24).map(|i| (i as f64) * 0.0625 - 0.5).collect();
    run_case::<f64, 3>(&ctx, &stream, input_shape, dims, ElementKind::F64, host_dy);
}
