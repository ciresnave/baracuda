//! Real-GPU smoke tests for `ArgReducePlan<T, N, I>` output-dtype
//! generality (Phase 12.2 — Fuel team feedback).
//!
//! Verifies that the `argmax` / `argmin` axis kernels produce
//! identical *index* results across the three supported output dtypes
//! (`u32`, `i32`, `i64`), each with `f32` input. PyTorch's i64-default
//! behavior is preserved (legacy callers continue to work unchanged);
//! u32 and i32 are opt-in via the third type parameter on
//! [`baracuda_kernels::ArgReducePlan`].
//!
//! Layout: a 2x4 input. Argmax along axis 1 ⇒ per-row index of the
//! maximum (extent 4). Same for argmin.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test arg_axis_dtype_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ArgReduceArgs, ArgReduceDescriptor, ArgReduceKind, ArgReducePlan,
    ElementKind, IndexOutputElement, PlanPreference, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// 2x4 row-major input:
//   row 0: [1.0,  3.0, -2.0,  0.0]  -> argmax = 1, argmin = 2
//   row 1: [4.0, -1.0,  2.0,  5.0]  -> argmax = 3, argmin = 1
const HOST_X: [f32; 8] = [1.0, 3.0, -2.0, 0.0, 4.0, -1.0, 2.0, 5.0];
const EXPECT_ARGMAX_AXIS1: [i64; 2] = [1, 3];
const EXPECT_ARGMIN_AXIS1: [i64; 2] = [2, 1];

/// Run argmax/argmin along axis 1 over a 2x4 f32 input with output
/// dtype `I` and return the downloaded host vector as `i64` (cast from
/// whatever the kernel wrote, so the assertion logic is uniform).
fn run_arg_axis<I>(kind: ArgReduceKind) -> Vec<i64>
where
    I: IndexOutputElement + Copy + 'static,
    i64: TryFrom<u64>,
{
    let (ctx, stream) = setup();
    let input_shape = [2i32, 4i32];
    let output_shape = [2i32, 1i32];

    let dev_x = DeviceBuffer::from_slice(&ctx, &HOST_X).expect("upload x");
    let mut dev_y: DeviceBuffer<I> = DeviceBuffer::zeros(&ctx, 2).expect("alloc y");

    let desc = ArgReduceDescriptor {
        kind,
        input_shape,
        reduce_axis: 1,
        element: ElementKind::F32,
    };
    let plan = ArgReducePlan::<f32, 2, I>::select(&stream, &desc, PlanPreference::default())
        .expect("select arg_reduce<f32, 2, I>");

    let args = ArgReduceArgs::<f32, 2, I> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    // Download into `Vec<I>` then widen to i64 for the per-test
    // assertion. We can't blanket-cast `I` to `i64` without trait
    // bounds, so each caller does the cast explicitly via a
    // per-type adapter (`download_as_i64`).
    download_as_i64(&dev_y, 2)
}

// Per-type download helpers. The three I types we test produce values
// that all fit in i64 (per-row index ∈ [0, 4)), so the casts are lossless.
fn download_as_i64<I: IndexOutputElement>(buf: &DeviceBuffer<I>, n: usize) -> Vec<i64> {
    use baracuda_kernels::IndexOutputKind;
    match I::KIND {
        IndexOutputKind::U32 => {
            // SAFETY: I is u32 by tag; the wire format is u32.
            let typed: &DeviceBuffer<u32> = unsafe {
                &*(buf as *const DeviceBuffer<I> as *const DeviceBuffer<u32>)
            };
            let mut got = vec![0u32; n];
            typed.copy_to_host(&mut got).expect("download u32");
            got.into_iter().map(|v| v as i64).collect()
        }
        IndexOutputKind::I32 => {
            let typed: &DeviceBuffer<i32> = unsafe {
                &*(buf as *const DeviceBuffer<I> as *const DeviceBuffer<i32>)
            };
            let mut got = vec![0i32; n];
            typed.copy_to_host(&mut got).expect("download i32");
            got.into_iter().map(|v| v as i64).collect()
        }
        IndexOutputKind::I64 => {
            let typed: &DeviceBuffer<i64> = unsafe {
                &*(buf as *const DeviceBuffer<I> as *const DeviceBuffer<i64>)
            };
            let mut got = vec![0i64; n];
            typed.copy_to_host(&mut got).expect("download i64");
            got
        }
        // `IndexOutputKind` is `#[non_exhaustive]` (Phase 28); a new
        // variant needs a matching download helper.
        _ => panic!("arg_axis_dtype_smoke: unsupported IndexOutputKind variant"),
    }
}

// =============================================================================
// Argmax × {u32, i32, i64}
// =============================================================================

#[test]
#[ignore]
fn argmax_axis1_f32_u32_output() {
    let got = run_arg_axis::<u32>(ArgReduceKind::Argmax);
    assert_eq!(got, EXPECT_ARGMAX_AXIS1, "argmax u32 mismatch");
}

#[test]
#[ignore]
fn argmax_axis1_f32_i32_output() {
    let got = run_arg_axis::<i32>(ArgReduceKind::Argmax);
    assert_eq!(got, EXPECT_ARGMAX_AXIS1, "argmax i32 mismatch");
}

#[test]
#[ignore]
fn argmax_axis1_f32_i64_output() {
    let got = run_arg_axis::<i64>(ArgReduceKind::Argmax);
    assert_eq!(got, EXPECT_ARGMAX_AXIS1, "argmax i64 mismatch");
}

// =============================================================================
// Argmin × {u32, i32, i64}
// =============================================================================

#[test]
#[ignore]
fn argmin_axis1_f32_u32_output() {
    let got = run_arg_axis::<u32>(ArgReduceKind::Argmin);
    assert_eq!(got, EXPECT_ARGMIN_AXIS1, "argmin u32 mismatch");
}

#[test]
#[ignore]
fn argmin_axis1_f32_i32_output() {
    let got = run_arg_axis::<i32>(ArgReduceKind::Argmin);
    assert_eq!(got, EXPECT_ARGMIN_AXIS1, "argmin i32 mismatch");
}

#[test]
#[ignore]
fn argmin_axis1_f32_i64_output() {
    let got = run_arg_axis::<i64>(ArgReduceKind::Argmin);
    assert_eq!(got, EXPECT_ARGMIN_AXIS1, "argmin i64 mismatch");
}

// =============================================================================
// SKU distinctness check — verifies the plan layer routes u32 vs i64
// output to *different* kernel SKUs (so an alpha.27 i64-only caller and
// an alpha.28 u32 caller don't collide on the same code path).
//
// Compile-time / select-time test: doesn't need GPU dispatch beyond
// successful `select`, so it doesn't need `#[ignore]`-gating beyond the
// driver init.
// =============================================================================

#[test]
#[ignore]
fn sku_differs_between_u32_and_i64_output() {
    let (_ctx, stream) = setup();
    let desc = ArgReduceDescriptor {
        kind: ArgReduceKind::Argmax,
        input_shape: [2i32, 4i32],
        reduce_axis: 1,
        element: ElementKind::F32,
    };
    let plan_u32 = ArgReducePlan::<f32, 2, u32>::select(&stream, &desc, PlanPreference::default())
        .expect("select u32");
    let plan_i64 = ArgReducePlan::<f32, 2, i64>::select(&stream, &desc, PlanPreference::default())
        .expect("select i64");
    let sku_u32 = plan_u32.sku();
    let sku_i64 = plan_i64.sku();
    // The two SKUs must differ — otherwise the plan layer couldn't tell
    // them apart at dispatch time. The current implementation
    // distinguishes them through `aux_element` (`None` for u32,
    // `Some(I64)` for i64).
    assert_ne!(
        sku_u32.aux_element, sku_i64.aux_element,
        "ArgReducePlan SKUs for u32 vs i64 output must differ (got {:?} vs {:?})",
        sku_u32.aux_element, sku_i64.aux_element
    );
}
