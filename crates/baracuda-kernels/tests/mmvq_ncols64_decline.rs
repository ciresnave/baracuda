//! #127: type-0/1 MMVQ declines widths that are not multiples of 64.
//!
//! The shared type-0/1 MMVQ kernel reads whole 64-column strides, so a width
//! with `ncols % 64 == 32` reads 32 columns past `ncols`: out of bounds on the
//! activation, and on the weight for the last row. Measured on an RTX 4070 under
//! compute-sanitizer memcheck: Q8_0 `nrows=2, ncols=32` gives 32 invalid reads.
//!
//! The decline tests use widths the plans previously ACCEPTED (they only
//! required a multiple of the 32-element block), so they fail if the decline is
//! removed. Each decline must carry the `#127` message, so an unrelated
//! `InvalidProblem` cannot satisfy it. The acceptance tests pin that valid
//! neighbouring widths still pass.
//!
//! `GgufMmvqMultiMPlan` is deliberately NOT declined: its loop is bounded by
//! whole blocks and does not over-read. `multim_plan_still_accepts_96` pins that.

use baracuda_driver::{Context, Device, Stream, init};
use baracuda_kernels::{
    Error, GgufBlockFormat, GgufMmvqBatchedDescriptor, GgufMmvqBatchedFormat, GgufMmvqBatchedPlan,
    GgufMmvqDescriptor, GgufMmvqMultiMDescriptor, GgufMmvqMultiMPlan, GgufMmvqPlan, PlanPreference,
};

const TYPE_0_1: [GgufBlockFormat; 5] = [
    GgufBlockFormat::Q4_0,
    GgufBlockFormat::Q4_1,
    GgufBlockFormat::Q5_0,
    GgufBlockFormat::Q5_1,
    GgufBlockFormat::Q8_0,
];

/// Multiples of the 32-element block that are NOT multiples of 64. `96` and
/// `160` come first because both plans accepted them before #127 (the batched
/// plan's old debug-only guard rejected only widths below 64), so a missing
/// decline fails on an ACCEPTED width rather than on a different error. `32`
/// is kept as the smallest such width.
const DECLINED_WIDTHS: [i32; 3] = [96, 160, 32];

/// Multiples of 64, which must still be accepted.
const ACCEPTED_WIDTHS: [i32; 2] = [64, 128];

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn single(stream: &Stream, fmt: GgufBlockFormat, ncols: i32) -> Result<(), Error> {
    let desc = GgufMmvqDescriptor {
        nrows: 1,
        ncols,
        block_format: fmt,
        w_start_byte_offset: 0,
    };
    GgufMmvqPlan::<f32>::select(stream, &desc, PlanPreference::default()).map(|_| ())
}

fn batched(stream: &Stream, fmt: GgufBlockFormat, n_cols: i32) -> Result<(), Error> {
    let desc = GgufMmvqBatchedDescriptor {
        n_experts: 1,
        n_rows_per_expert: 1,
        n_cols,
        m_total: 1,
        top_k: 1,
        format: GgufMmvqBatchedFormat::Quantized(fmt),
    };
    GgufMmvqBatchedPlan::<f32>::select(stream, &desc, PlanPreference::default()).map(|_| ())
}

fn assert_declined_for_127(r: Result<(), Error>, what: &str) {
    match r {
        Err(Error::InvalidProblem(m)) if m.contains("#127") => {}
        Err(e) => panic!("{what}: expected the #127 width decline, got a different error: {e:?}"),
        Ok(()) => panic!("{what}: expected the #127 width decline, got Ok"),
    }
}

#[test]
#[ignore]
fn single_plan_declines_type01_widths_not_multiple_of_64() {
    let (_ctx, stream) = setup();
    for fmt in TYPE_0_1 {
        for ncols in DECLINED_WIDTHS {
            assert_declined_for_127(
                single(&stream, fmt, ncols),
                &format!("GgufMmvqPlan {fmt:?} ncols={ncols}"),
            );
        }
    }
}

#[test]
#[ignore]
fn single_plan_accepts_type01_widths_multiple_of_64() {
    let (_ctx, stream) = setup();
    for fmt in TYPE_0_1 {
        for ncols in ACCEPTED_WIDTHS {
            if let Err(e) = single(&stream, fmt, ncols) {
                panic!("GgufMmvqPlan {fmt:?} ncols={ncols} must be accepted: {e:?}");
            }
        }
    }
}

#[test]
#[ignore]
fn single_plan_kquants_unaffected() {
    let (_ctx, stream) = setup();
    for fmt in [
        GgufBlockFormat::Q4K,
        GgufBlockFormat::Q6K,
        GgufBlockFormat::Q8K,
    ] {
        if let Err(e) = single(&stream, fmt, 256) {
            panic!("GgufMmvqPlan {fmt:?} ncols=256 must be accepted: {e:?}");
        }
    }
}

#[test]
#[ignore]
fn batched_plan_declines_type01_widths_not_multiple_of_64() {
    let (_ctx, stream) = setup();
    for fmt in TYPE_0_1 {
        for n_cols in DECLINED_WIDTHS {
            assert_declined_for_127(
                batched(&stream, fmt, n_cols),
                &format!("GgufMmvqBatchedPlan {fmt:?} n_cols={n_cols}"),
            );
        }
    }
}

#[test]
#[ignore]
fn batched_plan_accepts_type01_widths_multiple_of_64() {
    let (_ctx, stream) = setup();
    for fmt in TYPE_0_1 {
        for n_cols in ACCEPTED_WIDTHS {
            if let Err(e) = batched(&stream, fmt, n_cols) {
                panic!("GgufMmvqBatchedPlan {fmt:?} n_cols={n_cols} must be accepted: {e:?}");
            }
        }
    }
}

/// Scope pin: the multi-matrix plan's loop is block-bounded, so it must keep
/// accepting widths the other two plans now decline.
#[test]
#[ignore]
fn multim_plan_still_accepts_96() {
    let (_ctx, stream) = setup();
    for fmt in TYPE_0_1 {
        let desc = GgufMmvqMultiMDescriptor {
            nrows: 1,
            ncols: 96,
            m: 1,
            block_format: fmt,
            w_start_byte_offset: 0,
        };
        if let Err(e) = GgufMmvqMultiMPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        {
            panic!("GgufMmvqMultiMPlan {fmt:?} ncols=96 must be accepted: {e:?}");
        }
    }
}
