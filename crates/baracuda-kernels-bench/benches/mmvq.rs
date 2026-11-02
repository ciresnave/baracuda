//! GGUF MMVQ throughput — baracuda self-bench across block formats,
//! shapes, and activation dtypes.
//!
//! No reference equivalent: cuBLAS / cuDNN don't expose a GGUF MMVQ
//! primitive. This bench exists as the **objective measurement basis**
//! for the Tier A k-quant micro-optimizations.
//!
//! Sweeps:
//! - `nrows × ncols ∈ {(4096, 4096), (11008, 4096), (32000, 4096)}`.
//!   These mirror the three matmul shapes in Llama-2 7B's transformer
//!   layers: Q/K/V projections (square 4096), FFN up_proj (11008×4096),
//!   and the LM head (32000×4096).
//! - Block formats: `{Q4_0, Q4_K, Q6_K, Q8_0}`. Covers two 4-bit, one
//!   6-bit k-quant, and one 8-bit. Bench `ncols = 4096` cleanly divides
//!   every k-quant block size (256).
//! - Activations: `{f32, f16, bf16}`. Phase 18 fanout.

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::quantize::gguf::mmvq::GgufMmvqActivation;
use baracuda_kernels::{
    GgufBlockFormat, GgufMmvqArgs, GgufMmvqDescriptor, GgufMmvqPlan, PlanPreference, TensorMut,
    TensorRef, U8, Workspace,
};
use baracuda_kernels_bench::{
    CROSS_MMVQ_FORMATS, CROSS_MMVQ_SHAPES, LiveScalar, PhaseTwentyNineRow, append_csv_row,
    assert_cell_live, measure_median_ns, setup_device, time_with_events, warmup,
};
use baracuda_kernels_types::contiguous_stride;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use half::{bf16, f16};

const BENCH_NAME: &str = "mmvq";

fn block_label(fmt: GgufBlockFormat) -> &'static str {
    match fmt {
        GgufBlockFormat::Q4_0 => "q4_0",
        GgufBlockFormat::Q4_1 => "q4_1",
        GgufBlockFormat::Q5_0 => "q5_0",
        GgufBlockFormat::Q5_1 => "q5_1",
        GgufBlockFormat::Q8_0 => "q8_0",
        GgufBlockFormat::Q2K => "q2_k",
        GgufBlockFormat::Q3K => "q3_k",
        GgufBlockFormat::Q4K => "q4_k",
        GgufBlockFormat::Q5K => "q5_k",
        GgufBlockFormat::Q6K => "q6_k",
        GgufBlockFormat::Q8K => "q8_k",
        _ => "unknown",
    }
}

/// One benchmark cell: device buffers + selected plan for a (format, shape).
struct Cell<T: GgufMmvqActivation> {
    nrows: i32,
    ncols: i32,
    weight_bytes: usize,
    dev_w: DeviceBuffer<U8>,
    dev_act: DeviceBuffer<T>,
    dev_out: DeviceBuffer<T>,
    plan: GgufMmvqPlan<T>,
}

impl<T> Cell<T>
where
    T: GgufMmvqActivation + Copy + 'static + baracuda_kernels::Element,
{
    /// Allocate buffers and select the plan; `None` skips the cell (as the
    /// original inline `continue`s did: non-tiling width, allocation or select failure).
    fn build(
        ctx: &baracuda_driver::Context,
        stream: &baracuda_driver::Stream,
        fmt: GgufBlockFormat,
        (nrows, ncols): (i32, i32),
        fill: T,
    ) -> Option<Self> {
        let bs = fmt.block_size() as i32;
        if ncols % bs != 0 {
            return None;
        }
        let weight_bytes = (nrows as usize) * ((ncols / bs) as usize) * fmt.type_size();
        let host_w: Vec<U8> = vec![U8(0x10); weight_bytes];
        let host_act: Vec<T> = vec![fill; ncols as usize];
        let dev_w = DeviceBuffer::from_slice(ctx, &host_w).ok()?;
        let dev_act = DeviceBuffer::from_slice(ctx, &host_act).ok()?;
        let dev_out: DeviceBuffer<T> = DeviceBuffer::zeros(ctx, nrows as usize).ok()?;
        let desc = GgufMmvqDescriptor::new(nrows, ncols, fmt);
        let plan = GgufMmvqPlan::<T>::select(stream, &desc, PlanPreference::default()).ok()?;
        Some(Self {
            nrows,
            ncols,
            weight_bytes,
            dev_w,
            dev_act,
            dev_out,
            plan,
        })
    }

    /// Launch the plan once on contiguous tensors.
    fn run_once(&mut self, stream: &baracuda_driver::Stream, what: &str) {
        let w_shape = [self.weight_bytes as i32];
        let act_shape = [self.ncols];
        let out_shape = [self.nrows];
        let args = GgufMmvqArgs::<T> {
            weight: TensorRef::new(self.dev_w.as_slice(), w_shape, contiguous_stride(w_shape))
                .unwrap(),
            activation: TensorRef::new(
                self.dev_act.as_slice(),
                act_shape,
                contiguous_stride(act_shape),
            )
            .unwrap(),
            output: TensorMut::new(
                self.dev_out.as_slice_mut(),
                out_shape,
                contiguous_stride(out_shape),
            )
            .unwrap(),
        };
        self.plan.run(stream, Workspace::None, args).expect(what);
    }
}

/// Measure one cell, record its CSV row, and register its criterion benchmark.
fn bench_cell<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    ctx: &baracuda_driver::Context,
    stream: &baracuda_driver::Stream,
    labels: (&str, &str),
    mut cell: Cell<T>,
) where
    T: GgufMmvqActivation + Copy + 'static + baracuda_kernels::Element + LiveScalar,
{
    let (act_label, fmt_lbl) = labels;
    let shape = format!("N{}_C{}", cell.nrows, cell.ncols);
    warmup(stream, || cell.run_once(stream, "mmvq warmup"));

    // Liveness (NOT a reference): warmup populated dev_out; assert live
    // before timing — a bench over garbage output is fast and meaningless.
    assert_cell_live(
        &format!("mmvq/{act_label}/{fmt_lbl}/{shape}"),
        &cell.dev_out,
        cell.nrows as usize,
    );
    let baracuda_ns = measure_median_ns(ctx, stream, 11, 100, || cell.run_once(stream, "mmvq run"));
    append_csv_row(
        BENCH_NAME,
        &PhaseTwentyNineRow {
            op: "mmvq",
            shape: format!("{fmt_lbl}_{shape}"),
            dtype: leak_str(act_label),
            baracuda_ns,
            reference_ns: None,
            reference: "",
            pytorch_ns: None,
        },
    );

    group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
        bb.iter_custom(|iters| {
            time_with_events(ctx, stream, iters, || cell.run_once(stream, "mmvq run"))
        })
    });
}

/// Generic MMVQ bench body — one (block format × activation dtype) group.
fn bench_mmvq<T>(c: &mut Criterion, act_label: &str, fill: T)
where
    T: GgufMmvqActivation + Copy + 'static + baracuda_kernels::Element + LiveScalar,
{
    let (ctx, stream) = setup_device();

    for &fmt in CROSS_MMVQ_FORMATS {
        let fmt_lbl = block_label(fmt);
        let mut group = c.benchmark_group(format!("mmvq/{act_label}/{fmt_lbl}"));
        for &shape in CROSS_MMVQ_SHAPES {
            if let Some(cell) = Cell::<T>::build(&ctx, &stream, fmt, shape, fill) {
                bench_cell(&mut group, &ctx, &stream, (act_label, fmt_lbl), cell);
            }
        }
        group.finish();
    }
}

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

/// Top-level criterion entry - invoked by criterion_main!.
fn mmvq_benches(c: &mut Criterion) {
    bench_mmvq::<f32>(c, "f32", 1.0_f32);
    bench_mmvq::<f16>(c, "f16", f16::ONE);
    bench_mmvq::<bf16>(c, "bf16", bf16::ONE);
}

// `criterion_group!` expands into a `pub fn benches_grp` whose
// signature is fixed by the macro - can't doc-comment it directly, so
// suppress the workspace `missing_docs = deny` lint on the generated fn.
#[allow(
    missing_docs,
    reason = "criterion_group!/criterion_main! expand to undocumented public items"
)]
mod criterion_glue {
    use super::*;
    criterion_group!(benches_grp, mmvq_benches);
}
criterion_main!(criterion_glue::benches_grp);
