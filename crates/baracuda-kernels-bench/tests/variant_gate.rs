//! The **automated variant-gate loop**, end to end and on-device (phase 2):
//!
//! ```text
//! generate_variants → nvrtc compile → driver launch → numeric oracle
//!     → gate_cell (CUDA-event medians) → winner_of → merge over the seed
//!     → emit_dispatch_table (the committed routing artifact)
//! ```
//!
//! Runs on the split-K outer-axis reduction cell at a **starved** shape
//! (cols = 1024 → the baseline's one-thread-per-column schedule fills only 4
//! blocks; the sweep measured the variant ~4× faster there), so the gate should
//! elect the split-K variant and the artifact should record its entry point.
//!
//! Ignored by default (needs a CUDA device + nvrtc); run with:
//! `cargo test -p baracuda-kernels-bench --test variant_gate -- --ignored`

use baracuda_driver::{Device, Module};
use baracuda_kernelgen::{
    generate_variants, input, Compiler, Cuda, NvrtcCompiler, OpDef, ReduceOp, VariantFidelity,
};
use baracuda_kernelgen::emit_dispatch_table;
use baracuda_kernels_bench::{current_hwstamp, gate_cell, setup_device};
use baracuda_kernels_types::{
    merge, structure_key, ArchSku, AxisMask, DispatchEntry, DispatchTable, ElementKind,
    Implementor, OpCategory, OperandDesc, Provenance,
};
use baracuda_driver::DeviceBuffer;

const ROWS: i64 = 16_384;
const COLS: i64 = 1_024;
const N_CHUNKS: i64 = 256;
const BLOCK: u32 = 256;

#[test]
#[ignore = "requires a CUDA device + nvrtc"]
fn variant_gate_loop_end_to_end() {
    let (ctx, stream) = setup_device();
    let device = Device::get(0).expect("device");
    let stamp = current_hwstamp(&device).expect("hwstamp");
    if stamp.arch != ArchSku::Sm89 {
        eprintln!("skipping: cell is keyed sm89, device is {:?}", stamp.arch);
        return;
    }

    // ---- 1. Generate the variant set for the outer-axis sum cell. ----
    let op = OpDef::reduction_axes(
        "sum",
        1,
        &[ElementKind::F32],
        input(0),
        ReduceOp::Sum,
        AxisMask(0b1),
        false,
    );
    let a = OperandDesc::new(2, &[ROWS, COLS], &[COLS, 1], ElementKind::F32, 256);
    let o = OperandDesc::new(1, &[COLS], &[1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
    let variants = generate_variants(&op, &key, &Cuda);
    assert_eq!(variants.len(), 2, "base + splitk");
    assert_eq!(variants[1].fidelity, VariantFidelity::ReassociatedDeterministic);

    // ---- 2. nvrtc-compile every kernel of every variant; load via the driver. ----
    let compiler = NvrtcCompiler::new(ArchSku::Sm89);
    let mut modules = Vec::new(); // keep alive for the Functions' lifetime
    for v in &variants {
        for k in &v.kernels {
            let ptx = compiler
                .compile(&k.source, &k.name, 30_000)
                .unwrap_or_else(|e| panic!("nvrtc({}) failed: {e}", k.name));
            let ptx = String::from_utf8(ptx).expect("ptx is text");
            let module = Module::load_ptx(&ctx, &ptx).expect("module load");
            modules.push((k.name.clone(), module));
        }
    }
    let func = |name: &str| {
        let (_, m) = modules
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("module for {name}"));
        m.get_function(name).expect("get_function")
    };
    let base_name = variants[0].kernels[0].name.clone();
    let partial_name = variants[1].kernels[0].name.clone();
    let combine_name = variants[1].kernels[1].name.clone();
    let f_base = func(&base_name);
    let f_partial = func(&partial_name);
    let f_combine = func(&combine_name);

    // ---- 3. Buffers + a CPU f64 oracle. ----
    let n = (ROWS * COLS) as usize;
    let host: Vec<f32> = (0..n).map(|i| ((i % 37) as f32 - 18.0) * 0.25).collect();
    let mut oracle = vec![0.0f64; COLS as usize];
    for r in 0..ROWS as usize {
        for c in 0..COLS as usize {
            oracle[c] += f64::from(host[r * COLS as usize + c]);
        }
    }
    let d_in = DeviceBuffer::from_slice(&ctx, &host).expect("d_in");
    let d_out = DeviceBuffer::<f32>::new(&ctx, COLS as usize).expect("d_out");
    let d_ws = DeviceBuffer::<f32>::new(&ctx, (N_CHUNKS * COLS) as usize).expect("d_ws");
    // Extraction #1: dims ride by value as flattened scalar params.
    const ONE: i64 = 1;

    let col_tiles = ((COLS as u32) + BLOCK - 1) / BLOCK;
    let chunk_rows: i64 = (ROWS + N_CHUNKS - 1) / N_CHUNKS;

    let launch_base = || {
        // SAFETY: argument list matches the generated general-path signature
        // (in0, out, shape0, shape1, s0_0, s0_1, so_0, n_out) — dims by value
        // per extraction #1; buffers outlive the launch.
        unsafe {
            f_base
                .launch()
                .grid(col_tiles)
                .block(BLOCK)
                .stream(&stream)
                .arg(&d_in)
                .arg(&d_out)
                .arg(&ROWS)
                .arg(&COLS)
                .arg(&COLS)
                .arg(&ONE)
                .arg(&ONE)
                .arg(&COLS)
                .launch()
                .expect("base launch");
        }
    };
    let launch_splitk = || {
        // SAFETY: matches the split-K pair's signatures + two-launch protocol
        // (partial: in0, ws, rows, cols, chunk_rows; combine: ws, out, cols,
        // n_chunks); buffers outlive both launches.
        unsafe {
            f_partial
                .launch()
                .grid((col_tiles, N_CHUNKS as u32))
                .block(BLOCK)
                .stream(&stream)
                .arg(&d_in)
                .arg(&d_ws)
                .arg(&ROWS)
                .arg(&COLS)
                .arg(&chunk_rows)
                .launch()
                .expect("partial launch");
            f_combine
                .launch()
                .grid(col_tiles)
                .block(BLOCK)
                .stream(&stream)
                .arg(&d_ws)
                .arg(&d_out)
                .arg(&COLS)
                .arg(&N_CHUNKS)
                .launch()
                .expect("combine launch");
        }
    };

    // ---- 4. Numeric oracle FIRST (gate_cell's correctness precondition). ----
    let check = |name: &str, launch: &dyn Fn()| {
        launch();
        stream.synchronize().expect("sync");
        let mut got = vec![0.0f32; COLS as usize];
        d_out.copy_to_host(&mut got).expect("copy");
        for c in 0..COLS as usize {
            let denom = oracle[c].abs().max(1.0);
            let rel = (f64::from(got[c]) - oracle[c]).abs() / denom;
            assert!(rel < 1e-5, "{name}: col {c} rel err {rel}");
        }
    };
    check("base", &launch_base);
    check("splitk", &launch_splitk);

    // ---- 5. Gate: time both candidates, reduce to a measured decision. ----
    let entry = gate_cell(
        &ctx,
        &stream,
        &key,
        Some(stamp),
        7,  // samples
        10, // inner iterations per sample
        vec![
            (
                Implementor::Generated,
                Some(base_name.clone()),
                Box::new(launch_base),
            ),
            (
                Implementor::Generated,
                Some(partial_name.clone()),
                Box::new(launch_splitk),
            ),
        ],
    )
    .expect("measured entry");
    assert_eq!(entry.provenance, Provenance::Measured);
    assert_eq!(entry.ranked.len(), 2, "both candidates ranked");
    eprintln!(
        "gate: winner {:?} margin {:.2} ({} vs {})",
        entry.winner_entry, entry.margin, entry.ranked[0].median_ns, entry.ranked[1].median_ns
    );
    // At this starved shape the sweep measured the variant ~4x faster — the
    // gate must elect it (margin well past MIN_FLIP_MARGIN noise).
    assert_eq!(entry.winner_entry.as_deref(), Some(partial_name.as_str()));

    // ---- 6. Merge over the seeded default; emit the committed artifact. ----
    let mut table = DispatchTable::from_entries(vec![DispatchEntry::seeded(
        key.to_token(),
        Implementor::Generated,
    )]);
    merge(&mut table, &[entry]);
    let routed = table.lookup(&key).expect("routed");
    assert_eq!(routed.provenance, Provenance::Measured, "seed upgraded");
    assert_eq!(routed.winner_entry.as_deref(), Some(partial_name.as_str()));

    let artifact = emit_dispatch_table(&table);
    assert!(
        artifact.contains(&partial_name),
        "artifact names the winning variant's entry point"
    );
    eprintln!("--- committed artifact ---\n{artifact}");
}

/// The cross-pass materialization variant (Softmax's shared `exp(x−max)` cached
/// in a per-row smem tile), through the same automated loop — with the stronger
/// oracle its `BitIdentical` fidelity earns: the two kernels' outputs must be
/// **memcmp-identical**, not merely close.
#[test]
#[ignore = "requires a CUDA device + nvrtc"]
fn smemrow_variant_is_bit_identical_and_gated() {
    use baracuda_kernelgen::{reduced, ReduceStage};

    let (ctx, stream) = setup_device();
    let device = Device::get(0).expect("device");
    let stamp = current_hwstamp(&device).expect("hwstamp");
    if stamp.arch != ArchSku::Sm89 {
        eprintln!("skipping: cell is keyed sm89, device is {:?}", stamp.arch);
        return;
    }

    const RR_ROWS: i64 = 4_096;
    const K: i64 = 4_096; // 16 KB f32 row cache — inside the default smem window
    let softmax = OpDef::row_reduce(
        "softmax",
        1,
        &[ElementKind::F32],
        vec![
            ReduceStage { pre: input(0).0, op: ReduceOp::Max },
            ReduceStage { pre: (input(0) - reduced(0)).exp().0, op: ReduceOp::Sum },
        ],
        (input(0) - reduced(0)).exp() / reduced(1),
    );
    let x = OperandDesc::new(2, &[RR_ROWS, K], &[K, 1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::Softmax, &[x, x], ArchSku::Sm89);
    let variants = generate_variants(&softmax, &key, &Cuda);
    assert_eq!(variants.len(), 2, "base + smemrow");
    assert_eq!(variants[1].fidelity, VariantFidelity::BitIdentical);

    let compiler = NvrtcCompiler::new(ArchSku::Sm89);
    let mut modules = Vec::new();
    for v in &variants {
        for k in &v.kernels {
            let ptx = compiler
                .compile(&k.source, &k.name, 30_000)
                .unwrap_or_else(|e| panic!("nvrtc({}) failed: {e}", k.name));
            let ptx = String::from_utf8(ptx).expect("ptx is text");
            modules.push((k.name.clone(), Module::load_ptx(&ctx, &ptx).expect("load")));
        }
    }
    let func = |name: &str| {
        let (_, m) = modules.iter().find(|(n, _)| n == name).expect("module");
        m.get_function(name).expect("get_function")
    };
    let base_name = variants[0].kernels[0].name.clone();
    let smem_name = variants[1].kernels[0].name.clone();
    let f_base = func(&base_name);
    let f_smem = func(&smem_name);

    // Inputs spanning negatives, spikes, and a NaN-free varied surface (a NaN
    // row saturates to NaN identically in both — covered by unit reasoning; the
    // bit-compare here wants a live numeric surface).
    let n = (RR_ROWS * K) as usize;
    let host: Vec<f32> = (0..n).map(|i| ((i % 251) as f32 - 125.0) * 0.05).collect();
    let d_in = DeviceBuffer::from_slice(&ctx, &host).expect("d_in");
    let d_base = DeviceBuffer::<f32>::new(&ctx, n).expect("d_base");
    let d_smem = DeviceBuffer::<f32>::new(&ctx, n).expect("d_smem");
    let smem_bytes = (K as u32) * 4;

    let launch_base = || {
        // SAFETY: matches the rowreduce signature (in0, out, n_out, k).
        unsafe {
            f_base
                .launch()
                .grid(RR_ROWS as u32)
                .block(256)
                .stream(&stream)
                .arg(&d_in)
                .arg(&d_base)
                .arg(&RR_ROWS)
                .arg(&K)
                .launch()
                .expect("base launch");
        }
    };
    let launch_smem = || {
        // SAFETY: same signature + the variant's dynamic smem (k * 4 bytes).
        unsafe {
            f_smem
                .launch()
                .grid(RR_ROWS as u32)
                .block(256)
                .shared_mem_bytes(smem_bytes)
                .stream(&stream)
                .arg(&d_in)
                .arg(&d_smem)
                .arg(&RR_ROWS)
                .arg(&K)
                .launch()
                .expect("smem launch");
        }
    };

    // The BitIdentical oracle: raw u32 output buffers must match exactly.
    launch_base();
    launch_smem();
    stream.synchronize().expect("sync");
    let mut a = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    d_base.copy_to_host(&mut a).expect("copy a");
    d_smem.copy_to_host(&mut b).expect("copy b");
    for i in 0..n {
        assert_eq!(
            a[i].to_bits(),
            b[i].to_bits(),
            "bit mismatch at {i}: base {:08x} smemrow {:08x}",
            a[i].to_bits(),
            b[i].to_bits()
        );
    }

    // Gate the pair; report the measured margin (regime-dependent — occupancy
    // vs saved traffic — so no winner is asserted; the table records it).
    let entry = gate_cell(
        &ctx,
        &stream,
        &key,
        Some(stamp),
        7,
        10,
        vec![
            (Implementor::Generated, Some(base_name), Box::new(launch_base)),
            (Implementor::Generated, Some(smem_name), Box::new(launch_smem)),
        ],
    )
    .expect("measured entry");
    assert_eq!(entry.ranked.len(), 2);
    eprintln!(
        "softmax gate: winner {:?} margin {:.3} ({:.0} vs {:.0} ns)",
        entry.winner_entry, entry.margin, entry.ranked[0].median_ns, entry.ranked[1].median_ns
    );
}
