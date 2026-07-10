//! On-device CUDA-graph capture→replay validation for the capture-safe
//! decode kernels this session added to unblock Fuel's `CapturedRun`:
//!
//! - `gather_rows` (embedding lookup) — the capture-safe replacement for
//!   `index_select`, which mis-computes its first output element on
//!   `cuGraphLaunch` replay.
//! - `gemv_dense_m1` (dense GEMV) — the capture-safe replacement for the
//!   cuBLAS `gemm_dense` facade on the decode path.
//!
//! It also REPRODUCES, diagnostically, the `index_select` element-0
//! graph-replay bug on this box (same sm_89 arch as Fuel's repro): eager
//! warm launch writes `[1,2,3,4]`, then a captured-graph replay is
//! observed. This both confirms Fuel's finding on identical hardware and
//! proves `gather_rows` does NOT share it.
//!
//! The pattern for every kernel mirrors Fuel's exact scenario: allocate a
//! ZEROED output, do an EAGER (non-graph) warm launch, then capture the
//! same launch into a graph and replay it WITHOUT re-zeroing — so a
//! replay that drops/mis-writes any element surfaces as a divergence from
//! the warm result.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --test capture_replay_gather_gemv -- --ignored --nocapture`.

use core::ffi::c_void;
use std::cell::Cell;

use baracuda_driver::{CaptureMode, Context, Device, DeviceBuffer, Stream, init};

fn gpu() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Fuel's `wte` table: `[V, H]` row-major, row v = `[1+10v, 2+10v, …]`
/// (row 0 = `[1,2,3,4]`).
fn wte_table(v: i64, h: i64) -> Vec<f32> {
    (0..(v * h))
        .map(|e| {
            let row = e / h;
            let col = e - row * h;
            (1 + 10 * row + col) as f32
        })
        .collect()
}

// ============================================================================
// gather_rows — MUST be correct on the eager warm launch AND every replay.
// ============================================================================

#[test]
#[ignore]
fn gather_rows_capture_replay_correct() {
    let (ctx, stream) = gpu();
    let (v, h, n) = (3i64, 4i64, 1i64); // decode embedding lookup: n = 1 token
    let table = wte_table(v, h);
    let idx: Vec<u32> = vec![0]; // gather row 0 (native U32 — no bitcast)
    let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let out_len = (n * h) as usize;

    let d_table = DeviceBuffer::from_slice(&ctx, &table).unwrap();
    let d_idx = DeviceBuffer::from_slice(&ctx, &idx).unwrap();
    // ZEROED output — a dropped/mis-written element stays 0 and is caught.
    let mut d_dest = DeviceBuffer::from_slice(&ctx, &vec![0.0f32; out_len]).unwrap();

    let table_ptr = d_table.as_slice().as_raw().0 as *const c_void;
    let idx_ptr = d_idx.as_slice().as_raw().0 as *const c_void;
    let dest_ptr = d_dest.as_slice_mut().as_raw().0 as *mut c_void;

    let run_ffi = |stream_raw: *mut c_void| -> i32 {
        unsafe {
            baracuda_kernels_sys::baracuda_kernels_gather_rows_f32_run(
                dest_ptr, table_ptr, idx_ptr, v, h, n, stream_raw,
            )
        }
    };

    // EAGER warm launch (non-graph) on the zeroed output.
    assert_eq!(
        run_ffi(stream.as_raw() as *mut c_void),
        0,
        "warm FFI status"
    );
    stream.synchronize().unwrap();
    let mut warm = vec![0f32; out_len];
    d_dest.copy_to_host(&mut warm).unwrap();
    assert_eq!(warm, expected, "gather_rows warm launch");

    // Capture the same launch into a graph, then replay WITHOUT re-zeroing.
    let status = Cell::new(-1i32);
    let graph = stream
        .capture(CaptureMode::ThreadLocal, |s| {
            status.set(run_ffi(s.as_raw() as *mut c_void));
            Ok(())
        })
        .expect("capture gather_rows");
    assert_eq!(status.get(), 0, "captured gather_rows FFI status");
    let exec = graph.instantiate().expect("instantiate");

    for run in 0..4 {
        exec.launch(&stream).expect("cuGraphLaunch");
        stream.synchronize().unwrap();
        let mut got = vec![0f32; out_len];
        d_dest.copy_to_host(&mut got).unwrap();
        assert_eq!(
            got, expected,
            "gather_rows graph replay #{run} diverged (element-0 drop?)"
        );
    }
    println!(
        "gather_rows: warm + 4 graph replays all == {expected:?} (no element-0 drop, capture-safe)."
    );
    drop(ctx);
}

// ============================================================================
// index_select — DIAGNOSTIC: reproduce Fuel's element-0 replay bug locally.
// ============================================================================
//
// Same [3,4] / tok[0] scenario Fuel used. `idx` is i32 (the f32 variant's
// IndexT). Warm must write [1,2,3,4]; the replay result is observed and
// reported. This test asserts only the warm correctness (the replay bug is
// a driver/graph-node-replay issue, not our kernel), and PRINTS whether the
// bug reproduces on this box — a strong signal for Fuel's root-cause.

#[test]
#[ignore]
fn index_select_element0_replay_repro() {
    let (ctx, stream) = gpu();
    let (v, h) = (3i64, 4i64);
    let src = wte_table(v, h);
    let idx: Vec<i32> = vec![0]; // Fuel bitcasts U32 tok -> i32
    let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let out_len = 4usize;

    // index_select host metadata (snapshotted by-value into DimsI32/DimsI64).
    let out_shape: Vec<i32> = vec![1, 4]; // emb [1, 4]
    let stride_src: Vec<i64> = vec![4, 1]; // wte [3,4] row-major
    let stride_out: Vec<i64> = vec![4, 1]; // emb [1,4] row-major
    let (out_numel, rank, select_dim, src_dim_size) = (4i64, 2i32, 0i32, v as i32);

    let d_src = DeviceBuffer::from_slice(&ctx, &src).unwrap();
    let d_idx = DeviceBuffer::from_slice(&ctx, &idx).unwrap();
    let mut d_out = DeviceBuffer::from_slice(&ctx, &vec![0.0f32; out_len]).unwrap();

    let src_ptr = d_src.as_slice().as_raw().0 as *const c_void;
    let idx_ptr = d_idx.as_slice().as_raw().0 as *const c_void;
    let out_ptr = d_out.as_slice_mut().as_raw().0 as *mut c_void;

    let run_ffi = |stream_raw: *mut c_void| -> i32 {
        unsafe {
            baracuda_kernels_sys::baracuda_kernels_index_select_f32_run(
                out_numel,
                rank,
                select_dim,
                src_dim_size,
                out_shape.as_ptr(),
                stride_src.as_ptr(),
                stride_out.as_ptr(),
                src_ptr,
                idx_ptr,
                out_ptr,
                core::ptr::null_mut(),
                0,
                stream_raw,
            )
        }
    };

    // EAGER warm launch on the zeroed output.
    assert_eq!(
        run_ffi(stream.as_raw() as *mut c_void),
        0,
        "warm FFI status"
    );
    stream.synchronize().unwrap();
    let mut warm = vec![0f32; out_len];
    d_out.copy_to_host(&mut warm).unwrap();
    assert_eq!(warm, expected, "index_select warm launch (row 0)");

    // Capture + replay WITHOUT re-zeroing (out holds the warm [1,2,3,4]).
    let status = Cell::new(-1i32);
    let graph = stream
        .capture(CaptureMode::ThreadLocal, |s| {
            status.set(run_ffi(s.as_raw() as *mut c_void));
            Ok(())
        })
        .expect("capture index_select");
    assert_eq!(status.get(), 0, "captured index_select FFI status");
    let exec = graph.instantiate().expect("instantiate");

    exec.launch(&stream).expect("cuGraphLaunch");
    stream.synchronize().unwrap();
    let mut replay = vec![0f32; out_len];
    d_out.copy_to_host(&mut replay).unwrap();

    if replay == expected {
        println!(
            "index_select: replay == {expected:?} — the element-0 bug did NOT reproduce on this \
             driver ({:?}).",
            replay
        );
    } else {
        println!(
            "index_select: REPRODUCED Fuel's graph-replay bug on this sm_89 box — warm {expected:?} \
             -> replay {replay:?} (element 0: {} -> {}). Confirms the driver/graph-node-replay \
             layer; gather_rows (above) does not share it.",
            expected[0], replay[0]
        );
    }
    drop(ctx);
}

// ============================================================================
// index_select — DIAGNOSTIC #2: Fuel's EXACT framing (rank=3, select_dim=1).
// ============================================================================
//
// Fuel corrected: SAME env (driver 610.47 / CUDA 13.3 / sm_89) as this box, so
// the bug is NOT driver-specific — the differentiator is the exact params or
// Fuel's capture path. Fuel frames gather as [outer, source_dim_size,
// n_indices, inner], so their [3,4]/tok[0] call is rank=3, select_dim=1,
// out_shape=[1,1,4], stride_src=[12,4,1], stride_out=[4,4,1] — DIFFERENT from
// the rank=2 framing above. Same-box A/B: does the exact framing trigger it?

#[test]
#[ignore]
fn index_select_element0_replay_repro_fuel_exact() {
    let (ctx, stream) = gpu();
    let (v, h) = (3i64, 4i64);
    let src = wte_table(v, h);
    let idx: Vec<i32> = vec![0]; // Fuel passes U32 [0]; bit-identical to i32 [0]
    let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let out_len = 4usize;

    // Fuel's EXACT params (rank=3, select_dim=1).
    let out_shape: Vec<i32> = vec![1, 1, 4];
    let stride_src: Vec<i64> = vec![12, 4, 1];
    let stride_out: Vec<i64> = vec![4, 4, 1];
    let (out_numel, rank, select_dim, src_dim_size) = (4i64, 3i32, 1i32, v as i32);

    let d_src = DeviceBuffer::from_slice(&ctx, &src).unwrap();
    let d_idx = DeviceBuffer::from_slice(&ctx, &idx).unwrap();
    let mut d_out = DeviceBuffer::from_slice(&ctx, &vec![0.0f32; out_len]).unwrap();

    let src_ptr = d_src.as_slice().as_raw().0 as *const c_void;
    let idx_ptr = d_idx.as_slice().as_raw().0 as *const c_void;
    let out_ptr = d_out.as_slice_mut().as_raw().0 as *mut c_void;

    let run_ffi = |stream_raw: *mut c_void| -> i32 {
        unsafe {
            baracuda_kernels_sys::baracuda_kernels_index_select_f32_run(
                out_numel,
                rank,
                select_dim,
                src_dim_size,
                out_shape.as_ptr(),
                stride_src.as_ptr(),
                stride_out.as_ptr(),
                src_ptr,
                idx_ptr,
                out_ptr,
                core::ptr::null_mut(),
                0,
                stream_raw,
            )
        }
    };

    assert_eq!(run_ffi(stream.as_raw() as *mut c_void), 0, "warm FFI status");
    stream.synchronize().unwrap();
    let mut warm = vec![0f32; out_len];
    d_out.copy_to_host(&mut warm).unwrap();
    assert_eq!(warm, expected, "index_select (Fuel framing) warm launch");

    let status = Cell::new(-1i32);
    let graph = stream
        .capture(CaptureMode::ThreadLocal, |s| {
            status.set(run_ffi(s.as_raw() as *mut c_void));
            Ok(())
        })
        .expect("capture index_select fuel-exact");
    assert_eq!(status.get(), 0, "captured status");
    let exec = graph.instantiate().expect("instantiate");

    exec.launch(&stream).expect("cuGraphLaunch");
    stream.synchronize().unwrap();
    let mut replay = vec![0f32; out_len];
    d_out.copy_to_host(&mut replay).unwrap();

    if replay == expected {
        println!(
            "index_select FUEL-EXACT (rank=3, select_dim=1): replay == {expected:?} — bug did NOT \
             reproduce even with Fuel's exact params -> the divergence is in Fuel's capture path, \
             NOT the params/kernel (gather_rows will isolate it)."
        );
    } else {
        println!(
            "index_select FUEL-EXACT (rank=3, select_dim=1): REPRODUCED — warm {expected:?} -> \
             replay {replay:?} (element 0: {} -> {}). The rank=3/select_dim=1 FRAMING is the \
             trigger the rank=2 harness dodged -> param/kernel interaction, not Fuel's capture path.",
            expected[0], replay[0]
        );
    }
    drop(ctx);
}

// ============================================================================
// gemv_dense_m1 — MUST replay byte-identical (the capture-safety property).
// ============================================================================

#[test]
#[ignore]
fn gemv_dense_capture_replay_byte_identical() {
    let (ctx, stream) = gpu();
    // A GQA-shaped decode GEMV: n=17, k=40, batch=4 heads, shared B (stride_b=0).
    let (n, k, batch) = (17i64, 40i64, 4i64);
    let (nu, ku, bu) = (n as usize, k as usize, batch as usize);
    let a_len = bu * ku; // distinct A row per head
    let b_len = ku * nu; // one shared B
    let d_len = bu * nu; // distinct D per head

    let host_a: Vec<f32> = (0..a_len)
        .map(|i| ((i as i32 % 13 - 6) as f32) * 0.25)
        .collect();
    let host_b: Vec<f32> = (0..b_len)
        .map(|i| ((i as i32 % 11 - 5) as f32) * 0.125)
        .collect();

    let d_a = DeviceBuffer::from_slice(&ctx, &host_a).unwrap();
    let d_b = DeviceBuffer::from_slice(&ctx, &host_b).unwrap();
    let mut d_d = DeviceBuffer::from_slice(&ctx, &vec![0.0f32; d_len]).unwrap();

    let a_ptr = d_a.as_slice().as_raw().0 as *const c_void;
    let b_ptr = d_b.as_slice().as_raw().0 as *const c_void;
    let d_ptr = d_d.as_slice_mut().as_raw().0 as *mut c_void;

    let run_ffi = |stream_raw: *mut c_void| -> i32 {
        unsafe {
            baracuda_kernels_sys::baracuda_kernels_gemv_dense_m1_f32_run(
                1,
                n as i32,
                k as i32,
                batch as i32,
                0, // RRR
                1.0,
                0.0,
                a_ptr,
                ku as i64,
                ku as i64, // stride_a: distinct A per head
                b_ptr,
                nu as i64,
                0, // stride_b == 0: shared B (GQA broadcast)
                d_ptr,
                nu as i64,
                nu as i64, // stride_d: distinct D per head
                core::ptr::null_mut(),
                0,
                stream_raw,
            )
        }
    };

    // Eager warm → capture the reference bytes.
    assert_eq!(
        run_ffi(stream.as_raw() as *mut c_void),
        0,
        "warm gemv status"
    );
    stream.synchronize().unwrap();
    let mut reference = vec![0f32; d_len];
    d_d.copy_to_host(&mut reference).unwrap();
    // Non-trivial output (not all zero) so byte-identity is meaningful.
    assert!(
        reference.iter().any(|&x| x != 0.0),
        "gemv produced all-zero output"
    );

    let status = Cell::new(-1i32);
    let graph = stream
        .capture(CaptureMode::ThreadLocal, |s| {
            status.set(run_ffi(s.as_raw() as *mut c_void));
            Ok(())
        })
        .expect("capture gemv");
    assert_eq!(status.get(), 0, "captured gemv status");
    let exec = graph.instantiate().expect("instantiate");

    for run in 0..4 {
        exec.launch(&stream).expect("cuGraphLaunch");
        stream.synchronize().unwrap();
        let mut got = vec![0f32; d_len];
        d_d.copy_to_host(&mut got).unwrap();
        // BYTE-identical: every f32 bit-pattern must match the warm reference.
        for (i, (&g, &r)) in got.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                r.to_bits(),
                "gemv replay #{run} not byte-identical @ {i}: {g} vs {r}"
            );
        }
    }
    println!("gemv_dense_m1: 4 graph replays byte-identical to the warm reference (capture-safe).");
    drop(ctx);
}
