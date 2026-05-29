//! Smoke test for the host-side `gptq_to_marlin_repack` utility —
//! Phase 48 Goal C.
//!
//! This test runs **without** a GPU — `repack` is pure-Rust. It
//! validates:
//!
//! 1. The shape-validation arm rejects malformed input.
//! 2. The roundtrip on a minimal valid shape produces output of the
//!    expected size.
//! 3. The act_order=True rejection works (act_order=False is the
//!    Phase 48 supported scope; non-monotonic g_idx returns an
//!    error rather than producing silently-wrong output).
//!
//! No `#[ignore]` gate — runs in the standard `cargo test` matrix.

#![cfg(feature = "marlin")]

use baracuda_kernels::{gptq_to_marlin_repack, GptqWeights, MarlinWeights};

#[test]
fn rejects_bad_group_size() {
    let g = GptqWeights {
        qweight: &[0i32; 8],
        scales: &[1.0f32; 16],
        qzeros: &[0i32; 2],
        g_idx: None,
        ic: 128,
        oc: 16,
        group_size: 64, // Marlin supports only 128 (or -1 per-channel) in this scope
    };
    let err = gptq_to_marlin_repack(&g).unwrap_err();
    assert!(
        err.contains("group_size"),
        "expected group_size error, got: {err}"
    );
}

#[test]
fn rejects_shape_mismatch_qweight() {
    // group_size=128, IC=128, OC=16. Expected qweight = (128/8)*16 = 256 i32.
    // We pass 100 i32 to trigger the shape check.
    let g = GptqWeights {
        qweight: &[0i32; 100],
        scales: &[1.0f32; 16],
        qzeros: &[0i32; 2],
        g_idx: None,
        ic: 128,
        oc: 16,
        group_size: 128,
    };
    let err = gptq_to_marlin_repack(&g).unwrap_err();
    assert!(
        err.contains("qweight"),
        "expected qweight shape error, got: {err}"
    );
}

#[test]
fn rejects_act_order_true() {
    // g_idx with non-monotonic values triggers the act_order=True
    // path which is not yet implemented. Use IC=256 with
    // group_size=128 → 2 groups; assign half rows to group 1 and
    // half to group 0 (genuinely non-monotonic).
    let ic = 256usize;
    let oc = 16usize;
    let group_size = 128usize;
    let mut g_idx: Vec<i32> = vec![0i32; ic];
    for (i, v) in g_idx.iter_mut().enumerate() {
        // First half → group 1, second half → group 0.
        // This is non-monotonic (1, 1, ..., 1, 0, 0, ..., 0).
        *v = if i < ic / 2 { 1 } else { 0 };
    }
    let num_groups = ic / group_size;
    let qweight = vec![0i32; (ic / 8) * oc];
    let scales = vec![1.0f32; num_groups * oc];
    let qzeros = vec![0i32; num_groups * (oc / 8)];
    let g = GptqWeights {
        qweight: &qweight,
        scales: &scales,
        qzeros: &qzeros,
        g_idx: Some(&g_idx),
        ic,
        oc,
        group_size,
    };
    let err = gptq_to_marlin_repack(&g).unwrap_err();
    assert!(
        err.contains("act_order"),
        "expected act_order error, got: {err}"
    );
}

#[test]
fn minimal_repack_output_shape() {
    // IC=128, OC=256, group_size=128 (1 group).
    let ic = 128usize;
    let oc = 256usize;
    let group_size = 128usize;
    let num_groups = ic / group_size;

    // GPTQ weights: every nibble = 9 (so dequant via Marlin =
    // (9 - 8) * 1.0 = 1.0). Packed as int32 with 8 nibbles per
    // word → 0x99999999.
    let qweight = vec![0x99999999_u32 as i32; (ic / 8) * oc];
    let scales = vec![1.0f32; num_groups * oc];
    // Zero-points = 8 everywhere (i.e. symmetric — no shift); packed
    // as int32 with 8 nibbles per word along OC: 0x88888888.
    let qzeros = vec![0x88888888_u32 as i32; num_groups * (oc / 8)];

    let g = GptqWeights {
        qweight: &qweight,
        scales: &scales,
        qzeros: &qzeros,
        g_idx: None,
        ic,
        oc,
        group_size,
    };
    let MarlinWeights {
        weight_packed,
        scales: scales_out,
    } = gptq_to_marlin_repack(&g).expect("repack OK");

    // Expected packed length: K * N / 8 = 128 * 256 / 8 = 4096 i32.
    assert_eq!(weight_packed.len(), 4096, "weight_packed wrong length");
    assert_eq!(scales_out.len(), num_groups * oc, "scales wrong length");

    // With every input nibble = 9, every zp = 8, the folded value
    // should be `9 - 8 + 8 = 9`. Re-packed into int32 words of 8
    // nibbles each → 0x99999999. Every word in `weight_packed`
    // should equal this.
    let expected_word = 0x99999999_u32 as i32;
    for (i, &w) in weight_packed.iter().enumerate() {
        assert_eq!(
            w, expected_word,
            "weight_packed[{i}] = 0x{:08x}, expected 0x99999999",
            w as u32
        );
    }
}

#[test]
fn repack_handles_nonzero_zp() {
    // Sanity check that the zero-point fold actually shifts values.
    // GPTQ weights: every nibble = 5. zp = 3. Folded value should be
    // `5 - 3 + 8 = 10` everywhere → packed word 0xaaaaaaaa.
    let ic = 128usize;
    let oc = 256usize;
    let group_size = 128usize;
    let num_groups = ic / group_size;

    let qweight = vec![0x55555555_u32 as i32; (ic / 8) * oc];
    let scales = vec![1.0f32; num_groups * oc];
    // zp = 3 → packed nibbles 0x33333333.
    let qzeros = vec![0x33333333_u32 as i32; num_groups * (oc / 8)];

    let g = GptqWeights {
        qweight: &qweight,
        scales: &scales,
        qzeros: &qzeros,
        g_idx: None,
        ic,
        oc,
        group_size,
    };
    let MarlinWeights { weight_packed, .. } =
        gptq_to_marlin_repack(&g).expect("repack OK");
    let expected_word = 0xaaaaaaaa_u32 as i32;
    for (i, &w) in weight_packed.iter().enumerate() {
        assert_eq!(
            w, expected_word,
            "weight_packed[{i}] = 0x{:08x}, expected 0xaaaaaaaa (after zp fold)",
            w as u32
        );
    }
}

#[test]
fn repack_clamps_at_codebook_extremes() {
    // GPTQ qweight = 0, zp = 15 → fold = 0 - 15 + 8 = -7 → clamped to 0.
    let ic = 128usize;
    let oc = 256usize;
    let group_size = 128usize;
    let num_groups = ic / group_size;

    let qweight = vec![0i32; (ic / 8) * oc];
    let scales = vec![1.0f32; num_groups * oc];
    let qzeros = vec![0xffffffff_u32 as i32; num_groups * (oc / 8)]; // zp=15

    let g = GptqWeights {
        qweight: &qweight,
        scales: &scales,
        qzeros: &qzeros,
        g_idx: None,
        ic,
        oc,
        group_size,
    };
    let MarlinWeights { weight_packed, .. } =
        gptq_to_marlin_repack(&g).expect("repack OK");
    // Every nibble should be 0 (clamped from -7).
    let expected_word = 0i32;
    for (i, &w) in weight_packed.iter().enumerate() {
        assert_eq!(
            w, expected_word,
            "weight_packed[{i}] = 0x{:08x}, expected 0x00000000 (clamped from -7)",
            w as u32
        );
    }
}
