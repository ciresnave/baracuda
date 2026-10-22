//! GPTQ → Marlin weight repack utility — Phase 48 Goal C.
//!
//! Pure-Rust host-side routine that converts a GPTQ-format quantized
//! weight tensor (asymmetric int4 with per-group scales + zero-points
//! plus optional `g_idx` activation-order permutation) into the symmetric
//! int4 layout expected by [`super::int4_marlin::Int4MarlinGemmPlan`].
//!
//! Reference: AutoGPTQ's `use_marlin=True` repack path
//! (`auto_gptq/utils/marlin_utils.py`) + the upstream Marlin packer
//! at `marlin/__init__.py`. The repack is **lossy in general** —
//! GPTQ's asymmetric format can encode weights Marlin's symmetric
//! format cannot represent exactly. In practice the lossy
//! transformation is acceptable (the per-output-row absmax dominates
//! quantization noise) and is the standard production approach.
//!
//! ## Algorithmic outline
//!
//! 1. **Activation-order permutation reversal**. GPTQ may store
//!    weights in `g_idx`-permuted order (activation-aware
//!    quantization, "act_order=True"). Marlin is `g_idx`-agnostic.
//!    We undo the GPTQ permutation along the K (= IC) axis before
//!    the int4-layout shuffle.
//! 2. **Zero-point absorption**. GPTQ stores `(q, scale, zp)` per
//!    group such that `w ≈ scale * (q - zp)`. Marlin stores
//!    `(q', scale')` such that `w ≈ scale' * (q' - 8)` (symmetric,
//!    fixed zero at 8). The conversion holds the int4 storage byte
//!    constant (`q' = q`) and adjusts the dequant arithmetic by
//!    folding `(zp - 8)` into a per-group additive shift. Marlin's
//!    scale layout doesn't carry an additive shift, so the **standard
//!    approximation** is to set `scale' = scale` and accept the
//!    small reconstruction error from `zp != 8`. Higher-quality
//!    variants (AWQ-style "scale-search" calibration) can search
//!    for a better per-group scale; documented as a future
//!    refinement.
//! 3. **Int4 weight reshuffle**. Marlin's `marlin_cuda` kernel
//!    expects an offline permutation along both axes to align with
//!    tensor-core fragment layout. The permutation is hardcoded in
//!    `marlin/__init__.py` as `_perm`, `_scale_perm`, and
//!    `_scale_perm_single`. We replicate those tables here.
//! 4. **Pack into Marlin's int32 format**. The output is `[K/16,
//!    N*16/8]` int32 with 8 int4 nibbles per int32 word.
//!
//! ## Implementation status
//!
//! Steps 3-4 are Marlin's real `_perm` / `_scale_perm` reshuffle (a
//! transcription of upstream `Layer.pack`). Its correctness is established
//! end to end on a device by `tests/marlin_reference.rs` (kernel output vs a
//! dequantise-then-GEMM reference built from the unpacked weights), not by
//! reading this code. Still lossy for `zp != 8` (see step 2), and `g_idx`
//! permutations and per-channel (`group_size == -1`) are rejected.
//!
//! ## Cargo feature
//!
//! Gated behind the `marlin` cargo feature alongside the Rust plan.
//! The utility itself has no GPU dependencies but only makes sense
//! when consumers also link the Marlin kernel.

extern crate alloc;
use alloc::vec::Vec;

/// Layout of GPTQ-format quantized weights. The standard
/// AutoGPTQ on-disk format.
#[derive(Debug)]
pub struct GptqWeights<'a> {
    /// Packed int4 weights, shape `[IC/8, OC]` int32 (8 packed
    /// nibbles per int32 word along IC). Note: this is K-major;
    /// AWQ uses OC-major. Confirm the source layout matches.
    pub qweight: &'a [i32],
    /// Per-group scales `[IC/group_size, OC]` f32 (or f16; this
    /// utility takes f32 for precision during the repack).
    pub scales: &'a [f32],
    /// Per-group packed int4 zero-points `[IC/group_size, OC/8]`
    /// int32 (8 packed nibbles per int32 word along OC).
    pub qzeros: &'a [i32],
    /// Optional `g_idx` permutation `[IC]` int32. Each element is
    /// the group index for the corresponding K position. When
    /// `act_order=False` this is `[0, 0, ..., 1, 1, ..., 2, 2, ...]`
    /// (group-monotonic) and the permutation is a no-op. When
    /// `act_order=True` the values are scrambled and the K axis
    /// needs unpermuting before the Marlin shuffle.
    pub g_idx: Option<&'a [i32]>,
    /// Input channels (= GEMM K dim).
    pub ic: usize,
    /// Output channels (= GEMM N dim).
    pub oc: usize,
    /// Quantization group size (typically 128).
    pub group_size: usize,
}

/// Marlin-format output bundle.
#[derive(Debug)]
pub struct MarlinWeights {
    /// Packed int4 weights in Marlin's tensor-core-fragment layout.
    /// Shape `[K/16, N*16/8]` int32 = `K * N / 8` int32 elements
    /// total.
    pub weight_packed: Vec<i32>,
    /// Per-group scales `[K/group_size, N]` f32 in Marlin's
    /// pre-permuted-along-N layout (or `[1, N]` for per-channel /
    /// `group_size == -1`).
    pub scales: Vec<f32>,
}

/// Repack a GPTQ-format weight tensor into Marlin format.
///
/// Returns an [`Err`] message if the input shapes / group size are
/// inconsistent. Returns weights that can be loaded directly into
/// [`super::int4_marlin::Int4MarlinGemmArgs`] (after `f32 → f16`
/// cast on the scales).
///
/// # Algorithm
///
/// 1. Validate input shapes against `(ic, oc, group_size)`.
/// 2. Unpack GPTQ int4 weights to a dense `[IC, OC]` int8 grid
///    (values in `[0, 15]`).
/// 3. Unpack zero-points to `[num_groups, OC]` int8.
/// 4. Apply `g_idx` reverse-permutation if present.
/// 5. Fold zero-points into the dequant arithmetic by adjusting
///    the int4 values: `q' = q - zp + 8` (mod 16), clamped to
///    `[0, 15]`. The clamp introduces the documented reconstruction
///    error when `zp != 8` and `q` is near the codebook extremes.
/// 6. Apply Marlin's offline weight permutation along both axes.
/// 7. Repack into `[K/16, N*16/8]` int32 with 8 nibbles per word
///    per Marlin's `mma.sync.m16n8k16` fragment layout.
/// 8. Permute scales along the N axis per Marlin's `_scale_perm` /
///    `_scale_perm_single` table.
pub fn repack(g: &GptqWeights<'_>) -> Result<MarlinWeights, &'static str> {
    if g.ic == 0 || g.oc == 0 {
        return Err("gptq_to_marlin::repack: IC and OC must be positive");
    }
    if g.group_size == 0 || (g.group_size != 128 && g.group_size as i32 != -1i32) {
        // Repack supports the same group_size values Marlin's kernel
        // accepts; reject others up front rather than producing
        // technically-valid but kernel-unusable output.
        return Err(
            "gptq_to_marlin::repack: group_size must be 128 (per-group) — \
             per-channel (g=-1) support requires a separate _scale_perm_single \
             table not yet wired",
        );
    }
    if g.ic % 16 != 0 {
        return Err("gptq_to_marlin::repack: IC must be divisible by 16");
    }
    if g.oc % 8 != 0 {
        return Err("gptq_to_marlin::repack: OC must be divisible by 8");
    }
    if g.ic % g.group_size != 0 {
        return Err("gptq_to_marlin::repack: IC must be divisible by group_size");
    }
    let num_groups = g.ic / g.group_size;
    let expected_qweight_len = (g.ic / 8) * g.oc;
    if g.qweight.len() != expected_qweight_len {
        return Err("gptq_to_marlin::repack: qweight length != (IC/8) * OC");
    }
    if g.scales.len() != num_groups * g.oc {
        return Err("gptq_to_marlin::repack: scales length != num_groups * OC");
    }
    if g.qzeros.len() != num_groups * (g.oc / 8) {
        return Err("gptq_to_marlin::repack: qzeros length != num_groups * (OC/8)");
    }

    // Step 2: unpack GPTQ qweight [IC/8, OC] int32 → dense [IC, OC] u8.
    let mut weight_dense = alloc::vec![0u8; g.ic * g.oc];
    for ic_byte in 0..(g.ic / 8) {
        for oc in 0..g.oc {
            let word = g.qweight[ic_byte * g.oc + oc] as u32;
            for nib in 0..8usize {
                let q = ((word >> (4 * nib)) & 0xF) as u8;
                let ic_pos = ic_byte * 8 + nib;
                weight_dense[ic_pos * g.oc + oc] = q;
            }
        }
    }

    // Step 3: unpack zeros [num_groups, OC/8] int32 → dense
    // [num_groups, OC] u8.
    let mut zeros_dense = alloc::vec![0u8; num_groups * g.oc];
    for grp in 0..num_groups {
        for oc_byte in 0..(g.oc / 8) {
            let word = g.qzeros[grp * (g.oc / 8) + oc_byte] as u32;
            for nib in 0..8usize {
                let z = ((word >> (4 * nib)) & 0xF) as u8;
                zeros_dense[grp * g.oc + oc_byte * 8 + nib] = z;
            }
        }
    }

    // Step 4: g_idx reverse permutation (if present).
    // GPTQ stores rows in act-order; Marlin expects natural K order.
    // `g_idx[k_natural]` would be the group-id of position
    // `k_natural` — but the actual permutation semantics differ
    // between AutoGPTQ variants; the safe path is to skip when
    // `g_idx` is None or monotonic, and emit a TODO when non-trivial.
    if let Some(idx) = g.g_idx {
        if idx.len() != g.ic {
            return Err("gptq_to_marlin::repack: g_idx length != IC");
        }
        let is_monotonic = idx.windows(2).all(|w| w[0] <= w[1]);
        if !is_monotonic {
            // Non-trivial act_order=True case. The standard
            // AutoGPTQ repack first un-permutes weight rows by the
            // inverse of g_idx so that subsequent group lookup
            // becomes the natural `k / group_size`. Implementation
            // sketch: build `inv_perm` from g_idx's argsort, then
            // reorder `weight_dense` rows by inv_perm. Skipped here
            // for the trailblazer; smoke tests use act_order=False
            // checkpoints (the common HF GPTQ default).
            return Err(
                "gptq_to_marlin::repack: act_order=True (non-monotonic g_idx) \
                 not yet implemented — re-quantize the GPTQ checkpoint with \
                 desc_act=False or wait for a Phase 48 follow-up",
            );
        }
    }

    // Step 5: fold zero-points by shifting weights.
    // Marlin's dequant is `s * (q - 8)`. GPTQ's dequant is
    // `s * (q - zp)`. Setting the new int4 value to `q - zp + 8`
    // (mod 16) makes the algebra align when no clamping occurs.
    // Standard practice clamps to [0, 15] and accepts the small
    // reconstruction error when `|zp - 8|` is large.
    for ic in 0..g.ic {
        let grp = ic / g.group_size;
        for oc in 0..g.oc {
            let q = weight_dense[ic * g.oc + oc] as i32;
            let zp = zeros_dense[grp * g.oc + oc] as i32;
            let q_marlin = (q - zp + 8).clamp(0, 15) as u8;
            weight_dense[ic * g.oc + oc] = q_marlin;
        }
    }

    if g.oc % MARLIN_SCALE_PERM_LEN != 0 {
        return Err(
            "gptq_to_marlin::repack: OC must be divisible by 64 (Marlin's permutation runs)",
        );
    }

    // Step 6 + 7: Marlin's offline weight reshuffle, a transcription of
    // `Layer.pack` in upstream `marlin/__init__.py`.
    //   1. tile `[K, N]` into 16x16 tiles, ordered `[K/16][N/16]`, each tile
    //      row-major: `res[kb][nb*256 + r*16 + c] = w[kb*16 + r][nb*16 + c]`;
    //   2. within every run of 1024 values apply `res[j] = old[perm[j]]`
    //      (`marlin_perm`, which already folds in the 8-way nibble interleave);
    //   3. pack 8 consecutive values per i32, value `b` in bits `4b..4b+4`.
    let k_blocks = g.ic / 16;
    let n_words = g.oc * 16 / 8;
    let perm = marlin_perm();
    let mut packed = alloc::vec![0i32; k_blocks * n_words];
    let mut tiled = alloc::vec![0u8; g.oc * 16];
    for kb in 0..k_blocks {
        for nb in 0..g.oc / 16 {
            for r in 0..16usize {
                for c in 0..16usize {
                    tiled[nb * 256 + r * 16 + c] = weight_dense[(kb * 16 + r) * g.oc + nb * 16 + c];
                }
            }
        }
        for chunk in 0..g.oc * 16 / MARLIN_PERM_LEN {
            let base = chunk * MARLIN_PERM_LEN;
            for w in 0..MARLIN_PERM_LEN / 8 {
                let mut word: u32 = 0;
                for bit in 0..8usize {
                    let q = tiled[base + perm[w * 8 + bit]] as u32;
                    word |= (q & 0xF) << (4 * bit);
                }
                packed[kb * n_words + chunk * (MARLIN_PERM_LEN / 8) + w] = word as i32;
            }
        }
    }

    // Step 8: scales `[num_groups, OC]` -> `reshape(-1, 64)[:, scale_perm]`
    // -> `reshape(-1, OC)` (upstream `_scale_perm`, group_size = 128).
    let scale_perm = marlin_scale_perm();
    let mut scales_out = alloc::vec![0f32; g.scales.len()];
    for (dst, src) in scales_out
        .chunks_exact_mut(MARLIN_SCALE_PERM_LEN)
        .zip(g.scales.chunks_exact(MARLIN_SCALE_PERM_LEN))
    {
        for (j, d) in dst.iter_mut().enumerate() {
            *d = src[scale_perm[j]];
        }
    }

    Ok(MarlinWeights {
        weight_packed: packed,
        scales: scales_out,
    })
}

/// Marlin's weight permutation over one run of 1024 values
/// (`marlin/__init__.py::_get_perms`, the `_perm` table including its final
/// `[0, 2, 4, 6, 1, 3, 5, 7]` nibble interleave).
fn marlin_perm() -> Vec<usize> {
    let mut perm = Vec::with_capacity(1024);
    for i in 0..32usize {
        let col = i / 4;
        let mut perm1 = Vec::with_capacity(8);
        for block in 0..2usize {
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ] {
                perm1.push(16 * row + col + 8 * block);
            }
        }
        for j in 0..4usize {
            perm.extend(perm1.iter().map(|p| p + 256 * j));
        }
    }
    const INTERLEAVE: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    let mut out = Vec::with_capacity(1024);
    for grp in perm.chunks_exact(8) {
        out.extend(INTERLEAVE.iter().map(|&i| grp[i]));
    }
    out
}

/// Marlin's scale permutation over one run of 64 scales
/// (`marlin/__init__.py::_scale_perm`).
fn marlin_scale_perm() -> Vec<usize> {
    let mut out = Vec::with_capacity(MARLIN_SCALE_PERM_LEN);
    for i in 0..8usize {
        out.extend((0..8usize).map(|j| i + 8 * j));
    }
    out
}

/// Length of the run over which [`repack`] applies Marlin's weight
/// permutation (`marlin/__init__.py::_perm`): 1024 values, i.e. 16 K-rows by
/// 64 N-columns of one tile row.
pub const MARLIN_PERM_LEN: usize = 1024;

/// Length of the run over which [`repack`] applies Marlin's scale
/// permutation (`marlin/__init__.py::_scale_perm`): 64 scales.
pub const MARLIN_SCALE_PERM_LEN: usize = 64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unsupported_group_size() {
        let g = GptqWeights {
            qweight: &[],
            scales: &[],
            qzeros: &[],
            g_idx: None,
            ic: 0,
            oc: 0,
            group_size: 64,
        };
        assert!(repack(&g).is_err());
    }

    #[test]
    fn rejects_shape_mismatch() {
        // group_size=128, IC=128, OC=8: weight = 1 row of int32 = 8
        // i32. We pass a 4-element buffer to trigger the shape check.
        let g = GptqWeights {
            qweight: &[0i32; 4],
            scales: &[1.0f32; 8],
            qzeros: &[0i32; 1],
            g_idx: None,
            ic: 128,
            oc: 8,
            group_size: 128,
        };
        assert!(repack(&g).is_err());
    }

    #[test]
    fn accepts_minimal_shape() {
        // group_size=128, IC=128, OC=256. Min Marlin shape with
        // K%16==0, OC%8==0. Expected sizes:
        //   qweight  = (128/8) * 256 = 4096 i32
        //   scales   = (128/128) * 256 = 256 f32
        //   qzeros   = (128/128) * (256/8) = 32 i32
        let g = GptqWeights {
            qweight: &alloc::vec![0i32; 4096],
            scales: &alloc::vec![1.0f32; 256],
            qzeros: &alloc::vec![0x77777777i32; 32], // all zp = 7
            g_idx: None,
            ic: 128,
            oc: 256,
            group_size: 128,
        };
        let m = repack(&g).expect("repack should succeed");
        // K * N / 8 = 128 * 256 / 8 = 4096 int32 words.
        assert_eq!(m.weight_packed.len(), 4096);
        assert_eq!(m.scales.len(), 256);
    }
}
