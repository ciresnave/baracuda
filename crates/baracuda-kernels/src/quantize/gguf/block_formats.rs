//! Rust `#[repr(C, packed)]` mirrors of the GGUF block-format structs.
//!
//! Layouts match the C definitions in
//! `crates/baracuda-kernels-sys/kernels/include/baracuda_gguf.cuh`
//! exactly (which in turn mirror llama.cpp's `ggml.h`). These are
//! provided for caller-side packing / unpacking, struct-size sanity
//! checks, and the in-Rust dequant reference paths used by the tests.
//!
//! `f16` scale fields are stored as raw `u16` bit-patterns (the GGUF
//! file format never carries a native `__half` — it's always the IEEE
//! 754 binary16 bit pattern). Callers that need the FP value should
//! convert via `half::f16::from_bits(...)`.

use baracuda_kernels_types::GgufBlockFormat;

/// `block_q4_0` — 4-bit, 32 elements per block, single fp16 scale.
/// Layout: 2 bytes (scale `d`) + 16 bytes (nibble-packed quants `qs`).
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct BlockQ4_0 {
    /// fp16 scale (`d`), stored as the raw u16 bit-pattern.
    pub d: u16,
    /// Nibble-packed quants: high nibble holds value [16..32), low [0..16).
    pub qs: [u8; 16],
}

const _: () = assert!(core::mem::size_of::<BlockQ4_0>() == 18);

/// `block_q4_1` — 4-bit, 32 elements per block, fp16 scale + fp16 min.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct BlockQ4_1 {
    /// `dm.x = d` (scale), `dm.y = m` (min); raw fp16 bit-patterns.
    pub dm: [u16; 2],
    /// Nibble-packed quants.
    pub qs: [u8; 16],
}

const _: () = assert!(core::mem::size_of::<BlockQ4_1>() == 20);

/// `block_q5_0` — 5-bit, 32 elements per block, single fp16 scale.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct BlockQ5_0 {
    /// fp16 scale.
    pub d: u16,
    /// 5-th bit per quant (32 bits → 4 bytes).
    pub qh: [u8; 4],
    /// Low 4 bits per quant (nibble-packed).
    pub qs: [u8; 16],
}

const _: () = assert!(core::mem::size_of::<BlockQ5_0>() == 22);

/// `block_q5_1` — 5-bit, 32 elements per block, fp16 scale + fp16 min.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct BlockQ5_1 {
    /// `dm.x = d`, `dm.y = m`.
    pub dm: [u16; 2],
    /// 5-th bit per quant.
    pub qh: [u8; 4],
    /// Low 4 bits per quant.
    pub qs: [u8; 16],
}

const _: () = assert!(core::mem::size_of::<BlockQ5_1>() == 24);

/// `block_q8_0` — 8-bit, 32 elements per block, single fp16 scale.
/// One-byte quants; no min, no nibble packing.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct BlockQ8_0 {
    /// fp16 scale.
    pub d: u16,
    /// Signed 8-bit quants.
    pub qs: [i8; 32],
}

const _: () = assert!(core::mem::size_of::<BlockQ8_0>() == 34);

// ---- k-quants (256-element super-blocks) ----

/// `block_q2_K` — 2-bit (effective), 256-element super-block.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct BlockQ2K {
    /// 16 quantized scales + mins, 4 bits each.
    pub scales: [u8; 16],
    /// 64 bytes packing 256 2-bit quants.
    pub qs: [u8; 64],
    /// Super-block fp16 scale (`dm.x`) + fp16 min (`dm.y`).
    pub dm: [u16; 2],
}

const _: () = assert!(core::mem::size_of::<BlockQ2K>() == 84);

/// `block_q3_K` — 3-bit (effective), 256-element super-block.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct BlockQ3K {
    /// High-bit mask: 32 bytes packing 256 high bits.
    pub hmask: [u8; 32],
    /// Low 2 bits per quant: 64 bytes.
    pub qs: [u8; 64],
    /// 12 bytes packing 16 × 6-bit scales.
    pub scales: [u8; 12],
    /// Super-block fp16 scale.
    pub d: u16,
}

const _: () = assert!(core::mem::size_of::<BlockQ3K>() == 110);

/// `block_q4_K` — 4-bit (effective + scale overhead), 256-element block.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct BlockQ4K {
    /// Super-block fp16 scale + fp16 min.
    pub dm: [u16; 2],
    /// 12 bytes packing 8 × 6-bit scales and 8 × 6-bit mins.
    pub scales: [u8; 12],
    /// 128 bytes packing 256 4-bit quants.
    pub qs: [u8; 128],
}

const _: () = assert!(core::mem::size_of::<BlockQ4K>() == 144);

/// `block_q5_K` — 5-bit, 256-element super-block.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct BlockQ5K {
    /// Super-block fp16 scale + fp16 min.
    pub dm: [u16; 2],
    /// 12 bytes packing 8 × 6-bit scales and 8 × 6-bit mins.
    pub scales: [u8; 12],
    /// 5-th bit per quant.
    pub qh: [u8; 32],
    /// Low 4 bits per quant.
    pub qs: [u8; 128],
}

const _: () = assert!(core::mem::size_of::<BlockQ5K>() == 176);

/// `block_q6_K` — 6-bit, 256-element super-block.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct BlockQ6K {
    /// Low 4 bits per quant.
    pub ql: [u8; 128],
    /// Upper 2 bits per quant.
    pub qh: [u8; 64],
    /// 16 × signed 8-bit sub-block scales.
    pub scales: [i8; 16],
    /// Super-block fp16 scale.
    pub d: u16,
}

const _: () = assert!(core::mem::size_of::<BlockQ6K>() == 210);

/// `block_q8_K` — 8-bit, 256-element super-block (CPU-side intermediate
/// in llama.cpp; baracuda exposes dequant only, no MMVQ).
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct BlockQ8K {
    /// Super-block f32 scale (NOT fp16 — Q8_K uses f32 to match
    /// llama.cpp's `block_q8_K`).
    pub d: f32,
    /// 256 signed 8-bit quants.
    pub qs: [i8; 256],
    /// Sum of quants in groups of 16 (CPU-side acceleration scratch).
    pub bsums: [i16; 16],
}

const _: () = assert!(core::mem::size_of::<BlockQ8K>() == 292);

// =============================================================================
// Static size cross-check: each block's Rust `size_of` must match the
// corresponding `GgufBlockFormat::type_size()` discriminant.
// =============================================================================

const _: () = {
    assert!(core::mem::size_of::<BlockQ4_0>() == GgufBlockFormat::Q4_0.type_size());
    assert!(core::mem::size_of::<BlockQ4_1>() == GgufBlockFormat::Q4_1.type_size());
    assert!(core::mem::size_of::<BlockQ5_0>() == GgufBlockFormat::Q5_0.type_size());
    assert!(core::mem::size_of::<BlockQ5_1>() == GgufBlockFormat::Q5_1.type_size());
    assert!(core::mem::size_of::<BlockQ8_0>() == GgufBlockFormat::Q8_0.type_size());
    assert!(core::mem::size_of::<BlockQ2K>() == GgufBlockFormat::Q2K.type_size());
    assert!(core::mem::size_of::<BlockQ3K>() == GgufBlockFormat::Q3K.type_size());
    assert!(core::mem::size_of::<BlockQ4K>() == GgufBlockFormat::Q4K.type_size());
    assert!(core::mem::size_of::<BlockQ5K>() == GgufBlockFormat::Q5K.type_size());
    assert!(core::mem::size_of::<BlockQ6K>() == GgufBlockFormat::Q6K.type_size());
    assert!(core::mem::size_of::<BlockQ8K>() == GgufBlockFormat::Q8K.type_size());
};
