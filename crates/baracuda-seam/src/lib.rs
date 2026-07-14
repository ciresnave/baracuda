//! Profile v1 kernel-seam handshake — the `SeamHello` negotiation envelope.
//!
//! Per the *Kernel-Seam Interop Contract* §3.1, `SeamHello` is the
//! **frozen-forever** wire format by which each side advertises its supported
//! seam profiles + capabilities; the connection selects the highest
//! mutually-supported profile (§3.2) or hard-fails. Baracuda exposes
//! [`baracuda_seam_hello`] as the provider-side C-ABI entry point Fuel reads,
//! then runs the negotiation.
//!
//! The envelope is a fixed-size POD (no variable-length member) so the C ABI is
//! an out-param fill, never a by-value return — and its 56-byte layout is frozen
//! and cross-checked at compile time the way FDX checks its `#[repr(C)]` structs.

/// The envelope magic (§3.1). Equals `0x4D41_4553` so the on-wire bytes
/// (little-endian, offset 0) are `53 45 41 4D` = ASCII `"SEAM"`, per
/// KISS-ANNOUNCE §6.1-0004. The old `0x5345_414D` spelled `"SEAM"` only when read
/// big-endian and put `"MAES"` on a little-endian wire; flipped in lockstep with
/// Fuel's `fuel-kernel-seam-announce` seed (commit 1849bc9a).
pub const SEAM_MAGIC: u32 = 0x4D41_4553;
/// The envelope's own version (§3.1). Designed never to bump; only a change to
/// the envelope *shape* (e.g. raising [`SEAM_MAX_PROFILES`]) would force it.
pub const SEAM_ENVELOPE_VERSION: u8 = 1;
/// Fixed cap on simultaneously-advertised profiles (§3.1). Generous in practice
/// since profiles retire as the floor advances (§3.6).
pub const SEAM_MAX_PROFILES: usize = 16;

/// The seam profile Baracuda implements (Profile v1).
pub const BARACUDA_PROFILE: u16 = 1;

/// The kernel-seam negotiation envelope (Profile v1 §3.1).
///
/// A fixed-size POD with frozen field offsets; everything mutable lives *inside*
/// a negotiated profile, not the envelope. Layout matches the normative C struct
/// (56 bytes, 8-byte aligned).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SeamHello {
    /// `== SEAM_MAGIC`.
    pub magic: u32,
    /// `== SEAM_ENVELOPE_VERSION`.
    pub envelope_version: u8,
    /// Zero padding.
    pub reserved: [u8; 3],
    /// Number of valid entries in `profiles` (`<= SEAM_MAX_PROFILES`).
    pub profiles_len: u16,
    /// Supported profile integers, ascending; entries `[profiles_len..]` are 0.
    pub profiles: [u16; SEAM_MAX_PROFILES],
    /// Explicit alignment padding (offsets 42..48) between `profiles` and the
    /// 8-byte-aligned `capabilities` — MUST be zeroed on write; a KISS-conform
    /// reader hard-rejects a nonzero value (KISS-ANNOUNCE §6.2-0011). Made an
    /// explicit field (not implicit `#[repr(C)]` padding) to mirror Fuel's
    /// `SeamHello`, so the two seeds stay bit-for-bit identical.
    pub reserved1: [u8; 6],
    /// Optional-feature bitset within the selected profile (§3.4).
    pub capabilities: u64,
}

// The envelope is frozen at 56 bytes (§3.1); cross-check it the way FDX does.
const _: () = assert!(core::mem::size_of::<SeamHello>() == 56);
const _: () = assert!(core::mem::align_of::<SeamHello>() == 8);

/// Why a received `SeamHello` is rejected. A conforming reader hard-rejects
/// (never repairs, never panics) — KISS-ANNOUNCE §6.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeamError {
    /// The input buffer is shorter than the 56-byte envelope.
    Truncated,
    /// `magic` is not [`SEAM_MAGIC`] (on-wire bytes are not `53 45 41 4D`).
    BadMagic,
    /// `envelope_version` is not one this seed understands.
    UnsupportedEnvelopeVersion(u8),
    /// A reserved field (`reserved` or `reserved1`) is nonzero (§6.2-0011).
    ReservedNonZero,
    /// `profiles_len` exceeds [`SEAM_MAX_PROFILES`].
    ProfilesLenOutOfRange(u16),
}

impl SeamHello {
    /// Hard-reject a received envelope that violates a KISS-ANNOUNCE invariant:
    /// wrong magic, an unknown envelope version, a nonzero reserved field
    /// (§6.2-0011), or an out-of-range `profiles_len`. Never panics.
    pub fn validate(&self) -> Result<(), SeamError> {
        if self.magic != SEAM_MAGIC {
            return Err(SeamError::BadMagic);
        }
        if self.envelope_version != SEAM_ENVELOPE_VERSION {
            return Err(SeamError::UnsupportedEnvelopeVersion(self.envelope_version));
        }
        if self.reserved != [0u8; 3] || self.reserved1 != [0u8; 6] {
            return Err(SeamError::ReservedNonZero);
        }
        if self.profiles_len as usize > SEAM_MAX_PROFILES {
            return Err(SeamError::ProfilesLenOutOfRange(self.profiles_len));
        }
        Ok(())
    }

    /// Select the highest profile advertised by BOTH envelopes (§3.2). `None` on
    /// an empty intersection — a hard fail, never a panic. Only the first
    /// `profiles_len` entries (clamped to [`SEAM_MAX_PROFILES`]) are considered.
    #[must_use]
    pub fn negotiate(&self, other: &SeamHello) -> Option<u16> {
        let n = (self.profiles_len as usize).min(SEAM_MAX_PROFILES);
        let m = (other.profiles_len as usize).min(SEAM_MAX_PROFILES);
        let theirs = &other.profiles[..m];
        self.profiles[..n]
            .iter()
            .filter(|p| theirs.contains(p))
            .copied()
            .max()
    }

    /// Serialize to the 56-byte little-endian wire envelope.
    #[must_use]
    pub fn to_wire_bytes(&self) -> [u8; 56] {
        let mut b = [0u8; 56];
        b[0..4].copy_from_slice(&self.magic.to_le_bytes());
        b[4] = self.envelope_version;
        b[5..8].copy_from_slice(&self.reserved);
        b[8..10].copy_from_slice(&self.profiles_len.to_le_bytes());
        for (i, p) in self.profiles.iter().enumerate() {
            let off = 10 + i * 2;
            b[off..off + 2].copy_from_slice(&p.to_le_bytes());
        }
        b[42..48].copy_from_slice(&self.reserved1);
        b[48..56].copy_from_slice(&self.capabilities.to_le_bytes());
        b
    }

    /// Parse + validate a 56-byte little-endian wire envelope from an untrusted
    /// peer, byte-by-byte (no struct-cast) so it is a genuinely independent second
    /// parser. Hard-rejects a short buffer or any [`SeamHello::validate`]
    /// violation; never panics.
    pub fn parse_hello(bytes: &[u8]) -> Result<SeamHello, SeamError> {
        if bytes.len() < 56 {
            return Err(SeamError::Truncated);
        }
        let rd_u16 = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
        let mut profiles = [0u16; SEAM_MAX_PROFILES];
        for (i, slot) in profiles.iter_mut().enumerate() {
            *slot = rd_u16(10 + i * 2);
        }
        let mut reserved1 = [0u8; 6];
        reserved1.copy_from_slice(&bytes[42..48]);
        let hello = SeamHello {
            magic: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            envelope_version: bytes[4],
            reserved: [bytes[5], bytes[6], bytes[7]],
            profiles_len: rd_u16(8),
            profiles,
            reserved1,
            capabilities: u64::from_le_bytes([
                bytes[48], bytes[49], bytes[50], bytes[51], bytes[52], bytes[53], bytes[54],
                bytes[55],
            ]),
        };
        hello.validate()?;
        Ok(hello)
    }
}

// ---------------------------------------------------------------------------
// Capability bits (§3.4). The FDX `BackendProbe` tokens occupy the low bits;
// FKC-/JIT-level optional features take higher bits.
//
// NOTE: the FDX-token bit positions are normatively FDX §12's; pinned here to
// that order and to be reconciled against the FDX annex when wired.
// ---------------------------------------------------------------------------

/// FDX sub-byte / extension v1 (`DlpackExtV1`).
pub const SEAM_CAP_DLPACK_EXT_V1: u64 = 1 << 0;
/// FDX MX microscaling (`DlpackExtMx`).
pub const SEAM_CAP_DLPACK_EXT_MX: u64 = 1 << 1;
/// FDX GGML block quant (`DlpackExtGgml`).
pub const SEAM_CAP_DLPACK_EXT_GGML: u64 = 1 << 2;
/// FDX affine quant — incl. AFFINE_BLOCK / NF4-QLoRA (`DlpackExtAffine`).
pub const SEAM_CAP_DLPACK_EXT_AFFINE: u64 = 1 << 3;
/// FDX symbolic extents (`DlpackExtSymbolic`).
pub const SEAM_CAP_DLPACK_EXT_SYMBOLIC: u64 = 1 << 4;
/// FDX gather / paged-blocks residency (`DlpackExtGather`).
pub const SEAM_CAP_DLPACK_EXT_GATHER: u64 = 1 << 5;
/// JIT-on-request endpoint implemented (§5) — Baracuda's `Synthesizer` impl
/// (`baracuda_kernelgen::jit::seam::BaracudaSynthesizer`) builds a kernel for a
/// Fuel-chosen region against the published `fuel-kernel-seam` envelope.
pub const SEAM_CAP_JIT_ON_REQUEST: u64 = 1 << 32;

/// The capabilities Baracuda advertises at first connect: the FDX tokens for its
/// shipped kernel families **plus** `SeamCapJitOnRequest` — both halves of the §5
/// seam are now live (Fuel published the `fuel-kernel-seam` envelope; Baracuda
/// implements the `Synthesizer` it calls), so Fuel may issue JIT-on-request calls.
pub const BARACUDA_CAPABILITIES: u64 = SEAM_CAP_DLPACK_EXT_V1
    | SEAM_CAP_DLPACK_EXT_MX
    | SEAM_CAP_DLPACK_EXT_GGML
    | SEAM_CAP_DLPACK_EXT_AFFINE
    | SEAM_CAP_DLPACK_EXT_SYMBOLIC
    | SEAM_CAP_DLPACK_EXT_GATHER
    | SEAM_CAP_JIT_ON_REQUEST;

/// Build Baracuda's advertised `SeamHello`.
#[must_use]
pub fn baracuda_hello() -> SeamHello {
    let mut profiles = [0u16; SEAM_MAX_PROFILES];
    profiles[0] = BARACUDA_PROFILE;
    SeamHello {
        magic: SEAM_MAGIC,
        envelope_version: SEAM_ENVELOPE_VERSION,
        reserved: [0; 3],
        profiles_len: 1,
        profiles,
        reserved1: [0; 6],
        capabilities: BARACUDA_CAPABILITIES,
    }
}

/// Profile-v1 provider-side handshake entry point (§3.5). Fuel calls this, then
/// runs the §3.2 negotiation. Fills `*out`; returns `0` on success, `2` if `out`
/// is null (FDX house convention: 2 = invalid argument). Never panics, never
/// allocates.
///
/// # Safety
/// `out` must be a valid, writable pointer to a `SeamHello`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn baracuda_seam_hello(out: *mut SeamHello) -> i32 {
    if out.is_null() {
        return 2;
    }
    // SAFETY: non-null checked above; caller guarantees writability + alignment.
    unsafe { out.write(baracuda_hello()) };
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_advertises_profile_v1() {
        let h = baracuda_hello();
        assert_eq!(h.magic, SEAM_MAGIC);
        assert_eq!(h.envelope_version, 1);
        assert_eq!(h.profiles_len, 1);
        assert_eq!(h.profiles[0], 1);
        assert_eq!(h.profiles[1], 0); // unused entries zeroed
        // §5 now live: JIT-on-request advertised, alongside the FDX tokens.
        assert_ne!(h.capabilities & SEAM_CAP_JIT_ON_REQUEST, 0);
        assert_ne!(h.capabilities & SEAM_CAP_DLPACK_EXT_GGML, 0);
    }

    #[test]
    fn c_abi_fills_out_param() {
        let mut hello = SeamHello {
            magic: 0,
            envelope_version: 0,
            reserved: [0; 3],
            profiles_len: 0,
            profiles: [0; SEAM_MAX_PROFILES],
            reserved1: [0; 6],
            capabilities: 0,
        };
        let rc = unsafe { baracuda_seam_hello(&mut hello) };
        assert_eq!(rc, 0);
        assert_eq!(hello.magic, SEAM_MAGIC);
        assert_eq!(hello.profiles[0], 1);
        // null → typed error, no panic.
        assert_eq!(unsafe { baracuda_seam_hello(core::ptr::null_mut()) }, 2);
    }

    #[test]
    fn envelope_is_56_bytes() {
        assert_eq!(core::mem::size_of::<SeamHello>(), 56);
    }

    #[test]
    fn reserved_fields_are_zeroed() {
        // KISS-ANNOUNCE §6.2-0011: reserved bytes MUST be zero on the wire; a
        // conforming reader hard-rejects a nonzero reserved field.
        let h = baracuda_hello();
        assert_eq!(h.reserved, [0u8; 3]);
        assert_eq!(h.reserved1, [0u8; 6]);
    }

    #[test]
    fn seam_magic_wire_bytes_spell_seam() {
        // KISS-ANNOUNCE §6.1-0004: the `magic` field MUST equal 0x4D414553 read
        // as a little-endian u32, so the on-wire bytes at offset 0 are
        // `53 45 41 4D` = ASCII "SEAM". (0x5345414D spells "SEAM" only big-endian
        // and puts "MAES" on a little-endian wire.) Lockstep with Fuel's
        // `fuel-kernel-seam-announce` seed (commit 1849bc9a).
        assert_eq!(SEAM_MAGIC, 0x4D41_4553);
        assert_eq!(SEAM_MAGIC.to_le_bytes(), *b"SEAM");
    }

    #[test]
    fn validate_accepts_baracuda_hello() {
        assert_eq!(baracuda_hello().validate(), Ok(()));
    }

    #[test]
    fn validate_rejects_bad_magic() {
        let mut h = baracuda_hello();
        h.magic = 0x5345_414D; // the old big-endian "MAES"-on-wire value
        assert_eq!(h.validate(), Err(SeamError::BadMagic));
    }

    #[test]
    fn validate_rejects_nonzero_reserved() {
        let mut h = baracuda_hello();
        h.reserved1[0] = 1;
        assert_eq!(h.validate(), Err(SeamError::ReservedNonZero));
        let mut h2 = baracuda_hello();
        h2.reserved[2] = 0xFF;
        assert_eq!(h2.validate(), Err(SeamError::ReservedNonZero));
    }

    #[test]
    fn validate_rejects_out_of_range_profiles_len() {
        let mut h = baracuda_hello();
        h.profiles_len = (SEAM_MAX_PROFILES + 1) as u16;
        assert_eq!(
            h.validate(),
            Err(SeamError::ProfilesLenOutOfRange(
                (SEAM_MAX_PROFILES + 1) as u16
            ))
        );
    }

    #[test]
    fn negotiate_picks_highest_mutual_profile() {
        let mut a = baracuda_hello();
        a.profiles[..3].copy_from_slice(&[1, 2, 5]);
        a.profiles_len = 3;
        let mut b = baracuda_hello();
        b.profiles[..3].copy_from_slice(&[2, 5, 9]);
        b.profiles_len = 3;
        assert_eq!(a.negotiate(&b), Some(5)); // 2 and 5 shared; 5 is highest
    }

    #[test]
    fn negotiate_returns_none_on_empty_intersection() {
        let mut a = baracuda_hello();
        a.profiles[..2].copy_from_slice(&[1, 2]);
        a.profiles_len = 2;
        let mut b = baracuda_hello();
        b.profiles[..2].copy_from_slice(&[3, 4]);
        b.profiles_len = 2;
        assert_eq!(a.negotiate(&b), None); // hard fail, no panic
    }

    #[test]
    fn wire_round_trips_and_spells_seam_at_offset_zero() {
        let h = baracuda_hello();
        let bytes = h.to_wire_bytes();
        assert_eq!(&bytes[0..4], b"SEAM"); // LE magic on the wire
        let parsed = SeamHello::parse_hello(&bytes).expect("a valid envelope round-trips");
        assert_eq!(parsed, h);
    }

    #[test]
    fn parse_hello_rejects_truncated() {
        assert_eq!(
            SeamHello::parse_hello(&[0u8; 40]),
            Err(SeamError::Truncated)
        );
    }

    #[test]
    fn parse_hello_rejects_nonzero_reserved_on_wire() {
        let mut bytes = baracuda_hello().to_wire_bytes();
        bytes[42] = 1; // first byte of reserved1 (offset 42)
        assert_eq!(
            SeamHello::parse_hello(&bytes),
            Err(SeamError::ReservedNonZero)
        );
    }
}
