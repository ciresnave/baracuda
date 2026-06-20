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

/// `"SEAM"` — the envelope magic (§3.1). Never changes.
pub const SEAM_MAGIC: u32 = 0x5345_414D;
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
#[derive(Copy, Clone, Debug)]
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
    /// Optional-feature bitset within the selected profile (§3.4).
    pub capabilities: u64,
}

// The envelope is frozen at 56 bytes (§3.1); cross-check it the way FDX does.
const _: () = assert!(core::mem::size_of::<SeamHello>() == 56);
const _: () = assert!(core::mem::align_of::<SeamHello>() == 8);

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
/// JIT-on-request endpoint implemented (§5). Off until both sides build it.
pub const SEAM_CAP_JIT_ON_REQUEST: u64 = 1 << 32;

/// The capabilities Baracuda advertises at first connect: the FDX tokens for its
/// shipped kernel families. `SeamCapJitOnRequest` is deliberately **off** (§5 is
/// design-only until both sides build the base-emission seam).
pub const BARACUDA_CAPABILITIES: u64 = SEAM_CAP_DLPACK_EXT_V1
    | SEAM_CAP_DLPACK_EXT_MX
    | SEAM_CAP_DLPACK_EXT_GGML
    | SEAM_CAP_DLPACK_EXT_AFFINE
    | SEAM_CAP_DLPACK_EXT_SYMBOLIC
    | SEAM_CAP_DLPACK_EXT_GATHER;

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
        // JIT off at first connect; FDX tokens on.
        assert_eq!(h.capabilities & SEAM_CAP_JIT_ON_REQUEST, 0);
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
}
