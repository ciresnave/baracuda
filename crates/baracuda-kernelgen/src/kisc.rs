//! KISC self-delimiting contract framing (KISS-Contract §2.8 / §6.11).
//!
//! A KISC document is a **structured/text** frame — not a binary envelope: a
//! single header line carrying the magic, kind, version, inner body byte-length,
//! and a CRC-32 over the body, followed by the body itself. A reader hard-rejects
//! (never repairs) a document whose magic is wrong, whose header is malformed,
//! whose kind/version is unrecognized, or whose declared length/CRC don't match —
//! which is what kills the silent-adopt-empty and truncation hazards the older
//! `## `-heading framing was prone to.
//!
//! # Header-line format — PROVISIONAL
//!
//! KISS §6.11 pins the header's *fields* (magic `KISC`, `kiss-contract` kind, a
//! version, `len=<N>`, `crc32=<…>`) but not yet the exact literal bytes. The
//! spelling below is a concrete **strawman** proposed to Fuel for co-pinning
//! (`docs/fuel-reply-kisc-framing-2026-07-15.md`) and to KISS §6.11 as a golden
//! vector. It is isolated in [`kisc_frame`]/[`kisc_unframe`] so pinning the final
//! form is a localized change.
//!
//! ```text
//! KISC kiss-contract 1 len=<N> crc32=<8 lowercase hex>\n<body of N bytes>
//! ```

/// The pinned contract kind (KISS-Contract §6.11).
pub const KISC_KIND: &str = "kiss-contract";
/// The KISC document version this seed emits and accepts.
pub const KISC_VERSION: u32 = 1;

/// Why a KISC document is rejected. Hard-reject discipline (§6.11): the reader
/// never repairs and never adopts a malformed document as an empty/no-op contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KiscError {
    /// The document does not begin with the `KISC` magic.
    BadMagic,
    /// The header line is absent or structurally malformed.
    BadHeader,
    /// The `contract_kind` is not `kiss-contract`.
    UnknownKind,
    /// The `contract_version` is not one this seed understands.
    UnknownVersion,
    /// The declared `len=<N>` does not equal the actual body byte length.
    LenMismatch,
    /// The declared `crc32=<…>` does not match the body's CRC-32.
    CrcMismatch,
}

/// CRC-32 (IEEE 802.3, reflected, `0xEDB8_8320`), the checksum KISS §6.11 names.
#[must_use]
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Frame `body` as a single KISC document (see the module header for the format).
#[must_use]
pub fn kisc_frame(body: &str) -> String {
    format!(
        "KISC {KISC_KIND} {KISC_VERSION} len={} crc32={:08x}\n{body}",
        body.len(),
        crc32(body.as_bytes()),
    )
}

/// Parse + validate a single KISC document, returning its body on success or a
/// typed hard-reject. Never panics.
pub fn kisc_unframe(doc: &str) -> Result<&str, KiscError> {
    let (header, body) = doc.split_once('\n').ok_or(KiscError::BadHeader)?;
    let mut f = header.split(' ');
    match f.next() {
        Some("KISC") => {}
        _ => return Err(KiscError::BadMagic),
    }
    if f.next() != Some(KISC_KIND) {
        return Err(KiscError::UnknownKind);
    }
    match f.next().and_then(|v| v.parse::<u32>().ok()) {
        Some(KISC_VERSION) => {}
        _ => return Err(KiscError::UnknownVersion),
    }
    let len = f
        .next()
        .and_then(|s| s.strip_prefix("len="))
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or(KiscError::BadHeader)?;
    let crc = f
        .next()
        .and_then(|s| s.strip_prefix("crc32="))
        .and_then(|s| u32::from_str_radix(s, 16).ok())
        .ok_or(KiscError::BadHeader)?;
    if f.next().is_some() {
        return Err(KiscError::BadHeader);
    }
    if body.len() != len {
        return Err(KiscError::LenMismatch);
    }
    if crc32(body.as_bytes()) != crc {
        return Err(KiscError::CrcMismatch);
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_standard_check_value() {
        // The canonical CRC-32/ISO-HDLC check value for "123456789".
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn frame_carries_magic_kind_version_len_and_body() {
        let f = kisc_frame("hello");
        assert!(
            f.starts_with("KISC kiss-contract 1 len=5 crc32="),
            "header line: {f:?}"
        );
        assert!(
            f.ends_with("\nhello"),
            "body follows the header line: {f:?}"
        );
    }

    #[test]
    fn frame_unframe_round_trips_a_multiline_body() {
        // A realistic body has interior newlines (the seven contract sections).
        let body = "kernel: relu_add\nop_kind: ReluAddElementwise\naccept: sk1|...\n";
        let doc = kisc_frame(body);
        assert_eq!(kisc_unframe(&doc), Ok(body));
    }

    #[test]
    fn unframe_hard_rejects_a_magic_less_document() {
        // A `## `-heading document (the OLD framing) is magic-less → rejected, not
        // silently adopted as an empty contract.
        assert_eq!(
            kisc_unframe("## heading\n\nkernel: x\n"),
            Err(KiscError::BadMagic)
        );
        // No header line at all.
        assert_eq!(kisc_unframe(""), Err(KiscError::BadHeader));
    }

    #[test]
    fn unframe_hard_rejects_a_corrupted_body() {
        let mut doc = kisc_frame("kernel: relu_add\n");
        // Flip a byte in the body; the CRC (or length) must catch it.
        doc.push('X');
        assert!(matches!(
            kisc_unframe(&doc),
            Err(KiscError::LenMismatch) | Err(KiscError::CrcMismatch)
        ));
    }
}
