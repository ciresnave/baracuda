//! `baracuda-cuda-vocab` — owner source for the `cuda:` KISS §6.8 namespace
//! vocabulary.
//!
//! This crate is the machine-readable owner of what `spec/namespaces/cuda.md`
//! states in prose: the closed sm-token set (§2), each token's `base`|`a` class
//! (§2), and the two admission relations (§3). It exists so the §6.8-0008 machine
//! annex can be **generated** from a single source and **drift-gated** against the
//! KISS-committed copy — the mechanism that makes a `cuda:sm90`-class skew (§2 and
//! §4/§6 disagreeing on membership) impossible rather than merely caught late.
//!
//! **Token spellings are never hand-written here.** Each entry names an
//! [`unpopped_vocab::ArchSku`], and the `cuda:` bytes come from that crate's
//! `From<ArchSku> for TargetId` codec ([`Entry::token`]) — so a catalog token can
//! never drift from what a conforming peer emits. (The namespace-machine-spec RFC's
//! end state migrates `ArchSku` ownership *into* this crate and makes
//! `unpopped-vocab` token-opaque; until that release ships we CONSUME its codec,
//! which keeps one source and no transitional dual-ownership window — the very
//! drift this crate exists to prevent.)
//!
//! What lives here vs. not: the token set, classes, and admission relations are
//! this crate's to own. It does not re-implement the codec (that stays in
//! `unpopped-vocab`), and the §6.8-0008 manifest generator + drift-gate build on
//! [`CATALOG`] in follow-on work.

use unpopped_vocab::{ArchSku, TargetId};

/// The class of an sm-token (`spec/namespaces/cuda.md` §2), which selects its
/// admission relation in §3. The class — not the token bytes — is what makes the
/// vocabulary a `(sm_number, class)` space rather than a flat set: `sm90` and
/// `sm90a` share `sm_number` 90 but differ here, and admit under opposite clauses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenClass {
    /// No trailing `a`. Forward-compatible up-arch (PTX/JIT). Admits under the
    /// §cuda-3.1 `≤` relation.
    Base,
    /// Trailing `a`. Architecture-**exclusive** SASS, no forward-compatibility.
    /// Admits under the §cuda-3.2 `==` relation only.
    Exclusive,
}

/// One `cuda-vocab v1` entry: an sm-token, its class, and its `sm_number`.
///
/// The token spelling is **not** stored — it is derived from `sku` via
/// `unpopped-vocab`'s codec ([`Entry::token`]), so a catalog token can never drift
/// from what a conforming peer emits.
#[derive(Debug, Clone, Copy)]
pub struct Entry {
    /// The arch SKU — the byte source of truth for the token spelling.
    pub sku: ArchSku,
    /// `base` | `a` (§2), which selects the admission clause (§3).
    pub class: TokenClass,
    /// The decimal after `sm`, e.g. `sm90a`.`sm_number` = 90 (§3 definition).
    pub sm_number: u32,
}

impl Entry {
    /// The canonical `cuda:` token bytes, from `unpopped-vocab`'s `From<ArchSku>`
    /// codec. Never hand-spelled — this is the anti-drift lock.
    pub fn token(&self) -> String {
        TargetId::from(self.sku).as_str()
    }

    /// §3 admission: whether a device at compute capability `device_sm` (encoded
    /// `major × 10 + minor`, e.g. Hopper 9.0 → 90) admits a kernel bearing this
    /// token. Base tokens admit forward-compatibly (`≤`, §cuda-3.1); `a` tokens
    /// admit only on an exact arch match (`==`, §cuda-3.2) — never widen an `a`
    /// token to `≤`, which is the silent-merge hazard §3 exists to prevent.
    pub fn admits(&self, device_sm: u32) -> bool {
        match self.class {
            TokenClass::Base => self.sm_number <= device_sm,
            TokenClass::Exclusive => self.sm_number == device_sm,
        }
    }
}

/// The closed, exhaustive `cuda-vocab v1` token set (`spec/namespaces/cuda.md`
/// §2 table / §6 set): `{ sm80, sm89, sm90, sm90a }`, in §5 canonical order
/// (`sm_number` ascending, then `base` before `a`).
///
/// `cuda:sm90` is a member: KISS's §6.7 reference vectors name it, so a reader
/// that drops it rejects conforming vectors. (`unpopped-vocab`'s `ArchSku::Sm90`
/// doc says as much.)
pub const CATALOG: &[Entry] = &[
    Entry {
        sku: ArchSku::Sm80,
        class: TokenClass::Base,
        sm_number: 80,
    },
    Entry {
        sku: ArchSku::Sm89,
        class: TokenClass::Base,
        sm_number: 89,
    },
    Entry {
        sku: ArchSku::Sm90,
        class: TokenClass::Base,
        sm_number: 90,
    },
    Entry {
        sku: ArchSku::Sm90a,
        class: TokenClass::Exclusive,
        sm_number: 90,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Anti-drift lock: every catalog token is byte-identical to what
    /// `unpopped-vocab`'s codec emits. If the codec ever respelled a token this
    /// reds — the catalog can't silently diverge from the byte source of truth.
    #[test]
    fn catalog_tokens_match_the_unpopped_vocab_codec() {
        let got: Vec<String> = CATALOG.iter().map(Entry::token).collect();
        assert_eq!(got, ["cuda:sm80", "cuda:sm89", "cuda:sm90", "cuda:sm90a"]);
    }

    /// The catalog is exactly `cuda-vocab v1`'s four members (§2 table / §6 set),
    /// `cuda:sm90` included — the membership KISS's §6.7 vectors require and that
    /// PR #37 reconciled §4/§5/§6 to match §2 on.
    #[test]
    fn catalog_is_cuda_vocab_v1() {
        assert_eq!(CATALOG.len(), 4);
        let skus: Vec<ArchSku> = CATALOG.iter().map(|e| e.sku).collect();
        assert_eq!(
            skus,
            [ArchSku::Sm80, ArchSku::Sm89, ArchSku::Sm90, ArchSku::Sm90a]
        );
    }

    /// §5 canonical order: `sm_number` ascending, then `base` before `a` at a tie.
    #[test]
    fn catalog_is_in_canonical_order() {
        let sort_key = |e: &Entry| (e.sm_number, matches!(e.class, TokenClass::Exclusive));
        for w in CATALOG.windows(2) {
            assert!(
                sort_key(&w[0]) < sort_key(&w[1]),
                "{} must sort before {}",
                w[0].token(),
                w[1].token()
            );
        }
    }

    /// §cuda-3.1: a base token admits forward-compatibly (`≤`).
    #[test]
    fn base_tokens_admit_forward_compatibly() {
        let sm80 = CATALOG[0];
        assert_eq!(sm80.class, TokenClass::Base);
        assert!(sm80.admits(80)); // its own arch
        assert!(sm80.admits(89)); // up-arch Ada
        assert!(sm80.admits(90)); // up-arch Hopper
        assert!(sm80.admits(100)); // up-arch Blackwell
        assert!(!sm80.admits(70)); // an older device can't run an sm80 kernel
    }

    /// §cuda-3.2: an `a` token admits ONLY on an exact arch match (`==`) — the
    /// silent-merge hazard the two-relation split exists to prevent.
    #[test]
    fn exclusive_tokens_admit_only_on_exact_match() {
        let sm90a = CATALOG[3];
        assert_eq!(sm90a.class, TokenClass::Exclusive);
        assert!(sm90a.admits(90)); // its target, Hopper
        assert!(!sm90a.admits(100)); // NOT forward-compatible up-arch
        assert!(!sm90a.admits(89)); // NOT backward
    }

    /// The load-bearing discriminator: `sm90` (base) and `sm90a` (`a`) share
    /// `sm_number` 90 but take OPPOSITE admission clauses. A flat token set that
    /// can't tell the two 90s apart cannot express this — which is exactly why the
    /// listed→reachable witness gate must key on `(sm_number, class)`, not a token.
    #[test]
    fn the_two_90s_take_opposite_admission_clauses() {
        let sm90 = CATALOG[2];
        let sm90a = CATALOG[3];
        assert_eq!(sm90.sm_number, sm90a.sm_number);
        assert_ne!(sm90.class, sm90a.class);
        assert!(sm90.admits(100)); // base rides `≤` — runs up-arch on Blackwell
        assert!(!sm90a.admits(100)); // `a` rides `==` — does not
    }
}
