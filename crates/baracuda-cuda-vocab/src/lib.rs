//! `baracuda-cuda-vocab` — owner source for the `cuda:` KISS §6.8 namespace
//! vocabulary.
//!
//! This crate is the machine-readable owner of what `spec/namespaces/cuda.md`
//! states in prose: the closed sm-token set (§2), each token's `base`|`a` class
//! (§2), and the two admission relations (§3). It exists so the §6.8-0008 machine
//! annex can be **generated** from a single source and **agreement-gated** against
//! the prose annex — the mechanism that makes a `cuda:sm90`-class skew (§2 and
//! §4/§6 disagreeing on membership) impossible rather than merely caught late.
//!
//! **Nothing derived is hand-written.** An [`Entry`] holds only its
//! [`unpopped_vocab::ArchSku`]; the token bytes come from that crate's
//! `From<ArchSku> for TargetId` codec ([`Entry::token`]), and the `sm_number`
//! ([`Entry::sm_number`]) and class ([`Entry::class`]) — the two values the
//! admission relations actually read — are **parsed from that token**, not stored.
//! So the fields `admits` keys on cannot drift from the codec's spelling: a wrong
//! `sm_number` for `sm90a` is not representable, because there is no `sm_number`
//! field to get wrong. The only hand-authored fact is CATALOG *membership* (which
//! `ArchSku` variants are in v1), and that is asserted against the §2/§6 set.
//!
//! (The namespace-machine-spec RFC's end state migrates `ArchSku` ownership *into*
//! this crate and makes `unpopped-vocab` token-opaque; until that release ships we
//! CONSUME its codec, which keeps one source and no transitional dual-ownership
//! window — the very drift this crate exists to prevent. The §6.8-0008 manifest
//! generator + agreement gate build on [`CATALOG`] in follow-on work.)

use unpopped_vocab::{ArchSku, TargetId};

/// The class of an sm-token (`spec/namespaces/cuda.md` §2), which selects its
/// admission relation in §3. The class is what makes the vocabulary a
/// `(sm_number, class)` space rather than a flat set: `sm90` and `sm90a` share
/// `sm_number` 90 but differ here, and admit under opposite clauses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenClass {
    /// No trailing `a`. Forward-compatible up-arch (PTX/JIT). Admits under the
    /// §cuda-3.1 `≤` relation.
    Base,
    /// Trailing `a`. Architecture-**exclusive** SASS, no forward-compatibility.
    /// Admits under the §cuda-3.2 `==` relation only.
    Exclusive,
}

/// One `cuda-vocab v1` entry. It holds **only** its [`ArchSku`]; everything the
/// admission relations read is derived from the codec-produced token, so it cannot
/// drift from what a conforming peer emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entry {
    /// The arch SKU — the single source of truth for this entry.
    pub sku: ArchSku,
}

impl Entry {
    /// The canonical `cuda:` token bytes, from `unpopped-vocab`'s `From<ArchSku>`
    /// codec. Never hand-spelled — this is the byte source of truth from which
    /// [`Self::sm_number`] and [`Self::class`] are then derived.
    pub fn token(&self) -> String {
        TargetId::from(self.sku).as_str()
    }

    /// The `sm_number` (§3), **parsed from the token** — the decimal run after
    /// `cuda:sm`, e.g. `cuda:sm90a` → 90. Derived, not stored, so it cannot be
    /// given a value inconsistent with the token the codec emits.
    pub fn sm_number(&self) -> u32 {
        let token = self.token();
        let digits: String = token
            .strip_prefix("cuda:sm")
            .expect("a cuda: token from the codec begins `cuda:sm`")
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        digits
            .parse()
            .expect("a cuda: token carries at least one digit after `sm`")
    }

    /// The class (§2), **derived from the token's trailing `a`** (§cuda-1.3's
    /// optional suffix): an `a`-suffixed token is [`TokenClass::Exclusive`], every
    /// other is [`TokenClass::Base`]. Derived, not stored — the class and the token
    /// suffix cannot disagree.
    pub fn class(&self) -> TokenClass {
        if self.token().ends_with('a') {
            TokenClass::Exclusive
        } else {
            TokenClass::Base
        }
    }

    /// §3 admission: whether a device at compute capability `device_sm` (encoded
    /// `major × 10 + minor`, e.g. Hopper 9.0 → 90) admits a kernel bearing this
    /// token. Base tokens admit forward-compatibly (`≤`, §cuda-3.1); `a` tokens
    /// admit only on an exact arch match (`==`, §cuda-3.2) — never widen an `a`
    /// token to `≤`, which is the silent-merge hazard §3 exists to prevent. Both
    /// operands (`sm_number`, `class`) are token-derived, so this reads the codec.
    pub fn admits(&self, device_sm: u32) -> bool {
        match self.class() {
            TokenClass::Base => self.sm_number() <= device_sm,
            TokenClass::Exclusive => self.sm_number() == device_sm,
        }
    }
}

/// The closed, exhaustive `cuda-vocab v1` token set (`spec/namespaces/cuda.md`
/// §2 table / §6 set): `{ sm80, sm89, sm90, sm90a }`, in §5 canonical order
/// (`sm_number` ascending, then `base` before `a`).
///
/// The only hand-authored fact here is *membership* — which `ArchSku` variants
/// are in v1. `cuda:sm90` is one: KISS's §6.7 reference vectors name it, so a
/// reader that drops it rejects conforming vectors. (`unpopped-vocab`'s
/// `ArchSku::Sm90` doc says as much.)
pub const CATALOG: &[Entry] = &[
    Entry { sku: ArchSku::Sm80 },
    Entry { sku: ArchSku::Sm89 },
    Entry { sku: ArchSku::Sm90 },
    Entry {
        sku: ArchSku::Sm90a,
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
    /// PR #37 reconciled §4/§5/§6 to match §2 on. Membership is the one
    /// hand-authored fact, so it is the one asserted against the expected set.
    #[test]
    fn catalog_is_cuda_vocab_v1() {
        assert_eq!(CATALOG.len(), 4);
        let skus: Vec<ArchSku> = CATALOG.iter().map(|e| e.sku).collect();
        assert_eq!(
            skus,
            [ArchSku::Sm80, ArchSku::Sm89, ArchSku::Sm90, ArchSku::Sm90a]
        );
    }

    /// The derivation is correct for each member: `sm_number` and `class` come out
    /// of the token exactly as §2 states. This tests the PARSER — the mechanism
    /// that makes the derived fields trustworthy — against the known grammar, so a
    /// bug in `sm_number`/`class` can't ride in behind the token byte-lock.
    #[test]
    fn derivation_matches_the_grammar() {
        let expect = [
            ("cuda:sm80", 80, TokenClass::Base),
            ("cuda:sm89", 89, TokenClass::Base),
            ("cuda:sm90", 90, TokenClass::Base),
            ("cuda:sm90a", 90, TokenClass::Exclusive),
        ];
        for (e, (tok, sm, class)) in CATALOG.iter().zip(expect) {
            assert_eq!(e.token(), tok);
            assert_eq!(e.sm_number(), sm, "sm_number of {tok}");
            assert_eq!(e.class(), class, "class of {tok}");
        }
    }

    /// §5 canonical order: `sm_number` ascending, then `base` before `a` at a tie.
    #[test]
    fn catalog_is_in_canonical_order() {
        let sort_key = |e: &Entry| (e.sm_number(), matches!(e.class(), TokenClass::Exclusive));
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
        assert_eq!(sm80.class(), TokenClass::Base);
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
        assert_eq!(sm90a.class(), TokenClass::Exclusive);
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
        assert_eq!(sm90.sm_number(), sm90a.sm_number());
        assert_ne!(sm90.class(), sm90a.class());
        assert!(sm90.admits(100)); // base rides `≤` — runs up-arch on Blackwell
        assert!(!sm90a.admits(100)); // `a` rides `==` — does not
    }
}
