//! §6.8-0008 machine-manifest generator (KISS-Classify) for the `cuda:` namespace.
//!
//! Emits the machine-readable form of the annex from [`CATALOG`] as canonical
//! JSON. Every member is **derived from the codec** ([`Entry::token`] →
//! `sm_number`/`class`), so the manifest cannot spell a token the emitter would
//! not — the same anti-drift property the crate is built on, carried into the
//! generated artifact.
//!
//! Provenance vs. source (§6.8-0011, as amended): [`Manifest::generated_from`]
//! **names the producer** — this crate's codec — and is *not* a claim that the
//! producer is the normative source. The normative annex is KISS's registered
//! `spec/namespaces/cuda.md`; agreement between this manifest and that annex is a
//! separate relation, owed by the agreement gate (a follow-on to this generator).
//!
//! This module builds the manifest and self-checks it (well-formed, round-trips
//! [`CATALOG`], byte-deterministic). It deliberately does **not** wire the
//! agreement gate against the registered annex — that binds a document in the KISS
//! tree and is built once its comparison shape is settled.

use crate::{CATALOG, Entry, TokenClass};
use serde::{Deserialize, Serialize};

/// The §6.8-0008 `cuda:` vocabulary manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// `kiss-namespace-vocabulary-v1` — the §6.8-0008 schema id.
    pub schema: String,
    /// The namespace this manifest describes: `cuda`.
    pub namespace: String,
    /// Integer version (§6.8-0008): `1` for `cuda-vocab v1`.
    pub vocabulary_version: u32,
    /// `enumerated` — cuda's sm-set is closed/finite, never a `generated` product.
    pub kind: String,
    /// The maintainer that owns this namespace annex.
    pub maintainer: String,
    /// The crate the §6.8-0003 registry points at as the reference implementation.
    pub reference_implementation: String,
    /// Names the PRODUCER of this manifest (§6.8-0011: provenance, not source).
    pub generated_from: String,
    /// The token grammar, hoisted top-level per the §6.8-0008 field review.
    pub grammar: String,
    /// The codec-neutral encoding layer (cuda.md §1).
    pub encoding: Encoding,
    /// The closed, enumerated token set — the members of `cuda-vocab v1`.
    pub members: Vec<Member>,
    /// Governs token↔DEVICE admission (may this device RUN a kernel built for this
    /// token), never token↔token — matching stays byte-exact under §6.8-0002.
    pub device_admission: DeviceAdmission,
    /// The canonical external ordering of members (cuda.md §5).
    pub ordering: String,
    /// Which §6.8-wide obligations do and don't apply to this closed scalar set.
    pub coverage_note: String,
    /// What constitutes a version bump for this vocabulary (cuda.md §6).
    pub versioning: String,
}

/// The codec-neutral encoding layer (cuda.md §1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Encoding {
    /// The character set allowed after the `cuda:` prefix (`[a-z0-9]`, §cuda-1.3).
    pub charset_after_prefix: String,
    /// How tokens are compared: `byte-exact` (§6.8-0002).
    pub comparison: String,
    /// The capability-set shape: `single-scalar` (§4) — one token, never a list.
    pub capability_set_shape: String,
}

/// One enumerated member. Every field is derived from the codec token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Member {
    /// The canonical `cuda:` token (from the codec).
    pub token: String,
    /// `base` | `a` (cuda.md §2), derived from the token's trailing `a`.
    pub class: String,
    /// The CUDA `-arch` target, e.g. `sm_90a`.
    pub target: String,
    /// The decimal after `sm` (cuda.md §3), parsed from the token.
    pub sm_number: u32,
}

/// The two independent admission relations (cuda.md §3), as text — this is a
/// human/spec artifact; the executable relations live in [`Entry::admits`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceAdmission {
    /// How a device's compute capability is encoded (`major*10 + minor`).
    pub device_sm_number: String,
    /// §cuda-3.1: a base token admits under `≤` (forward-compatible up-arch).
    pub base_clause: String,
    /// §cuda-3.2: an `a` token admits only under `==` (exact architecture match).
    pub a_clause: String,
}

fn class_str(c: TokenClass) -> &'static str {
    match c {
        TokenClass::Base => "base",
        TokenClass::Exclusive => "a",
    }
}

impl Member {
    fn from_entry(e: &Entry) -> Self {
        let sm = e.sm_number();
        let suffix = if e.class() == TokenClass::Exclusive {
            "a"
        } else {
            ""
        };
        Member {
            token: e.token(),
            class: class_str(e.class()).to_string(),
            target: format!("sm_{sm}{suffix}"),
            sm_number: sm,
        }
    }
}

impl Manifest {
    /// Generate the manifest from [`CATALOG`]. Members are derived from the codec,
    /// so the manifest cannot spell a token the emitter would not.
    pub fn generate() -> Self {
        Manifest {
            schema: "kiss-namespace-vocabulary-v1".to_string(),
            namespace: "cuda".to_string(),
            vocabulary_version: 1,
            kind: "enumerated".to_string(),
            maintainer: "baracuda".to_string(),
            reference_implementation: "baracuda-cuda-vocab".to_string(),
            generated_from: "baracuda-cuda-vocab (unpopped-vocab `From<ArchSku>` codec)"
                .to_string(),
            grammar: "cuda:sm<digits>[a]".to_string(),
            encoding: Encoding {
                charset_after_prefix: "[a-z0-9]".to_string(),
                comparison: "byte-exact".to_string(),
                capability_set_shape: "single-scalar".to_string(),
            },
            members: CATALOG.iter().map(Member::from_entry).collect(),
            device_admission: DeviceAdmission {
                device_sm_number: "major*10 + minor".to_string(),
                base_clause: "device D admits base token T iff T.sm_number <= D.sm_number"
                    .to_string(),
                a_clause: "device D admits a-token T iff T.sm_number == D.sm_number".to_string(),
            },
            ordering: "sort by (sm_number ascending, variant: base < a)".to_string(),
            coverage_note: "closed enumeration; a cuda: capability-set is a single scalar token \
                            (§4), so the §6.8-0007 length-digest and §6.8-0013 vector obligations \
                            do not apply"
                .to_string(),
            versioning: "adding or removing a token, altering an admission relation, or changing \
                         the encoding layer is a version bump"
                .to_string(),
        }
    }

    /// Canonical JSON: `serde_json` pretty (2-space indent, struct field order)
    /// plus a trailing newline. Deterministic — the byte artifact the freshness /
    /// agreement gate compares against.
    pub fn to_json(&self) -> String {
        let mut s = serde_json::to_string_pretty(self).expect("Manifest serializes");
        s.push('\n');
        s
    }
}

/// The generated §6.8-0008 `cuda:` manifest as canonical JSON.
pub fn manifest_json() -> String {
    Manifest::generate().to_json()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The RENDERED string values carry no stray whitespace runs. A byte-diff
    /// freshness gate compares these exact bytes, and a `\` line-continuation in a
    /// source string literal keeps the next line's leading whitespace — a known
    /// footgun where "deterministic" is not "clean" (deterministic stray spaces are
    /// still deterministic, so the round-trip/determinism tests can't see it). This
    /// reads the RENDERED values, not the source: it parses the manifest and asserts
    /// no string value contains a run of spaces. JSON indentation is structural and
    /// lives outside string values, so this checks content, not layout.
    #[test]
    fn string_values_carry_no_whitespace_runs() {
        fn check(v: &serde_json::Value) {
            match v {
                serde_json::Value::String(s) => {
                    assert!(
                        !s.contains("  "),
                        "string value has a whitespace run: {s:?}"
                    )
                }
                serde_json::Value::Array(a) => a.iter().for_each(check),
                serde_json::Value::Object(o) => o.values().for_each(check),
                _ => {}
            }
        }
        check(&serde_json::from_str(&manifest_json()).unwrap());
    }

    /// Well-formed + stable: the JSON round-trips through `serde_json` back to an
    /// equal `Manifest`. Proves the emitted bytes are valid JSON and lossless.
    #[test]
    fn manifest_round_trips_through_json() {
        let m = Manifest::generate();
        let parsed: Manifest = serde_json::from_str(&m.to_json()).expect("valid JSON");
        assert_eq!(parsed, m);
    }

    /// The manifest's members ARE the catalog, member-for-member: same count, and
    /// each member's token/class/sm_number is exactly the codec-derived value. So
    /// the generated artifact cannot silently disagree with CATALOG.
    #[test]
    fn members_round_trip_the_catalog() {
        let m = Manifest::generate();
        assert_eq!(m.members.len(), CATALOG.len());
        for (mem, e) in m.members.iter().zip(CATALOG) {
            assert_eq!(mem.token, e.token());
            assert_eq!(mem.sm_number, e.sm_number());
            assert_eq!(mem.class, class_str(e.class()));
        }
    }

    /// Byte-determinism: two generations produce identical bytes. The freshness /
    /// agreement gate is a byte-compare, so a non-deterministic manifest would make
    /// the gate flap; this pins it.
    #[test]
    fn manifest_json_is_byte_deterministic() {
        assert_eq!(manifest_json(), manifest_json());
    }

    /// The §6.8-0008 required fields are present under their required names. Guards
    /// against a rename dropping a field the schema mandates.
    #[test]
    fn required_6_8_0008_fields_are_present() {
        let v: serde_json::Value = serde_json::from_str(&manifest_json()).unwrap();
        for key in [
            "schema",
            "vocabulary_version",
            "kind",
            "generated_from",
            "grammar",
            "members",
            "device_admission",
            "coverage_note",
        ] {
            assert!(v.get(key).is_some(), "§6.8-0008 field `{key}` missing");
        }
        assert_eq!(v["schema"], "kiss-namespace-vocabulary-v1");
        assert_eq!(v["kind"], "enumerated");
        assert_eq!(v["vocabulary_version"], 1);
    }

    /// PRE-DECLARED DRIFT (not a bug): this manifest lists FOUR members, while
    /// KISS's registered SSOT seed in `spec/namespaces/cuda.md` currently lists
    /// THREE — it omits `cuda:sm90` although `ArchSku::Sm90` is wired, so KISS's
    /// own §2 table (four) and its appendix seed (three) disagree. The agreement
    /// gate (a follow-on, binding KISS's registered annex) will therefore red by
    /// exactly one member on its FIRST run: that is the gate catching live drift on
    /// day one, and it clears when the seed row lands (KISS #334). This test pins
    /// the count so that expected red is predicted, never diagnosed as our bug.
    #[test]
    fn manifest_lists_four_members_seed_lists_three_expect_agreement_red_by_one() {
        assert_eq!(
            Manifest::generate().members.len(),
            4,
            "cuda-vocab v1 has four members; the KISS seed's three is the stale side"
        );
    }
}
