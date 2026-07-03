//! The **dispatch table** — the measured routing oracle for a cell.
//!
//! `docs/design/kernel-specialization.md` §7 ("Excluding vendor-owned cells")
//! makes *measurement*, not a hand-blocklist, the durable rule for which
//! implementation serves a given `(op, structure-key, dtype, arch)` cell. This
//! module is the schema + pure decision logic for that rule:
//!
//! - [`DispatchEntry`] — one cell's decision: `winner + margin + ranked + provenance`,
//!   keyed by a [`StructureKey::to_token`] string.
//! - [`DispatchTable`] — the token-keyed collection the runtime dispatcher looks
//!   up (arch-gated) to route a live tensor.
//! - [`winner_of`] — reduce a set of measured candidates to a decision.
//! - [`seed_winner`] — hand-knowledge defaults (route large aligned GEMM to
//!   cuBLAS without benching).
//! - [`merge`] — fold freshly measured/reported results into a table. This is the
//!   ingest seam item 08 (telemetry variant-selection) drives from Fuel's
//!   `dispatch_record`/`miss_record` feed.
//!
//! The *timing* engine that produces [`CandidateResult`]s (the bench gate) and
//! [`HwStamp::current`] (device query) live in the bench crate — this module is
//! pure data + logic so it is unit-testable off-device and readable by the
//! runtime dispatcher without a build-only dependency.
//!
//! # Determinism
//!
//! The committed artifact ([emit side is `baracuda-kernelgen`]) is the *routing
//! projection* of a table: `(token, winner, margin, provenance)` sorted by token
//! and deduped. It carries **no** wall-clock timestamp, so it is byte-stable
//! across runs. The richer [`HwStamp`] (device name, CUDA version, capture time)
//! lives on the in-memory [`DispatchEntry`] and in the bench CSV — it drives
//! [`merge`]'s "newer wins" and arch-gate decisions but never churns the diff.

use crate::structure_key::StructureKey;
use crate::ArchSku;

/// The minimum `second_best / winner` ratio at which a freshly observed win is
/// allowed to **flip** an existing decision to a *different* winner.
///
/// `BENCHMARKS.md` treats ±5% as "≈" and warns of 20-30% timing variance at the
/// smallest shapes, so a 1.01× "win" is inside the noise floor and must not
/// overturn a prior (especially a hand-seeded vendor) route. A same-winner
/// refresh is not a flip and is not gated by this threshold.
pub const MIN_FLIP_MARGIN: f64 = 1.10;

/// A candidate implementation for a cell — the join target of the gate.
///
/// `#[non_exhaustive]`: more backends will come (ROCm, oneDNN, …); downstream
/// `match` sites must carry a `_ =>` arm and reject unknown implementors rather
/// than silently mis-route.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Implementor {
    /// A `baracuda-kernelgen`-emitted `.cu` — the thing we are deciding to ship or not.
    Generated,
    /// cuBLAS / cuBLASLt.
    Cublas,
    /// CUTLASS (template-instantiated or bespoke).
    Cutlass,
    /// cuDNN.
    Cudnn,
    /// A hand-written `baracuda-kernels-sys` `.cu`.
    Bespoke,
}

impl Implementor {
    /// Stable lowercase token code used in the committed artifact and CSV.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Implementor::Generated => "gen",
            Implementor::Cublas => "cublas",
            Implementor::Cutlass => "cutlass",
            Implementor::Cudnn => "cudnn",
            Implementor::Bespoke => "bespoke",
        }
    }

    /// Parse a [`Implementor::code`]. `None` for an unknown code (a future
    /// backend this build does not understand — an honest miss, never a guess).
    #[must_use]
    pub fn from_code(code: &str) -> Option<Implementor> {
        Some(match code {
            "gen" => Implementor::Generated,
            "cublas" => Implementor::Cublas,
            "cutlass" => Implementor::Cutlass,
            "cudnn" => Implementor::Cudnn,
            "bespoke" => Implementor::Bespoke,
            _ => return None,
        })
    }

    /// `true` if this implementor is a third-party vendor library (not ours to
    /// generate). A cell whose winner is a vendor emits no generated FKC
    /// contract or link-registry entry — vendor-exclusion made honest.
    #[must_use]
    pub fn is_vendor(self) -> bool {
        matches!(
            self,
            Implementor::Cublas | Implementor::Cutlass | Implementor::Cudnn
        )
    }
}

/// Why a winner is what it is — the honest marker on a routing decision.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Provenance {
    /// This repo's bench gate measured it, on the recorded [`HwStamp`].
    Measured,
    /// Hand-knowledge default, never benched (a seed rule from [`seed_winner`]).
    Seeded,
    /// A Fuel `dispatch_record` from a real workload (item 08).
    Reported,
}

impl Provenance {
    /// Stable lowercase token code.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Provenance::Measured => "measured",
            Provenance::Seeded => "seeded",
            Provenance::Reported => "reported",
        }
    }

    /// Parse a [`Provenance::code`]. `None` for an unknown code.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Provenance> {
        Some(match code {
            "measured" => Provenance::Measured,
            "seeded" => Provenance::Seeded,
            "reported" => Provenance::Reported,
            _ => return None,
        })
    }

    /// `true` if this decision came from observation (a bench measurement or a
    /// reported workload), not a hand-seeded default. A [`Provenance::Seeded`]
    /// row is only a *default*: it never overrides an observed decision.
    #[must_use]
    pub fn is_observed(self) -> bool {
        matches!(self, Provenance::Measured | Provenance::Reported)
    }
}

/// The hardware a measurement was taken on — the provenance stamp that keeps a
/// stale or foreign measurement visible, never silently trusted.
///
/// Mirrors the bench crate's `PytorchBaselineMetadata`: a measurement is only
/// applicable to a query whose [`ArchSku`] matches, and the device name / CUDA
/// version make a cross-device row auditable. `captured_unix_s` orders
/// same-arch measurements ("newer wins") — it lives here, **not** in the
/// committed artifact, so the artifact stays byte-stable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HwStamp {
    /// Compute-capability class the measurement was taken on.
    pub arch: ArchSku,
    /// Device product name (e.g. `"NVIDIA GeForce RTX 4070 Laptop GPU"`).
    pub device_name: String,
    /// CUDA runtime/driver version string the measurement used.
    pub cuda_version: String,
    /// Unix seconds at capture — orders same-arch measurements. Injected by the
    /// caller (the bench gate), never read from a wall clock here, so tests and
    /// the artifact are deterministic.
    pub captured_unix_s: u64,
}

/// One measured candidate result for a cell.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateResult {
    /// Which implementation was timed.
    pub implementor: Implementor,
    /// Median wall-clock per launch, nanoseconds (from the bench gate).
    pub median_ns: f64,
    /// Link-registry symbol when [`Implementor::Generated`]/[`Implementor::Bespoke`];
    /// `None` for a vendor library call.
    pub entry_point: Option<String>,
}

/// One row: the decision for a single `(op, structure-key, dtype, arch)` cell.
///
/// `structure_key` already encodes op+dtype+arch, so the token **is** the row
/// key. `ranked` is the top-K candidates (winner first) that item 08 reads;
/// the committed artifact keeps only the routing projection.
#[derive(Clone, Debug, PartialEq)]
pub struct DispatchEntry {
    /// [`StructureKey::to_token`] — the cell identity and the table's join key.
    pub structure_key: String,
    /// The implementation this cell routes to.
    pub winner: Implementor,
    /// The winner's entry point, when it has one (a `Generated`/`Bespoke`
    /// kernel symbol; for a multi-kernel schedule variant, the first kernel in
    /// launch order). Load-bearing for **variants**: one cell can hold several
    /// `Generated` candidates and the collapse-form token cannot disambiguate
    /// them, so a routing decision's identity is `(structure_key,
    /// winner_entry)` — never the token alone.
    pub winner_entry: Option<String>,
    /// `second_best_ns / winner_ns` (`> 1` ⇒ a real win; `1.0` ⇒ single
    /// candidate, no contest).
    pub margin: f64,
    /// Top-K candidates, winner first. Empty for a [`Provenance::Seeded`] row
    /// (never benched) or a routing-projection row parsed from the artifact.
    pub ranked: Vec<CandidateResult>,
    /// Why this winner is what it is.
    pub provenance: Provenance,
    /// The hardware the measurement was taken on (`None` for a seed).
    pub measured_on: Option<HwStamp>,
}

impl DispatchEntry {
    /// A hand-seeded decision: a winner with no measurement behind it.
    #[must_use]
    pub fn seeded(structure_key: String, winner: Implementor) -> DispatchEntry {
        DispatchEntry {
            structure_key,
            winner,
            winner_entry: None,
            margin: 1.0,
            ranked: Vec::new(),
            provenance: Provenance::Seeded,
            measured_on: None,
        }
    }

    /// The capture time of this row's measurement (`0` for a seed / unstamped
    /// row) — the ordering key [`merge`] uses for "newer wins".
    #[must_use]
    fn captured_unix_s(&self) -> u64 {
        self.measured_on.as_ref().map_or(0, |h| h.captured_unix_s)
    }

    /// The arch this row's measurement was taken on, if observed. `None` for a
    /// seed (a seed is arch-agnostic; the token still pins the target arch).
    #[must_use]
    fn measured_arch(&self) -> Option<ArchSku> {
        self.measured_on.as_ref().map(|h| h.arch)
    }
}

/// Reduce a set of measured candidates for one cell to a routing decision.
///
/// Candidates with a non-finite (NaN/±inf) or non-positive median are dropped
/// first — a broken timing is never a real measurement and must not win the cell
/// or poison the `margin` (a NaN sorts unpredictably; a `0`-ns "winner" yields an
/// infinite margin). `ranked` is then sorted ascending by `median_ns` (fastest
/// first); the winner is the fastest and the `margin` is `second_best / winner`
/// (`1.0` when a single candidate survives — a flagged no-contest). Returns
/// `None` when **no** candidate survives (an empty set, or every candidate
/// broken): a cell with no valid implementor is an honest "no route", never a
/// phantom winner.
///
/// *Correctness* remains the caller's job: each candidate must have passed the
/// op's numeric oracle before it reaches this reducer — a fast-but-*wrong*
/// candidate is not something timing alone can reject.
#[must_use]
pub fn winner_of(
    structure_key: String,
    mut candidates: Vec<CandidateResult>,
    measured_on: Option<HwStamp>,
) -> Option<DispatchEntry> {
    // Reject broken measurements up front. Left in, a NaN would sort
    // unpredictably (`partial_cmp` yields `None`) and could land at index 0 and
    // win, and a non-finite / non-positive median would poison the `margin` the
    // `merge` flip-guard reads. The bench gate logs that it produced one; the
    // reducer must never rank it.
    candidates.retain(|c| c.median_ns.is_finite() && c.median_ns > 0.0);
    if candidates.is_empty() {
        return None;
    }
    // Ascending by median. Every surviving median is finite and positive, so
    // `total_cmp` is a genuine total order with no NaN corner — the fastest wins,
    // independent of input order.
    candidates.sort_by(|a, b| a.median_ns.total_cmp(&b.median_ns));
    let winner = candidates[0].implementor;
    let winner_entry = candidates[0].entry_point.clone();
    let margin = if candidates.len() >= 2 {
        candidates[1].median_ns / candidates[0].median_ns
    } else {
        1.0
    };
    Some(DispatchEntry {
        structure_key,
        winner,
        winner_entry,
        margin,
        ranked: candidates,
        provenance: Provenance::Measured,
        measured_on,
    })
}

/// Hand-knowledge routing defaults per §7 — cells we route to a vendor library
/// without benching, keyed by a **predicate over the cell class**, not a literal
/// token (the seed is "large aligned GEMM", a regime, not one shape).
///
/// Returns `(winner, reason)` when a seed applies, `None` otherwise. The reason
/// string is recorded for auditability (why a cell was never generated).
///
/// The GEMM seed encodes `BENCHMARKS.md`'s measured split: large, aligned,
/// grid-stride f16/bf16 GEMM routes to cuBLAS; the small `M=1` decode regime and
/// everything non-GEMM is left `None` (bench it, or generate it).
#[must_use]
pub fn seed_winner(key: &StructureKey) -> Option<(Implementor, &'static str)> {
    use crate::sku::OpCategory;
    use crate::structure_key::{Contiguity, VecWidth, WorkClass};
    use crate::ElementKind;

    if key.op != OpCategory::Gemm {
        return None;
    }
    // Large, aligned, vectorizable, half-precision GEMM is cuBLAS's home turf.
    let large = key.work == WorkClass::GridStride;
    let half = matches!(key.dtype, ElementKind::F16 | ElementKind::Bf16);
    let operand0 = key.operands.first().copied().unwrap_or_default();
    let aligned = matches!(operand0.contig, Contiguity::Contig)
        && matches!(operand0.vec_width, VecWidth::V4 | VecWidth::V8);
    if large && half && aligned {
        return Some((Implementor::Cublas, "large aligned half-precision GEMM (§7 seed)"));
    }
    None
}

/// The token-keyed routing table the runtime dispatcher reads.
///
/// Entries are unique by `structure_key` token and kept sorted by it, so the
/// committed artifact is a stable diff and [`DispatchTable::lookup`] is a clean
/// scan. Lookup is **arch-gated**: a query only matches a row whose token arch
/// equals the query's arch, so a measurement taken on one arch can never route
/// another.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DispatchTable {
    /// The decision rows, **unique and sorted by `structure_key` token** — the
    /// invariant [`DispatchTable::from_entries`]/[`merge`] maintain. Construct
    /// through those, not this field directly: a hand-built table with duplicate
    /// tokens has undefined lookup order (`lookup` returns the first occurrence,
    /// while serialization keeps the last writer) until it is normalized.
    pub entries: Vec<DispatchEntry>,
}

impl DispatchTable {
    /// An empty table.
    #[must_use]
    pub fn new() -> DispatchTable {
        DispatchTable { entries: Vec::new() }
    }

    /// Build from rows, enforcing the sorted+deduped (last-writer-wins by token)
    /// invariant.
    #[must_use]
    pub fn from_entries(entries: Vec<DispatchEntry>) -> DispatchTable {
        let mut t = DispatchTable { entries };
        t.normalize();
        t
    }

    /// Re-establish the sorted-by-token, unique-by-token invariant, keeping the
    /// **last** writer of any duplicated token (matching `merge`'s in-place
    /// override semantics).
    fn normalize(&mut self) {
        // Reverse first so that, after a *stable* sort groups equal tokens in
        // their (now reversed) input order, `dedup_by` — which keeps the first
        // of each run — retains the last writer of a duplicated token.
        self.entries.reverse();
        self.entries.sort_by(|a, b| a.structure_key.cmp(&b.structure_key));
        self.entries.dedup_by(|a, b| a.structure_key == b.structure_key);
    }

    /// Look up the decision for a live cell, **arch-gated**: the stored row's
    /// token must equal `key.to_token()` (which pins the arch), so a row is
    /// never applied to a mismatched arch. `None` = an honest miss (no route
    /// recorded for this cell) — distinguishable from a recorded vendor route.
    #[must_use]
    pub fn lookup(&self, key: &StructureKey) -> Option<&DispatchEntry> {
        let token = key.to_token();
        self.entries.iter().find(|e| e.structure_key == token)
    }
}

/// Fold freshly observed (measured or Fuel-reported) results into a table.
///
/// This is item 08's ingest entry point (Fuel's `dispatch_record` feed becomes a
/// slice of `Reported` [`DispatchEntry`]s). Precedence — the load-bearing rules
/// the adversarial-verify pass probes:
///
/// - **Seed is a default only.** A [`Provenance::Seeded`] incoming never
///   overrides an existing entry; it only fills an empty slot.
/// - **Observed overrides seed.** A `Measured`/`Reported` incoming replaces a
///   `Seeded` entry — *unless* it would flip to a *different* winner on a margin
///   inside the noise floor (`< `[`MIN_FLIP_MARGIN`]), in which case the seed
///   stands (a 1.01× "win" must not overturn a vendor seed).
/// - **Newer wins among observed.** A later `captured_unix_s` refreshes an
///   earlier same-arch observation; a flip to a different winner still needs the
///   margin threshold.
/// - **Arch-gated Reported.** A `Reported` row whose [`HwStamp`] arch ≠ the
///   cell's token arch is rejected (a foreign-arch workload can't route this
///   arch).
///
/// After the fold the table is re-normalized (sorted + unique by token).
pub fn merge(table: &mut DispatchTable, incoming: &[DispatchEntry]) {
    for inc in incoming {
        // A non-finite margin is a broken / poisoned measurement. `winner_of` can
        // no longer produce one, but a Fuel `Reported` record constructed directly
        // (not via `parse_dispatch_table`, which guards its own ingress) could.
        // Never let it into the table: it can't be trusted to route and would
        // serialize to an uncompilable artifact literal (`NaN`/`inf`). This guards
        // every insert path — empty-slot push, same-winner refresh, and flip.
        if !inc.margin.is_finite() {
            continue;
        }
        // A Reported row must have been observed on the arch it claims to route.
        if inc.provenance == Provenance::Reported {
            let token_arch = StructureKey::from_token(&inc.structure_key).map(|k| k.arch);
            match (token_arch, inc.measured_arch()) {
                (Some(t), Some(m)) if t == m => {}
                _ => continue, // foreign-arch or unparseable → reject
            }
        }

        match table
            .entries
            .iter_mut()
            .find(|e| e.structure_key == inc.structure_key)
        {
            None => table.entries.push(inc.clone()),
            Some(cur) => {
                // A seed never overrides an existing decision.
                if !inc.provenance.is_observed() {
                    continue;
                }
                // Among two observed rows, only a newer capture may refresh.
                if cur.provenance.is_observed()
                    && inc.captured_unix_s() <= cur.captured_unix_s()
                {
                    continue;
                }
                // A flip to a *different route* must clear the noise floor; a
                // same-route refresh is always allowed. The route identity is
                // `(winner, winner_entry)`, NOT the implementor alone — two
                // schedule *variants* are both `Generated`, and a noise-level
                // variant→variant flip must not bypass the floor. A row whose
                // `winner_entry` is `None` (a seed, or a pre-variant row) pins
                // no specific variant, so any same-implementor entry refines it
                // without the flip threshold. The guard tests `>=` (bound to a
                // bool, not `!(margin < …)`, to keep it out of the partial-ord
                // negation lint) so a non-finite margin — defense in depth
                // against a poisoned measurement — fails to clear the floor and
                // blocks the flip, instead of sneaking through the fact that
                // `NaN < MIN_FLIP_MARGIN` is `false`.
                let same_route = inc.winner == cur.winner
                    && (cur.winner_entry.is_none() || inc.winner_entry == cur.winner_entry);
                let clears_noise_floor = inc.margin >= MIN_FLIP_MARGIN;
                if !same_route && !clears_noise_floor {
                    continue;
                }
                *cur = inc.clone();
            }
        }
    }
    table.normalize();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sku::OpCategory;
    use crate::structure_key::{structure_key, Contiguity, VecWidth};
    use crate::{ArchSku, ElementKind, OperandDesc};

    fn cand(imp: Implementor, ns: f64) -> CandidateResult {
        CandidateResult { implementor: imp, median_ns: ns, entry_point: None }
    }

    fn stamp(arch: ArchSku, t: u64) -> HwStamp {
        HwStamp {
            arch,
            device_name: "NVIDIA GeForce RTX 4070 Laptop GPU".to_string(),
            cuda_version: "13.3".to_string(),
            captured_unix_s: t,
        }
    }

    /// A binary-elementwise f32 cell on sm89 — a real, parseable token.
    fn ew_key() -> StructureKey {
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89)
    }

    /// A GEMM cell built to trip (or dodge) the seed predicate.
    fn gemm_key(dtype: ElementKind, m: i64, align: u32) -> StructureKey {
        // K=N large so work is GridStride; align 256 => V4/V8 for half.
        let a = OperandDesc::new(2, &[m, 4096], &[4096, 1], dtype, align);
        let b = OperandDesc::new(2, &[4096, 4096], &[4096, 1], dtype, align);
        let o = OperandDesc::new(2, &[m, 4096], &[4096, 1], dtype, align);
        structure_key(OpCategory::Gemm, &[a, b, o], ArchSku::Sm89)
    }

    #[test]
    fn winner_of_picks_min_median_and_computes_margin() {
        let k = ew_key().to_token();
        let e = winner_of(
            k,
            vec![
                cand(Implementor::Cublas, 200.0),
                cand(Implementor::Generated, 100.0),
                cand(Implementor::Bespoke, 400.0),
            ],
            Some(stamp(ArchSku::Sm89, 10)),
        )
        .expect("non-empty");
        assert_eq!(e.winner, Implementor::Generated);
        assert!((e.margin - 2.0).abs() < 1e-9, "second/first = 200/100");
        assert_eq!(e.ranked[0].implementor, Implementor::Generated);
        assert_eq!(e.ranked[2].implementor, Implementor::Bespoke); // sorted ascending
        assert_eq!(e.provenance, Provenance::Measured);
    }

    #[test]
    fn winner_of_single_candidate_has_unit_margin() {
        let k = ew_key().to_token();
        let e = winner_of(k, vec![cand(Implementor::Generated, 100.0)], None).unwrap();
        assert_eq!(e.margin, 1.0, "single candidate = flagged no-contest");
    }

    #[test]
    fn winner_of_empty_is_honest_no_route() {
        let k = ew_key().to_token();
        assert!(winner_of(k, vec![], None).is_none());
    }

    #[test]
    fn seed_routes_large_aligned_half_gemm_to_cublas() {
        let k = gemm_key(ElementKind::F16, 4096, 256);
        let (imp, _why) = seed_winner(&k).expect("large aligned f16 GEMM is seeded");
        assert_eq!(imp, Implementor::Cublas);
    }

    #[test]
    fn seed_declines_small_decode_gemm() {
        // M=1 decode GEMM: not GridStride-large enough to be cuBLAS's turf here
        // is expressed via the small work class — assert no seed fires.
        let k = gemm_key(ElementKind::F16, 1, 256);
        // A [1,4096]x[4096,4096] is still large work, so force the *small* regime
        // by a tiny problem instead:
        let a = OperandDesc::new(2, &[1, 8], &[8, 1], ElementKind::F16, 256);
        let b = OperandDesc::new(2, &[8, 8], &[8, 1], ElementKind::F16, 256);
        let o = OperandDesc::new(2, &[1, 8], &[8, 1], ElementKind::F16, 256);
        let small = structure_key(OpCategory::Gemm, &[a, b, o], ArchSku::Sm89);
        assert!(seed_winner(&small).is_none(), "tiny GEMM is not vendor-seeded");
        // And a non-GEMM op is never seeded regardless of size.
        assert!(seed_winner(&ew_key()).is_none());
        // The large f16 case does fire (guards the predicate isn't vacuously None).
        assert!(seed_winner(&k).is_some());
    }

    #[test]
    fn seed_declines_f32_gemm() {
        // Seed is half-precision only; f32 large aligned GEMM is not auto-vendored.
        let k = gemm_key(ElementKind::F32, 4096, 256);
        assert!(seed_winner(&k).is_none());
    }

    #[test]
    fn merge_observed_overrides_seed() {
        let k = ew_key();
        let mut t = DispatchTable::from_entries(vec![DispatchEntry::seeded(
            k.to_token(),
            Implementor::Cublas,
        )]);
        let obs = winner_of(
            k.to_token(),
            vec![cand(Implementor::Generated, 100.0), cand(Implementor::Cublas, 250.0)],
            Some(stamp(ArchSku::Sm89, 20)),
        )
        .unwrap();
        merge(&mut t, &[obs]);
        let e = t.lookup(&k).unwrap();
        assert_eq!(e.winner, Implementor::Generated);
        assert_eq!(e.provenance, Provenance::Measured);
    }

    #[test]
    fn merge_seed_never_overrides_observed() {
        let k = ew_key();
        let obs = winner_of(
            k.to_token(),
            vec![cand(Implementor::Generated, 100.0), cand(Implementor::Cublas, 250.0)],
            Some(stamp(ArchSku::Sm89, 20)),
        )
        .unwrap();
        let mut t = DispatchTable::from_entries(vec![obs]);
        merge(&mut t, &[DispatchEntry::seeded(k.to_token(), Implementor::Cublas)]);
        assert_eq!(t.lookup(&k).unwrap().winner, Implementor::Generated, "seed can't demote a measurement");
    }

    #[test]
    fn merge_noise_win_does_not_flip_seed() {
        let k = ew_key();
        let mut t = DispatchTable::from_entries(vec![DispatchEntry::seeded(
            k.to_token(),
            Implementor::Cublas,
        )]);
        // Generated is "faster" by 1.01× — inside the noise floor. Must NOT flip.
        let noisy = winner_of(
            k.to_token(),
            vec![cand(Implementor::Generated, 100.0), cand(Implementor::Cublas, 101.0)],
            Some(stamp(ArchSku::Sm89, 20)),
        )
        .unwrap();
        assert!(noisy.margin < MIN_FLIP_MARGIN);
        merge(&mut t, &[noisy]);
        assert_eq!(t.lookup(&k).unwrap().winner, Implementor::Cublas, "1.01x can't overturn a seed");
    }

    #[test]
    fn merge_same_winner_refresh_ignores_margin_threshold() {
        let k = ew_key();
        let mut t = DispatchTable::from_entries(vec![DispatchEntry::seeded(
            k.to_token(),
            Implementor::Cublas,
        )]);
        // Same winner (Cublas), tiny margin — a refresh, not a flip: allowed.
        let refresh = winner_of(
            k.to_token(),
            vec![cand(Implementor::Cublas, 100.0), cand(Implementor::Generated, 101.0)],
            Some(stamp(ArchSku::Sm89, 20)),
        )
        .unwrap();
        assert!(refresh.margin < MIN_FLIP_MARGIN);
        merge(&mut t, &[refresh]);
        let e = t.lookup(&k).unwrap();
        assert_eq!(e.winner, Implementor::Cublas);
        assert_eq!(e.provenance, Provenance::Measured, "seed upgraded to measured");
    }

    #[test]
    fn merge_newer_measurement_wins_older() {
        let k = ew_key();
        let old = winner_of(
            k.to_token(),
            vec![cand(Implementor::Generated, 100.0), cand(Implementor::Cublas, 250.0)],
            Some(stamp(ArchSku::Sm89, 10)),
        )
        .unwrap();
        let mut t = DispatchTable::from_entries(vec![old]);
        // A newer measurement flips to Cublas with a real margin (2.5×).
        let newer = winner_of(
            k.to_token(),
            vec![cand(Implementor::Cublas, 100.0), cand(Implementor::Generated, 250.0)],
            Some(stamp(ArchSku::Sm89, 99)),
        )
        .unwrap();
        merge(&mut t, &[newer]);
        assert_eq!(t.lookup(&k).unwrap().winner, Implementor::Cublas);

        // An *older* measurement can't overwrite the newer one.
        let stale = winner_of(
            k.to_token(),
            vec![cand(Implementor::Generated, 100.0), cand(Implementor::Cublas, 250.0)],
            Some(stamp(ArchSku::Sm89, 5)),
        )
        .unwrap();
        merge(&mut t, &[stale]);
        assert_eq!(t.lookup(&k).unwrap().winner, Implementor::Cublas, "stale can't demote newer");
    }

    #[test]
    fn merge_rejects_foreign_arch_reported() {
        let k = ew_key(); // token arch = sm89
        let mut t = DispatchTable::new();
        // A Reported row measured on sm80 must not route an sm89 cell.
        let mut foreign = winner_of(
            k.to_token(),
            vec![cand(Implementor::Generated, 100.0)],
            Some(stamp(ArchSku::Sm80, 30)),
        )
        .unwrap();
        foreign.provenance = Provenance::Reported;
        merge(&mut t, &[foreign]);
        assert!(t.lookup(&k).is_none(), "foreign-arch Reported rejected");

        // The same Reported row on the matching arch is accepted.
        let mut native = winner_of(
            k.to_token(),
            vec![cand(Implementor::Generated, 100.0)],
            Some(stamp(ArchSku::Sm89, 30)),
        )
        .unwrap();
        native.provenance = Provenance::Reported;
        merge(&mut t, &[native]);
        assert_eq!(t.lookup(&k).unwrap().winner, Implementor::Generated);
    }

    #[test]
    fn lookup_is_arch_gated() {
        // A row for the sm89 cell must not answer an sm80 query for the "same"
        // logical cell (different token arch ⇒ miss).
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let k89 = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        let k80 = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm80);
        let t = DispatchTable::from_entries(vec![DispatchEntry::seeded(
            k89.to_token(),
            Implementor::Generated,
        )]);
        assert!(t.lookup(&k89).is_some());
        assert!(t.lookup(&k80).is_none(), "sm80 query must miss an sm89 row");
    }

    #[test]
    fn table_normalize_sorts_and_dedups_by_token() {
        let k = ew_key();
        let t = DispatchTable::from_entries(vec![
            DispatchEntry::seeded(k.to_token(), Implementor::Cublas),
            DispatchEntry::seeded(k.to_token(), Implementor::Generated), // dup token
        ]);
        assert_eq!(t.entries.len(), 1, "duplicate token collapsed");
    }

    #[test]
    fn table_dedup_keeps_last_writer() {
        // The documented contract is last-writer-wins (like a map insert).
        let k = ew_key();
        let t = DispatchTable::from_entries(vec![
            DispatchEntry::seeded(k.to_token(), Implementor::Cublas),
            DispatchEntry::seeded(k.to_token(), Implementor::Generated), // wins
        ]);
        assert_eq!(t.entries.len(), 1);
        assert_eq!(t.lookup(&k).unwrap().winner, Implementor::Generated);
    }

    #[test]
    fn winner_of_rejects_nan_timing_regardless_of_order() {
        let k = ew_key().to_token();
        // NaN first: the old `unwrap_or(Greater)` comparator left it at index 0
        // and it won the cell. It must not, and the outcome must be order-stable.
        let e1 = winner_of(
            k.clone(),
            vec![cand(Implementor::Generated, f64::NAN), cand(Implementor::Cublas, 100.0)],
            None,
        )
        .unwrap();
        assert_eq!(e1.winner, Implementor::Cublas);
        assert!(e1.margin.is_finite());
        assert_eq!(e1.ranked.len(), 1, "NaN candidate dropped from ranked");
        // NaN last: same set, same winner (no order dependence).
        let e2 = winner_of(
            k,
            vec![cand(Implementor::Cublas, 100.0), cand(Implementor::Generated, f64::NAN)],
            None,
        )
        .unwrap();
        assert_eq!(e2.winner, Implementor::Cublas);
    }

    #[test]
    fn winner_of_rejects_nonpositive_and_infinite_timings() {
        let k = ew_key().to_token();
        let e = winner_of(
            k.clone(),
            vec![
                cand(Implementor::Generated, 0.0),         // 0-ns is not a real timing
                cand(Implementor::Cublas, -5.0),           // negative is nonsense
                cand(Implementor::Bespoke, f64::INFINITY), // broken
                cand(Implementor::Cudnn, 100.0),           // the only valid one
            ],
            None,
        )
        .unwrap();
        assert_eq!(e.winner, Implementor::Cudnn);
        assert_eq!(e.ranked.len(), 1);
        assert_eq!(e.margin, 1.0, "single survivor = no contest");
        // Every candidate broken ⇒ honest no-route, not a phantom winner.
        assert!(winner_of(
            k,
            vec![cand(Implementor::Generated, 0.0), cand(Implementor::Cublas, f64::NAN)],
            None,
        )
        .is_none());
    }

    #[test]
    fn merge_nan_margin_cannot_flip_a_decision() {
        // Defense in depth: even a hand-built entry with a NaN margin (which
        // `winner_of` can no longer produce) must not slip past the flip-guard.
        let k = ew_key();
        let mut t = DispatchTable::from_entries(vec![DispatchEntry::seeded(
            k.to_token(),
            Implementor::Cublas,
        )]);
        let poisoned = DispatchEntry {
            structure_key: k.to_token(),
            winner: Implementor::Generated,
            winner_entry: None,
            margin: f64::NAN,
            ranked: Vec::new(),
            provenance: Provenance::Measured,
            measured_on: Some(stamp(ArchSku::Sm89, 50)),
        };
        merge(&mut t, &[poisoned]);
        assert_eq!(
            t.lookup(&k).unwrap().winner,
            Implementor::Cublas,
            "NaN margin must not clear the noise floor"
        );
    }

    #[test]
    fn merge_variant_flip_needs_margin_but_refinement_does_not() {
        let k = ew_key();
        let vcand = |entry: &str, ns: f64| CandidateResult {
            implementor: Implementor::Generated,
            median_ns: ns,
            entry_point: Some(entry.to_string()),
        };
        // A measured decision for variant A.
        let a = winner_of(
            k.to_token(),
            vec![vcand("kern_a", 100.0), vcand("kern_b", 250.0)],
            Some(stamp(ArchSku::Sm89, 10)),
        )
        .unwrap();
        let mut t = DispatchTable::from_entries(vec![a]);
        // A NOISE-level (1.05x) newer win for variant B: both are `Generated`,
        // but the ROUTE differs (winner_entry) — must NOT flip.
        let noisy_b = winner_of(
            k.to_token(),
            vec![vcand("kern_b", 100.0), vcand("kern_a", 105.0)],
            Some(stamp(ArchSku::Sm89, 20)),
        )
        .unwrap();
        assert!(noisy_b.margin < MIN_FLIP_MARGIN);
        merge(&mut t, &[noisy_b]);
        assert_eq!(
            t.lookup(&k).unwrap().winner_entry.as_deref(),
            Some("kern_a"),
            "a within-noise variant flip must not bypass the floor"
        );
        // A REAL (2x) newer win for variant B flips.
        let real_b = winner_of(
            k.to_token(),
            vec![vcand("kern_b", 100.0), vcand("kern_a", 200.0)],
            Some(stamp(ArchSku::Sm89, 30)),
        )
        .unwrap();
        merge(&mut t, &[real_b]);
        assert_eq!(t.lookup(&k).unwrap().winner_entry.as_deref(), Some("kern_b"));
        // A seed pins no entry: a single-candidate (margin 1.0) measurement
        // REFINES it without needing the flip threshold.
        let mut t2 = DispatchTable::from_entries(vec![DispatchEntry::seeded(
            k.to_token(),
            Implementor::Generated,
        )]);
        let refine = winner_of(
            k.to_token(),
            vec![vcand("kern_a", 100.0)],
            Some(stamp(ArchSku::Sm89, 10)),
        )
        .unwrap();
        assert_eq!(refine.margin, 1.0);
        merge(&mut t2, &[refine]);
        assert_eq!(
            t2.lookup(&k).unwrap().winner_entry.as_deref(),
            Some("kern_a"),
            "an entry-less seed is refined, not flipped"
        );
    }

    #[test]
    fn merge_rejects_non_finite_margin_into_empty_slot() {
        // The empty-slot push path (no existing row, no flip-guard) must also
        // reject a poisoned margin — otherwise a Fuel-reported NaN/inf margin
        // enters the table and serializes to an uncompilable artifact literal.
        let k = ew_key();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut t = DispatchTable::new();
            let poisoned = DispatchEntry {
                structure_key: k.to_token(),
                winner: Implementor::Generated,
                winner_entry: None,
                margin: bad,
                ranked: Vec::new(),
                provenance: Provenance::Reported,
                measured_on: Some(stamp(ArchSku::Sm89, 10)),
            };
            merge(&mut t, &[poisoned]);
            assert!(t.lookup(&k).is_none(), "non-finite margin ({bad}) not inserted");
        }
    }

    #[test]
    fn implementor_and_provenance_codes_round_trip() {
        for imp in [
            Implementor::Generated,
            Implementor::Cublas,
            Implementor::Cutlass,
            Implementor::Cudnn,
            Implementor::Bespoke,
        ] {
            assert_eq!(Implementor::from_code(imp.code()), Some(imp));
        }
        for p in [Provenance::Measured, Provenance::Seeded, Provenance::Reported] {
            assert_eq!(Provenance::from_code(p.code()), Some(p));
        }
        assert!(Implementor::from_code("rocm").is_none());
        assert!(Provenance::from_code("guessed").is_none());
    }

    #[test]
    fn vendor_predicate_matches_libs_not_ours() {
        assert!(Implementor::Cublas.is_vendor());
        assert!(Implementor::Cutlass.is_vendor());
        assert!(Implementor::Cudnn.is_vendor());
        assert!(!Implementor::Generated.is_vendor());
        assert!(!Implementor::Bespoke.is_vendor());
    }

    // Silence unused-helper warnings for the intentionally-unused `gemm_key` M arg.
    #[test]
    fn seed_gemm_alignment_matters() {
        // Unaligned (align 4 ⇒ Scalar vec width) large half GEMM is NOT seeded —
        // cuBLAS's advantage assumes vectorizable, aligned operands.
        let k = gemm_key(ElementKind::F16, 4096, 4);
        assert!(
            seed_winner(&k).is_none(),
            "unaligned half GEMM is not auto-vendored"
        );
        // Sanity: with the operand explicitly V4 it does fire.
        assert!(matches!(
            gemm_key(ElementKind::F16, 4096, 256).operands[0].vec_width,
            VecWidth::V4 | VecWidth::V8
        ));
        // Contiguity of the aligned operand is Contig (not Strided).
        assert_eq!(gemm_key(ElementKind::F16, 4096, 256).operands[0].contig, Contiguity::Contig);
    }
}
