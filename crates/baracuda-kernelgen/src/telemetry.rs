//! Telemetry ingest (item 08 — Baracuda half of telemetry-driven variant
//! selection).
//!
//! Fuel emits per-cell **dispatch** and **miss** records as JSON Lines. This
//! module reads that feed, ranks cells by demand, and folds the reported wins
//! into a [`DispatchTable`] through the ALREADY-FROZEN `dispatch.rs` seams
//! ([`reported_entry`] + [`merge`]) — it never re-implements the merge policy.
//! Everything here is pure-CPU and unit-testable off-device; the live feed is
//! Fuel-blocked, but nothing below waits on it.
//!
//! # Wire shape (the ratified 2026-07-03 FINAL, not the 08-doc §5.2 sketch)
//!
//! The record types match `docs/fuel-reply-dispatch-records-variants-2026-07-03.md`:
//! `schema` is `u32`; the chosen/fallback implementation is a nested [`ImplId`]
//! object (five **separable** fields — never concatenate or hash them);
//! candidates carry a per-entry `latency_ns`; `count` is the aggregation count
//! (the rank key — `est_speedup_if_available` is NOT on the wire); and the
//! envelope carries an optional hardware fingerprint
//! (`compute_capability`/`hardware_sku`/`driver_version`).
//!
//! # JSON parsing
//!
//! The wire is **nested** (the 08-doc's "flat one-object-per-line" is wrong), so
//! this module carries a small, self-contained recursive JSON-value scanner
//! rather than pulling `serde`/`serde_json` into the published crate's dependency
//! surface. It is **panic-free**: every malformed line increments a skip counter
//! and is dropped, never panics.

use std::io::BufRead;

use baracuda_kernel_vocab::{
    ArchSku, DispatchTable, HwStamp, Implementor, ReportedCandidate, STRUCTURE_KEY_VERSION,
    StructureKey, merge, reported_entry,
};

/// Highest telemetry-envelope `schema` this build understands. A record with a
/// larger `schema` is a **future** envelope and is skipped (forward-compat).
///
/// The FINAL wire doc widened `schema` to `u32` and bumped it to cover the
/// hardware fingerprint, but does not pin the exact integer; `1` is taken here as
/// "the current fingerprint-bearing envelope this build reads". Adjust in lockstep
/// with Fuel when the envelope grows again.
pub const TELEMETRY_SCHEMA_MAX: u32 = 1;

// ===========================================================================
// Wire types (ratified 2026-07-03)
// ===========================================================================

/// A Fuel implementation identity — the join target of a dispatch/miss record.
///
/// The five fields are **separable** and stay separate: they are NEVER
/// concatenated or hashed into one opaque id (the identity is the tuple). Only
/// `backend` drives Baracuda's [`Implementor`] resolution; the rest are carried
/// verbatim for provenance and variant voting.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImplId {
    /// Backend code (e.g. `"gen"`, `"cublas"`) — resolves to an [`Implementor`].
    pub backend: String,
    /// Op name the implementation serves.
    pub op: String,
    /// Operand dtype codes.
    pub dtypes: Vec<String>,
    /// Kernel source identity — the entry-point symbol for a generated/bespoke
    /// kernel (mirrors the link-registry `entry_point`).
    pub kernel_source: String,
    /// Source-revision hash (the per-variant discriminator [`variant_votes`]
    /// tallies on).
    pub kernel_revision_hash: String,
}

/// One considered candidate in a dispatch record, with its (possibly missing)
/// measured latency. `latency_ns: None` = "considered, unmeasured" — Fuel's
/// Judge ranked it by static cost; per the FINAL wire, sparse lists are the norm.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// Which implementation this candidate is.
    pub impl_id: ImplId,
    /// Measured latency in nanoseconds, or `None` when unmeasured.
    pub latency_ns: Option<u64>,
}

/// The device a record was captured on. `compute_capability: None` (non-CUDA
/// backend) is exactly the record [`merge_reports`] drops, "stampless ⇒ dropped,
/// not guessed".
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HwFingerprint {
    /// Compute capability `(major, minor)`, or `None` on non-CUDA backends.
    pub compute_capability: Option<(u32, u32)>,
    /// Device product name (the `HwStamp` device name).
    pub hardware_sku: Option<String>,
    /// Driver version string (the `HwStamp` CUDA version).
    pub driver_version: Option<String>,
}

/// One aggregated Fuel dispatch record: for a cell, the chosen implementation,
/// the considered candidates, an aggregation `count`, and the capture hardware.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchRecord {
    /// Telemetry envelope schema version.
    pub schema: u32,
    /// The cell's [`StructureKey::to_token`] string, or `None` (Fuel's field is
    /// optional). A `None`-token record is dropped by [`merge_reports`].
    pub structure_key: Option<String>,
    /// The implementation Fuel chose (honored verbatim, never re-ranked).
    pub chosen: ImplId,
    /// The considered candidates (per-candidate latencies; may be sparse).
    pub candidates: Vec<Candidate>,
    /// Aggregation count — the rank key.
    pub count: u64,
    /// Capture hardware fingerprint, when present.
    pub hw: Option<HwFingerprint>,
}

/// One aggregated Fuel miss record: a cell that was requested (`wanted`) but
/// served by a `fallback` (best-admissible-match-is-generic). Drives demand
/// ranking — misses flow first (no Judge dependency).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissRecord {
    /// Telemetry envelope schema version.
    pub schema: u32,
    /// The requested cell's token (required — a miss with no cell is dropped).
    pub wanted: String,
    /// The implementation that served the miss.
    pub fallback: ImplId,
    /// Aggregation count — the demand-rank key.
    pub count: u64,
    /// Capture hardware fingerprint, when present.
    pub hw: Option<HwFingerprint>,
}

/// The parsed result of an ingest pass: the surviving records plus a count of
/// lines dropped by the validation ladder (malformed JSON, wrong schema,
/// unparseable / wrong-version token, …).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ingest {
    /// Surviving dispatch records.
    pub dispatches: Vec<DispatchRecord>,
    /// Surviving miss records.
    pub misses: Vec<MissRecord>,
    /// Number of non-blank lines dropped (skip-with-count, never a panic).
    pub skipped: usize,
}

// ===========================================================================
// A self-contained recursive JSON-value scanner (no serde on the wire)
// ===========================================================================

/// A parsed JSON value. Numbers are kept as their raw literal so integer fields
/// (`u64` counts, `u32` capabilities) parse without `f64` precision loss.
enum Json {
    Null,
    Bool,
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Field lookup on an object value (`None` for a non-object or a missing key).
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Num(s) => s.parse().ok(),
            _ => None,
        }
    }
    fn as_u32(&self) -> Option<u32> {
        match self {
            Json::Num(s) => s.parse().ok(),
            _ => None,
        }
    }
    fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
}

/// Maximum JSON nesting depth the scanner will descend before returning `None`
/// (a counted skip). WITHOUT this cap, a single deeply-nested line (e.g. a run of
/// `[`) drives the `value → array/object → value` recursion one native stack frame
/// per level and OVERFLOWS the stack — an uncatchable process abort that bypasses
/// the skip counter, breaking the panic-free/TOTAL contract. The ratified wire
/// nests ~6 deep (record → candidates[] → Candidate → impl_id → dtypes[] → string),
/// so 128 is far above any legitimate telemetry and far below the overflow
/// threshold.
const MAX_JSON_DEPTH: usize = 128;

/// Recursive-descent JSON scanner over a char buffer. Every method returns
/// `Option`; nothing panics, and nesting past [`MAX_JSON_DEPTH`] is a `None`
/// (counted skip), never a stack overflow.
struct Scanner {
    chars: Vec<char>,
    pos: usize,
}

impl Scanner {
    /// Parse exactly one JSON value with no trailing garbage. The trailing-garbage
    /// rejection is what makes two concatenated objects on one line a skip, not a
    /// silently-accepted first object.
    fn parse(s: &str) -> Option<Json> {
        let mut sc = Scanner {
            chars: s.chars().collect(),
            pos: 0,
        };
        sc.skip_ws();
        let v = sc.value(0)?;
        sc.skip_ws();
        if sc.pos == sc.chars.len() {
            Some(v)
        } else {
            None
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t' | '\n' | '\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, c: char) -> Option<()> {
        (self.bump()? == c).then_some(())
    }

    /// `depth` is the current nesting level (0 at the top). Recursion past
    /// [`MAX_JSON_DEPTH`] returns `None` (a counted skip) rather than overflowing
    /// the native stack. `depth` is a call-stack-scoped parameter (not a mutable
    /// field), so a WIDE-but-shallow array like `[[a],[b],[c]]` does not
    /// accumulate depth across siblings.
    fn value(&mut self, depth: usize) -> Option<Json> {
        if depth > MAX_JSON_DEPTH {
            return None;
        }
        match self.peek()? {
            '{' => self.object(depth),
            '[' => self.array(depth),
            '"' => self.string().map(Json::Str),
            't' | 'f' => self.boolean(),
            'n' => self.null(),
            '-' | '0'..='9' => self.number(),
            _ => None,
        }
    }

    fn object(&mut self, depth: usize) -> Option<Json> {
        self.expect('{')?;
        let mut fields = Vec::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.pos += 1;
            return Some(Json::Obj(fields));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(':')?;
            self.skip_ws();
            let val = self.value(depth + 1)?;
            fields.push((key, val));
            self.skip_ws();
            match self.bump()? {
                ',' => {}
                '}' => break,
                _ => return None,
            }
        }
        Some(Json::Obj(fields))
    }

    fn array(&mut self, depth: usize) -> Option<Json> {
        self.expect('[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.pos += 1;
            return Some(Json::Arr(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.bump()? {
                ',' => {}
                ']' => break,
                _ => return None,
            }
        }
        Some(Json::Arr(items))
    }

    fn string(&mut self) -> Option<String> {
        self.expect('"')?;
        let mut out = String::new();
        loop {
            match self.bump()? {
                '"' => return Some(out),
                '\\' => match self.bump()? {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    'r' => out.push('\r'),
                    'b' => out.push('\u{8}'),
                    'f' => out.push('\u{c}'),
                    'u' => {
                        let mut code: u32 = 0;
                        for _ in 0..4 {
                            code = code * 16 + self.bump()?.to_digit(16)?;
                        }
                        // Lone code unit only (no surrogate-pair join): a
                        // surrogate yields None → the line is skipped. Wire tokens
                        // are ASCII in practice, so this is a documented corner.
                        out.push(char::from_u32(code)?);
                    }
                    _ => return None,
                },
                c => out.push(c),
            }
        }
    }

    fn number(&mut self) -> Option<Json> {
        let start = self.pos;
        while matches!(self.peek(), Some('0'..='9' | '-' | '+' | '.' | 'e' | 'E')) {
            self.pos += 1;
        }
        if self.pos == start {
            return None;
        }
        Some(Json::Num(self.chars[start..self.pos].iter().collect()))
    }

    fn literal(&mut self, word: &str) -> Option<()> {
        for want in word.chars() {
            if self.bump()? != want {
                return None;
            }
        }
        Some(())
    }

    fn boolean(&mut self) -> Option<Json> {
        match self.peek()? {
            't' => self.literal("true").map(|()| Json::Bool),
            _ => self.literal("false").map(|()| Json::Bool),
        }
    }

    fn null(&mut self) -> Option<Json> {
        self.literal("null").map(|()| Json::Null)
    }
}

// ===========================================================================
// Ingest — the validation ladder
// ===========================================================================

/// One parsed record, before it is filed into an [`Ingest`].
enum Record {
    Dispatch(DispatchRecord),
    Miss(MissRecord),
}

/// Read a JSONL telemetry feed into an [`Ingest`]. **Panic-free**: one JSON
/// object per line, and every failure increments `skipped` (blank lines are
/// ignored, not counted).
///
/// The validation ladder per line: (1) the line parses as one JSON object with no
/// trailing garbage; (2) `schema ≤ `[`TELEMETRY_SCHEMA_MAX`]; (3) the structure
/// token feeds [`StructureKey::from_token`] and its `key.version` **explicitly**
/// equals [`STRUCTURE_KEY_VERSION`] (`from_token` alone does NOT gate the version,
/// so a future `sk2|…` token that otherwise parses is still skipped); (4) unknown
/// JSON fields are ignored (forward-compat). A `wanted` field routes to a miss, a
/// `chosen` field to a dispatch.
#[must_use]
pub fn ingest_jsonl(reader: impl BufRead) -> Ingest {
    let mut ingest = Ingest::default();
    for line in reader.lines() {
        let Ok(line) = line else {
            ingest.skipped += 1;
            continue;
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue; // blank separator line — not a record, not a skip
        }
        match parse_record(trimmed) {
            Some(Record::Dispatch(d)) => ingest.dispatches.push(d),
            Some(Record::Miss(m)) => ingest.misses.push(m),
            None => ingest.skipped += 1,
        }
    }
    ingest
}

fn parse_record(line: &str) -> Option<Record> {
    let json = Scanner::parse(line)?;
    if !matches!(json, Json::Obj(_)) {
        return None;
    }
    let schema = json.get("schema")?.as_u32()?;
    if schema > TELEMETRY_SCHEMA_MAX {
        return None;
    }
    if json.get("wanted").is_some() {
        parse_miss(&json, schema).map(Record::Miss)
    } else if json.get("chosen").is_some() {
        parse_dispatch(&json, schema).map(Record::Dispatch)
    } else {
        None
    }
}

/// Validate a token: it must parse AND be the current structure-key version.
/// `from_token` does not gate the version, so the explicit check here is what
/// skips a future `sk2|…` token.
fn valid_token(tok: &str) -> Option<String> {
    let key = StructureKey::from_token(tok)?;
    (key.version == STRUCTURE_KEY_VERSION).then(|| tok.to_string())
}

fn parse_dispatch(json: &Json, schema: u32) -> Option<DispatchRecord> {
    let structure_key = match json.get("structure_key") {
        None | Some(Json::Null) => None,
        Some(v) => Some(valid_token(v.as_str()?)?),
    };
    let chosen = parse_impl_id(json.get("chosen")?)?;
    let candidates = match json.get("candidates") {
        None | Some(Json::Null) => Vec::new(),
        Some(v) => parse_candidates(v)?,
    };
    let count = json.get("count").and_then(Json::as_u64).unwrap_or(0);
    let hw = json.get("hw").and_then(parse_hw);
    Some(DispatchRecord {
        schema,
        structure_key,
        chosen,
        candidates,
        count,
        hw,
    })
}

fn parse_miss(json: &Json, schema: u32) -> Option<MissRecord> {
    let wanted = valid_token(json.get("wanted")?.as_str()?)?;
    let fallback = parse_impl_id(json.get("fallback")?)?;
    let count = json.get("count").and_then(Json::as_u64).unwrap_or(0);
    let hw = json.get("hw").and_then(parse_hw);
    Some(MissRecord {
        schema,
        wanted,
        fallback,
        count,
        hw,
    })
}

fn parse_impl_id(v: &Json) -> Option<ImplId> {
    if !matches!(v, Json::Obj(_)) {
        return None;
    }
    let field = |k: &str| v.get(k).and_then(Json::as_str).unwrap_or("").to_string();
    let dtypes = v
        .get("dtypes")
        .and_then(Json::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|j| j.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    Some(ImplId {
        backend: field("backend"),
        op: field("op"),
        dtypes,
        kernel_source: field("kernel_source"),
        kernel_revision_hash: field("kernel_revision_hash"),
    })
}

fn parse_candidates(v: &Json) -> Option<Vec<Candidate>> {
    let arr = v.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let impl_id = parse_impl_id(item.get("impl_id")?)?;
        let latency_ns = item.get("latency_ns").and_then(Json::as_u64);
        out.push(Candidate {
            impl_id,
            latency_ns,
        });
    }
    Some(out)
}

fn parse_hw(v: &Json) -> Option<HwFingerprint> {
    if !matches!(v, Json::Obj(_)) {
        return None;
    }
    let compute_capability = v.get("compute_capability").and_then(parse_cc);
    let hardware_sku = v
        .get("hardware_sku")
        .and_then(Json::as_str)
        .map(str::to_string);
    let driver_version = v
        .get("driver_version")
        .and_then(Json::as_str)
        .map(str::to_string);
    Some(HwFingerprint {
        compute_capability,
        hardware_sku,
        driver_version,
    })
}

fn parse_cc(v: &Json) -> Option<(u32, u32)> {
    let arr = v.as_array()?;
    match arr {
        [major, minor] => Some((major.as_u32()?, minor.as_u32()?)),
        _ => None,
    }
}

// ===========================================================================
// Resolver + merge bridge (call-through to the FROZEN dispatch.rs seams)
// ===========================================================================

/// Map a device `(major, minor)` compute capability to the specialization
/// [`ArchSku`], mirroring `baracuda-kernels-bench::arch_sku_of`.
///
/// Duplicated here (the ~4-line map) rather than lifted into
/// `baracuda-kernels-types` so this pure-CPU crate touches neither the frozen
/// `dispatch.rs` nor the GPU-linking bench crate — see the item-08 §3 decision.
/// `arch_sku_maps_capabilities` pins it against the bench's cases so drift is
/// caught in this suite.
#[must_use]
pub fn arch_sku_of(major: u32, minor: u32) -> Option<ArchSku> {
    match (major, minor) {
        (8, 9) => Some(ArchSku::Sm89),  // Ada — RTX 4070 / RTX 6000 Ada / L40S
        (8, _) => Some(ArchSku::Sm80),  // Ampere — A100 (sm_80), consumer sm_86/87
        (9, _) => Some(ArchSku::Sm90a), // Hopper
        _ => None,
    }
}

/// Resolve a Fuel [`ImplId`] to Baracuda's `(Implementor, entry_point)` vocabulary
/// (mirrors `Implementor::from_code`). An unknown backend yields `None` — an
/// honest skip, never a guess. `entry_point` is the kernel symbol
/// (`kernel_source`) for a generated/bespoke kernel, `None` for a vendor library.
#[must_use]
pub fn resolve_impl(id: &ImplId) -> Option<(Implementor, Option<String>)> {
    let implementor = Implementor::from_code(&id.backend)?;
    let entry_point = matches!(implementor, Implementor::Generated | Implementor::Bespoke)
        .then(|| id.kernel_source.clone());
    Some((implementor, entry_point))
}

/// Fold Fuel-reported dispatch wins into `table` through the FROZEN seams.
///
/// Per record: a record with no hardware stamp or `compute_capability: None`, or
/// a capability with no built [`ArchSku`], is **dropped** (stampless ⇒ dropped,
/// not guessed). The stamp's `captured_unix_s` is **injected** by the caller (it
/// is not on the wire). The chosen and each candidate are resolved to
/// [`ReportedCandidate`]s (the chosen's latency read out of `candidates[]`), one
/// [`reported_entry`] is built per record, and a **single** frozen [`merge`] folds
/// them in — honoring Fuel's `chosen`, arch-gating on `compute_capability`, and
/// respecting `MIN_FLIP_MARGIN`, all of which live in `dispatch.rs`.
pub fn merge_reports(ingest: &Ingest, captured_unix_s: u64, table: &mut DispatchTable) {
    let mut entries = Vec::new();
    for rec in &ingest.dispatches {
        let Some(hw) = &rec.hw else { continue };
        let Some((major, minor)) = hw.compute_capability else {
            continue;
        };
        let Some(arch) = arch_sku_of(major, minor) else {
            continue;
        };
        let Some((chosen_imp, chosen_entry)) = resolve_impl(&rec.chosen) else {
            continue;
        };
        let stamp = HwStamp {
            arch,
            device_name: hw.hardware_sku.clone().unwrap_or_default(),
            cuda_version: hw.driver_version.clone().unwrap_or_default(),
            captured_unix_s,
        };
        // The chosen's latency is whichever candidate carries its ImplId.
        let chosen_latency = rec
            .candidates
            .iter()
            .find(|c| c.impl_id == rec.chosen)
            .and_then(|c| c.latency_ns);
        let chosen_rc = ReportedCandidate {
            implementor: chosen_imp,
            entry_point: chosen_entry,
            latency_ns: chosen_latency,
        };
        let cand_rcs = rec
            .candidates
            .iter()
            .filter_map(|c| {
                let (implementor, entry_point) = resolve_impl(&c.impl_id)?;
                Some(ReportedCandidate {
                    implementor,
                    entry_point,
                    latency_ns: c.latency_ns,
                })
            })
            .collect();
        if let Some(entry) =
            reported_entry(rec.structure_key.clone(), chosen_rc, cand_rcs, Some(stamp))
        {
            entries.push(entry);
        }
    }
    merge(table, &entries);
}

// ===========================================================================
// Demand ranking + variant votes
// ===========================================================================

/// A cell's demand row: how much it was missed vs how often a generated
/// implementation won a dispatch there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankedCell {
    /// The cell.
    pub key: StructureKey,
    /// Σ `count` over miss records requesting this cell.
    pub miss_count: u64,
    /// Σ `count` over dispatch records this cell where the chosen resolves to
    /// [`Implementor::Generated`].
    pub dispatch_wins: u64,
}

/// A per-variant vote: how much a generated kernel revision won a cell.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VariantVote {
    /// The cell.
    pub key: StructureKey,
    /// The generated kernel's revision hash (the per-variant discriminator).
    pub revision_hash: String,
    /// Σ `count` of generated dispatch wins for this `(cell, revision)`.
    pub wins: u64,
}

/// Rank cells by demand: misses grouped by `wanted` (Σ `count`) and generated
/// dispatch wins grouped by `structure_key` (Σ `count` where the chosen resolves
/// to [`Implementor::Generated`]). Sorted **miss_count DESC, then stable token
/// order**, so a dispatch-only cell (`miss_count == 0`) ranks below any missed
/// cell. `est_speedup_if_available` is not on the ratified wire — demand is ranked
/// by `count`, not a speedup estimate. Deterministic under input permutation
/// (records are aggregated by token first).
#[must_use]
pub fn rank_matrix(ingest: &Ingest) -> Vec<RankedCell> {
    use std::collections::BTreeMap;
    let mut agg: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for m in &ingest.misses {
        let slot = agg.entry(m.wanted.clone()).or_default();
        slot.0 = slot.0.saturating_add(m.count);
    }
    for d in &ingest.dispatches {
        if let Some(tok) = &d.structure_key {
            if matches!(resolve_impl(&d.chosen), Some((Implementor::Generated, _))) {
                let slot = agg.entry(tok.clone()).or_default();
                slot.1 = slot.1.saturating_add(d.count);
            }
        }
    }
    let mut cells: Vec<RankedCell> = agg
        .into_iter()
        .filter_map(|(tok, (miss_count, dispatch_wins))| {
            Some(RankedCell {
                key: StructureKey::from_token(&tok)?,
                miss_count,
                dispatch_wins,
            })
        })
        .collect();
    // Stable sort keeps the BTreeMap's token order for equal miss counts.
    cells.sort_by_key(|c| std::cmp::Reverse(c.miss_count));
    cells
}

/// Tally generated-dispatch wins per `(cell, revision_hash)`. Only records whose
/// chosen resolves to [`Implementor::Generated`] vote. Sorted **wins DESC, then
/// stable `(token, revision)` order**; deterministic under input permutation.
///
/// The revision hash is carried verbatim from the wire; cross-checking it against
/// the generated `BARACUDA_LINK_REGISTRY` is a downstream (CLI/runtime) step — this
/// pure module does not depend on the generated sys artifact.
#[must_use]
pub fn variant_votes(ingest: &Ingest) -> Vec<VariantVote> {
    use std::collections::BTreeMap;
    let mut agg: BTreeMap<(String, String), u64> = BTreeMap::new();
    for d in &ingest.dispatches {
        let Some(tok) = &d.structure_key else {
            continue;
        };
        if matches!(resolve_impl(&d.chosen), Some((Implementor::Generated, _))) {
            let slot = agg
                .entry((tok.clone(), d.chosen.kernel_revision_hash.clone()))
                .or_default();
            *slot = slot.saturating_add(d.count);
        }
    }
    let mut votes: Vec<VariantVote> = agg
        .into_iter()
        .filter_map(|((tok, revision_hash), wins)| {
            Some(VariantVote {
                key: StructureKey::from_token(&tok)?,
                revision_hash,
                wins,
            })
        })
        .collect();
    // Stable sort keeps the BTreeMap's (token, revision) order for equal wins.
    votes.sort_by_key(|v| std::cmp::Reverse(v.wins));
    votes
}

#[cfg(test)]
mod tests {
    use super::*;
    use baracuda_kernel_vocab::{
        ArchSku, DispatchEntry, DispatchTable, ElementKind, HwStamp, Implementor, OpCategory,
        OperandDesc, Provenance, ReportedCandidate, StructureKey, merge, reported_entry,
        structure_key, structure_key_token,
    };
    use std::io::Cursor;

    // ---- fixtures ---------------------------------------------------------

    /// A real, parseable sm89 binary-elementwise f32 token.
    fn ew_token() -> String {
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        structure_key_token(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89)
    }

    /// A second, distinct sm89 token (unary f32) so rank/vote grouping is real.
    fn une_token() -> String {
        let a = OperandDesc::new(2, &[64, 128], &[128, 1], ElementKind::F32, 256);
        structure_key_token(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89)
    }

    fn ew_key() -> StructureKey {
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89)
    }

    /// Build a dispatch-record JSON line from parts.
    fn dispatch_line(token: &str, chosen_backend: &str, chosen_src: &str, count: u64) -> String {
        format!(
            r#"{{"schema":1,"structure_key":"{token}","chosen":{{"backend":"{chosen_backend}","op":"add","dtypes":["f32"],"kernel_source":"{chosen_src}","kernel_revision_hash":"abc123"}},"candidates":[{{"impl_id":{{"backend":"{chosen_backend}","op":"add","dtypes":["f32"],"kernel_source":"{chosen_src}","kernel_revision_hash":"abc123"}},"latency_ns":100}}],"count":{count},"hw":{{"compute_capability":[8,9],"hardware_sku":"RTX 4070","driver_version":"13.3"}}}}"#
        )
    }

    fn miss_line(wanted: &str, count: u64) -> String {
        format!(
            r#"{{"schema":1,"wanted":"{wanted}","fallback":{{"backend":"cublas","op":"gemm","dtypes":["f16"],"kernel_source":"","kernel_revision_hash":""}},"count":{count}}}"#
        )
    }

    fn ingest_str(s: &str) -> Ingest {
        ingest_jsonl(Cursor::new(s.to_string()))
    }

    // ---- ingest: round-trip + separable fields ----------------------------

    #[test]
    fn ingest_round_trips_records_with_separable_impl_id_fields() {
        let tok = ew_token();
        let line = format!(
            r#"{{"schema":1,"structure_key":"{tok}","chosen":{{"backend":"gen","op":"add","dtypes":["f32","f16"],"kernel_source":"baracuda_gen_add","kernel_revision_hash":"deadbeef"}},"candidates":[],"count":7,"hw":{{"compute_capability":[8,9],"hardware_sku":"RTX 4070","driver_version":"13.3"}}}}"#
        );
        let ing = ingest_str(&line);
        assert_eq!(ing.skipped, 0);
        assert_eq!(ing.dispatches.len(), 1);
        let d = &ing.dispatches[0];
        // The five ImplId fields land SEPARABLE — no concatenation/hashing.
        assert_eq!(d.chosen.backend, "gen");
        assert_eq!(d.chosen.op, "add");
        assert_eq!(d.chosen.dtypes, vec!["f32".to_string(), "f16".to_string()]);
        assert_eq!(d.chosen.kernel_source, "baracuda_gen_add");
        assert_eq!(d.chosen.kernel_revision_hash, "deadbeef");
        assert_eq!(d.count, 7);
        assert_eq!(d.structure_key.as_deref(), Some(tok.as_str()));
        let hw = d.hw.as_ref().unwrap();
        assert_eq!(hw.compute_capability, Some((8, 9)));
        assert_eq!(hw.hardware_sku.as_deref(), Some("RTX 4070"));
    }

    #[test]
    fn ingest_parses_multiple_records_and_both_kinds() {
        let feed = format!(
            "{}\n{}\n{}\n",
            dispatch_line(&ew_token(), "gen", "k_a", 3),
            miss_line(&ew_token(), 5),
            dispatch_line(&une_token(), "gen", "k_b", 2),
        );
        let ing = ingest_str(&feed);
        assert_eq!(ing.skipped, 0);
        assert_eq!(ing.dispatches.len(), 2);
        assert_eq!(ing.misses.len(), 1);
    }

    // ---- ingest: the skip ladder ------------------------------------------

    #[test]
    fn ingest_skips_are_counted_and_panic_free() {
        let tok = ew_token();
        let cases = [
            // unparseable JSON
            r#"{"schema":1,"chosen":"#.to_string(),
            // wrong |-count token (too few fields) on a miss
            r#"{"schema":1,"wanted":"sk2|bin","fallback":{"backend":"gen"},"count":1}"#.to_string(),
            // empty JSON object (no chosen/wanted)
            "{}".to_string(),
            // non-ascii garbage token on a dispatch
            r#"{"schema":1,"structure_key":"sk2|café","chosen":{"backend":"gen"}}"#.to_string(),
            // schema over the max
            format!(r#"{{"schema":999,"structure_key":"{tok}","chosen":{{"backend":"gen"}}}}"#),
            // two concatenated objects on one line (no-concatenation guard)
            format!(
                "{}{}",
                dispatch_line(&tok, "gen", "k", 1),
                dispatch_line(&tok, "gen", "k", 1)
            ),
        ];
        let feed = cases.join("\n");
        let ing = ingest_str(&feed);
        assert_eq!(ing.dispatches.len(), 0);
        assert_eq!(ing.misses.len(), 0);
        assert_eq!(
            ing.skipped,
            cases.len(),
            "every malformed line skipped once"
        );
    }

    #[test]
    fn ingest_deeply_nested_line_is_a_counted_skip_not_a_stack_overflow() {
        // Regression (adversarial review, MAX_JSON_DEPTH): the recursive-descent
        // scanner must BOUND its nesting depth so a pathological/corrupt telemetry
        // line of deeply-nested brackets is a counted skip — NOT an uncatchable
        // stack-overflow process abort (the panic-free/TOTAL contract). Before the
        // depth cap this 200k-`[` line aborted the whole process; after, `value`
        // returns `None` past the cap and the line is skipped. The next valid line
        // still ingests, proving the scanner recovers gracefully.
        let deep = "[".repeat(200_000);
        let feed = format!("{deep}\n{}", dispatch_line(&ew_token(), "gen", "k", 1));
        let ing = ingest_str(&feed);
        assert_eq!(
            ing.skipped, 1,
            "the over-deep line is a counted skip, not a crash"
        );
        assert_eq!(
            ing.dispatches.len(),
            1,
            "the following valid line still ingests after the skip"
        );
    }

    #[test]
    fn ingest_version_gate_skips_future_structure_key() {
        // A syntactically valid future `sk3|…` token (one past the current v2):
        // from_token would parse the version field, so ONLY the explicit version
        // check (`key.version == STRUCTURE_KEY_VERSION`) rejects it.
        let future = ew_token().replacen("sk2|", "sk3|", 1);
        assert!(future.starts_with("sk3|"));
        let line = format!(
            r#"{{"schema":1,"wanted":"{future}","fallback":{{"backend":"gen","op":"","dtypes":[],"kernel_source":"","kernel_revision_hash":""}},"count":1}}"#
        );
        let ing = ingest_str(&line);
        assert_eq!(ing.misses.len(), 0);
        assert_eq!(
            ing.skipped, 1,
            "future sk2 token skipped by the version gate"
        );
    }

    #[test]
    fn ingest_tolerates_unknown_fields_and_blank_lines() {
        let tok = ew_token();
        let line = format!(
            r#"{{"schema":1,"structure_key":"{tok}","chosen":{{"backend":"gen","op":"add","dtypes":["f32"],"kernel_source":"k","kernel_revision_hash":"r","future_field":true}},"count":1,"count_unit":"elements","hw":null}}"#
        );
        let feed = format!("\n   \n{line}\n\n");
        let ing = ingest_str(&feed);
        assert_eq!(ing.skipped, 0, "blank lines are not skips");
        assert_eq!(ing.dispatches.len(), 1);
        assert!(ing.dispatches[0].hw.is_none());
    }

    #[test]
    fn ingest_keeps_stampless_dispatch_then_merge_drops_it() {
        // compute_capability:None survives ingest but is dropped at merge.
        let tok = ew_token();
        let line = format!(
            r#"{{"schema":1,"structure_key":"{tok}","chosen":{{"backend":"gen","op":"add","dtypes":["f32"],"kernel_source":"k","kernel_revision_hash":"r"}},"candidates":[],"count":1,"hw":{{"hardware_sku":"cpu","driver_version":"n/a"}}}}"#
        );
        let ing = ingest_str(&line);
        assert_eq!(ing.dispatches.len(), 1);
        assert_eq!(
            ing.dispatches[0].hw.as_ref().unwrap().compute_capability,
            None
        );
        let mut table = DispatchTable::new();
        merge_reports(&ing, 42, &mut table);
        assert!(
            table.lookup(&ew_key()).is_none(),
            "compute_capability:None dropped at merge"
        );
    }

    // ---- resolver + arch map ---------------------------------------------

    #[test]
    fn arch_sku_maps_capabilities() {
        // Pinned against baracuda-kernels-bench::arch_sku_of so a drift is caught.
        assert_eq!(arch_sku_of(8, 9), Some(ArchSku::Sm89));
        assert_eq!(arch_sku_of(8, 0), Some(ArchSku::Sm80));
        assert_eq!(arch_sku_of(8, 6), Some(ArchSku::Sm80));
        assert_eq!(arch_sku_of(9, 0), Some(ArchSku::Sm90a));
        assert_eq!(arch_sku_of(7, 5), None);
    }

    #[test]
    fn resolve_impl_mirrors_from_code_and_entry_point_policy() {
        let id = |backend: &str, src: &str| ImplId {
            backend: backend.to_string(),
            op: String::new(),
            dtypes: vec![],
            kernel_source: src.to_string(),
            kernel_revision_hash: String::new(),
        };
        assert_eq!(
            resolve_impl(&id("gen", "sym")),
            Some((Implementor::Generated, Some("sym".to_string())))
        );
        assert_eq!(
            resolve_impl(&id("bespoke", "sym")),
            Some((Implementor::Bespoke, Some("sym".to_string())))
        );
        // Vendor libraries have no entry point.
        assert_eq!(
            resolve_impl(&id("cublas", "sym")),
            Some((Implementor::Cublas, None))
        );
        // Unknown backend is an honest None.
        assert_eq!(resolve_impl(&id("rocm", "sym")), None);
    }

    // ---- merge bridge composition ----------------------------------------

    #[test]
    fn merge_reports_composes_to_hand_built_reported_entry_and_merge() {
        let tok = ew_token();
        // Fuel chose gen/kern_b (200ns), with a faster gen/kern_a (100ns) present.
        let line = format!(
            r#"{{"schema":1,"structure_key":"{tok}","chosen":{{"backend":"gen","op":"add","dtypes":["f32"],"kernel_source":"kern_b","kernel_revision_hash":"rb"}},"candidates":[{{"impl_id":{{"backend":"gen","op":"add","dtypes":["f32"],"kernel_source":"kern_a","kernel_revision_hash":"ra"}},"latency_ns":100}},{{"impl_id":{{"backend":"gen","op":"add","dtypes":["f32"],"kernel_source":"kern_b","kernel_revision_hash":"rb"}},"latency_ns":200}}],"count":9,"hw":{{"compute_capability":[8,9],"hardware_sku":"RTX 4070","driver_version":"13.3"}}}}"#
        );
        let ing = ingest_str(&line);
        let mut via_bridge = DispatchTable::new();
        merge_reports(&ing, 40, &mut via_bridge);

        // Hand-built through the frozen seams directly.
        let stamp = HwStamp {
            arch: ArchSku::Sm89,
            device_name: "RTX 4070".to_string(),
            cuda_version: "13.3".to_string(),
            captured_unix_s: 40,
        };
        let rc = |src: &str, ns: Option<u64>| ReportedCandidate {
            implementor: Implementor::Generated,
            entry_point: Some(src.to_string()),
            latency_ns: ns,
        };
        let entry = reported_entry(
            Some(tok.clone()),
            rc("kern_b", Some(200)),
            vec![rc("kern_a", Some(100)), rc("kern_b", Some(200))],
            Some(stamp),
        )
        .unwrap();
        let mut hand = DispatchTable::new();
        merge(&mut hand, &[entry]);

        assert_eq!(
            via_bridge, hand,
            "bridge == hand-built reported_entry+merge"
        );
        // Fuel's chosen honored (kern_b), margin < 1 recorded (100/200).
        let e = via_bridge.lookup(&ew_key()).unwrap();
        assert_eq!(e.winner, Implementor::Generated);
        assert_eq!(e.winner_entry.as_deref(), Some("kern_b"));
        assert_eq!(e.provenance, Provenance::Reported);
        assert!((e.margin - 0.5).abs() < 1e-9);
    }

    #[test]
    fn merge_reports_arch_gates_and_respects_min_flip_margin() {
        // A record captured on sm80 must not route the sm89 cell (arch-gate).
        let tok = ew_token(); // sm89
        let line = format!(
            r#"{{"schema":1,"structure_key":"{tok}","chosen":{{"backend":"gen","op":"add","dtypes":["f32"],"kernel_source":"k","kernel_revision_hash":"r"}},"candidates":[{{"impl_id":{{"backend":"gen","op":"add","dtypes":["f32"],"kernel_source":"k","kernel_revision_hash":"r"}},"latency_ns":100}}],"count":1,"hw":{{"compute_capability":[8,0],"hardware_sku":"A100","driver_version":"12.4"}}}}"#
        );
        let ing = ingest_str(&line);
        let mut table = DispatchTable::new();
        merge_reports(&ing, 30, &mut table);
        assert!(
            table.lookup(&ew_key()).is_none(),
            "sm80 record can't route sm89"
        );

        // A sub-1 margin Reported row cannot flip an established different route.
        let mut table2 = DispatchTable::from_entries(vec![DispatchEntry {
            structure_key: tok.clone(),
            winner: Implementor::Generated,
            winner_entry: Some("kern_a".to_string()),
            margin: 2.0,
            ranked: Vec::new(),
            provenance: Provenance::Measured,
            measured_on: Some(HwStamp {
                arch: ArchSku::Sm89,
                device_name: "RTX 4070".to_string(),
                cuda_version: "13.3".to_string(),
                captured_unix_s: 10,
            }),
        }]);
        // Fuel chose kern_b (200), faster kern_a (100) present → margin 0.5 < 1.10.
        let flip = format!(
            r#"{{"schema":1,"structure_key":"{tok}","chosen":{{"backend":"gen","op":"add","dtypes":["f32"],"kernel_source":"kern_b","kernel_revision_hash":"rb"}},"candidates":[{{"impl_id":{{"backend":"gen","op":"add","dtypes":["f32"],"kernel_source":"kern_a","kernel_revision_hash":"ra"}},"latency_ns":100}},{{"impl_id":{{"backend":"gen","op":"add","dtypes":["f32"],"kernel_source":"kern_b","kernel_revision_hash":"rb"}},"latency_ns":200}}],"count":1,"hw":{{"compute_capability":[8,9],"hardware_sku":"RTX 4070","driver_version":"13.3"}}}}"#
        );
        merge_reports(&ingest_str(&flip), 50, &mut table2);
        assert_eq!(
            table2.lookup(&ew_key()).unwrap().winner_entry.as_deref(),
            Some("kern_a"),
            "sub-MIN_FLIP_MARGIN Reported row cannot flip a different route"
        );
    }

    // ---- rank + votes -----------------------------------------------------

    #[test]
    fn rank_matrix_sorts_miss_desc_then_token_and_is_permutation_stable() {
        let ew = ew_token();
        let une = une_token();
        // ew missed 5, une missed 5 (tie → token order); a dispatch-only cell.
        let base = vec![
            miss_line(&ew, 2),
            miss_line(&ew, 3), // ew total 5
            miss_line(&une, 5),
            dispatch_line(&ew, "gen", "k", 4), // ew dispatch_wins 4
        ];
        let a = rank_matrix(&ingest_str(&base.join("\n")));
        // Two missed cells, tie at 5 → token ascending; dispatch win folded in.
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].miss_count, 5);
        assert_eq!(a[1].miss_count, 5);
        assert!(
            a[0].key.to_token() < a[1].key.to_token(),
            "tie broken by token"
        );
        let ew_cell = a.iter().find(|c| c.key.to_token() == ew).unwrap();
        assert_eq!(ew_cell.dispatch_wins, 4);

        // Permuting input order yields the identical ranking.
        let mut perm = base.clone();
        perm.reverse();
        assert_eq!(rank_matrix(&ingest_str(&perm.join("\n"))), a);

        // A dispatch-only cell (no misses) ranks below any missed cell.
        let d_only = [dispatch_line(&une, "gen", "k", 9), miss_line(&ew, 1)];
        let r = rank_matrix(&ingest_str(&d_only.join("\n")));
        assert_eq!(r[0].key.to_token(), ew, "missed cell first");
        assert_eq!(r[0].miss_count, 1);
        assert_eq!(r.last().unwrap().miss_count, 0, "dispatch-only cell last");
    }

    #[test]
    fn variant_votes_tally_per_token_and_revision() {
        // Two revisions of gen at the ew cell + a vendor dispatch (no vote).
        let ew = ew_token();
        let rev = |src: &str, rev: &str, count: u64| {
            format!(
                r#"{{"schema":1,"structure_key":"{ew}","chosen":{{"backend":"gen","op":"add","dtypes":["f32"],"kernel_source":"{src}","kernel_revision_hash":"{rev}"}},"candidates":[],"count":{count}}}"#
            )
        };
        let feed = vec![
            rev("k_a", "r1", 3),
            rev("k_a", "r1", 2), // r1 total 5
            rev("k_b", "r2", 4), // r2 total 4
            // a cublas win — not generated, no vote
            dispatch_line(&ew, "cublas", "", 100),
        ];
        let votes = variant_votes(&ingest_str(&feed.join("\n")));
        assert_eq!(votes.len(), 2, "only generated revisions vote");
        // wins DESC: r1 (5) before r2 (4).
        assert_eq!(votes[0].revision_hash, "r1");
        assert_eq!(votes[0].wins, 5);
        assert_eq!(votes[1].revision_hash, "r2");
        assert_eq!(votes[1].wins, 4);
        // Deterministic under permutation.
        let mut perm = feed.clone();
        perm.reverse();
        assert_eq!(variant_votes(&ingest_str(&perm.join("\n"))), votes);
    }

    #[test]
    fn token_join_fidelity_round_trips_to_the_reducer_key() {
        // The token the reducer keys on parses back to the same StructureKey.
        let key = ew_key();
        let tok = key.to_token();
        let ing = ingest_str(&miss_line(&tok, 1));
        let ranked = rank_matrix(&ing);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].key, key, "reducer key == the source StructureKey");
    }
}
