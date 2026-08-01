# Sub-spec A: General Contraction Operand Roles + Layout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a `Permute`/`Broadcast`/rank-3 role-reorder on a contraction operand fold into the M/N/K address math (no materialized `contiguize` copy), unblocking transposed / mixed-layout / batched matmul and the SDPA QKᵀ path.

**Architecture:** Layout becomes a discrete class in `ContractionKey` (so it re-keys to a distinct cell), lowered through ONE extent-product stride-binding emitter that reads `ContractionAxes` roles (semantic) + each operand's `View::Permute` (storage order). Two plan gates lift; the vocab crate's `dense()` predicate generalizes to `classify_mat_layout`; the CPU differential oracle is the correctness gate; every stride is an extent-product of the existing `m/n/k/B` launch args (no new runtime stride args in v1).

**Tech Stack:** Rust (workspace crates `baracuda-kernel-vocab`, `baracuda-kernelgen`); CUDA C string emission; `cargo test` (host) + `pwsh scripts/gpu-run.ps1` (RTX 4070 sm_89, CUDA 13.3).

**Spec:** `docs/superpowers/specs/2026-08-01-ir-layout-shape-subspec-a-roles-layout-design.md` (parent: `…-parent-design.md`).

## Global Constraints

- **Byte-identical back-compat:** every existing row-major (canonical) contraction cell MUST emit a byte-identical `StructureKey` token AND byte-identical CUDA source. Pin with golden diffs (HEAD vs branch = 0). This is the load-bearing invariant.
- **Additive codec, NO `STRUCTURE_KEY_VERSION` bump:** the new layout-order token component appears ONLY for a non-identity layout (mirrors `batch: Option<SizeClass>` → `/b<class>`). Identity default serializes byte-identically.
- **Key carries size/layout CLASSES, never concrete extents** (`structure_key.rs:302`). Extent/stride agreement is a caller precondition, not plan-validated.
- **View stays out of `ScalarExpr` and `Access`** — consumed into key + binding at plan time only.
- **Correct-first / optimize-as-variant:** emit correct (possibly non-coalesced) address math for every admitted layout; coalescing schedules are deferred bench-gated variants. Correctness (CPU oracle bit-exact) is the gate, not speed.
- **Single K-group in v1:** exactly one `ContractedK` axis per operand.
- **Output stays canonical row-major** (output views are sub-spec B).
- **Every on-device run routes through the lock:** `pwsh scripts/gpu-run.ps1 -Project baracuda -- <cmd>`.
- Determinism: `AccumSpec::WideFloat`, K ascending, matching the CPU oracle's accumulation order.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `crates/baracuda-kernel-vocab/src/structure_key.rs` | keys + operand classification | add `LayoutOrder` type + `lhs_order`/`rhs_order` on `ContractionKey`; `dense()`/`dense3()` → `classify_mat_layout`; additive token codec |
| `crates/baracuda-kernel-vocab/src/shape_expr.rs` | wire codec | order-token round-trip (only if the codec lives here; else co-located with `to_token`) |
| `crates/baracuda-kernelgen/src/plan.rs` | plan gates + role legality | add `validate_contraction_roles`; lift `:1043` for Contraction; replace `:366` pin |
| `crates/baracuda-kernelgen/src/cuda.rs` | emitter | `emit_contraction` stride-binding; extend `assert_views_lowerable` backstop (`:1757`) |
| `crates/baracuda-kernelgen/src/oracle.rs` | CPU differential reference | generalize `eval_contraction` to the admitted layouts (roles + storage order + batch + broadcast) |
| `crates/baracuda-kernelgen/tests/` + `ondevice/` | validation | oracle matrix + on-device headline cells |

Dependency order: **T1 → T2** (vocab keying), **T3 → T4** (plan gates), **T5** (emitter), **T6** (oracle gate), **T7** (on-device). T3 is independent of T1/T2 and may proceed in parallel; T5 depends on T2+T4; T6 depends on T5.

---

### Task 1: `LayoutOrder` type + `ContractionKey` fields + additive codec

**Files:**
- Modify: `crates/baracuda-kernel-vocab/src/structure_key.rs` (`ContractionKey` at `:239`; token codec — locate `to_token`/`from_token` for the contraction group, the `/b<class>` batch component is the pattern to mirror)
- Test: same file's `#[cfg(test)]` module (existing contraction key tests live there)

**Interfaces:**
- Produces: `pub struct LayoutOrder { perm: [u8; MAX_RANK], rank: u8 }` with `LayoutOrder::identity(rank: u8) -> Self`, `LayoutOrder::from_perm(&[u8]) -> Self`, `fn is_identity(&self) -> bool`, `fn perm(&self) -> &[u8]`; `Copy + Clone + Debug + Eq + PartialEq + Hash + Default` (Default = `identity(0)`). New `ContractionKey` fields `pub lhs_order: LayoutOrder, pub rhs_order: LayoutOrder`.

- [ ] **Step 1: Write the failing test — canonical token is byte-identical; transposed round-trips**

```rust
#[test]
fn contraction_layout_order_token_is_additive() {
    // A canonical rank-2 matmul key: both orders identity → token unchanged.
    let canon = sample_matmul_key(); // helper building a ContractionKey via structure_key() with dense row-major operands
    let tok_canon = canon.to_token();
    assert!(!tok_canon.contains("/ol") && !tok_canon.contains("/or"),
        "identity layout must emit no order component: {tok_canon}");
    assert_eq!(StructureKey::from_token(&tok_canon).unwrap(), canon, "canonical round-trips");

    // A transposed-rhs key: rhs storage order [1,0] → adds `/or10`, round-trips.
    let mut trans = canon;
    trans.contraction.as_mut().unwrap().rhs_order = LayoutOrder::from_perm(&[1, 0]);
    let tok_trans = trans.to_token();
    assert!(tok_trans.contains("/or10"), "transposed rhs emits /or10: {tok_trans}");
    assert_eq!(StructureKey::from_token(&tok_trans).unwrap(), trans, "transposed round-trips");
    assert_ne!(tok_trans, tok_canon, "transposed cell re-keys");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p baracuda-kernel-vocab contraction_layout_order_token_is_additive`
Expected: FAIL — `LayoutOrder` and the `lhs_order`/`rhs_order` fields do not exist.

- [ ] **Step 3: Add the `LayoutOrder` type and the `ContractionKey` fields**

```rust
/// Storage-order class of a contraction operand: `perm[d]` is the storage axis
/// read at role/logical position `d` (identity = canonical row-major). Copy,
/// heap-free; the identity default serializes byte-identically (additive codec).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct LayoutOrder {
    perm: [u8; MAX_RANK],
    rank: u8,
}
impl Default for LayoutOrder {
    fn default() -> Self { Self::identity(0) }
}
impl LayoutOrder {
    #[must_use] pub fn identity(rank: u8) -> Self {
        let mut perm = [0u8; MAX_RANK];
        for (i, p) in perm.iter_mut().enumerate() { *p = i as u8; }
        Self { perm, rank }
    }
    #[must_use] pub fn from_perm(p: &[u8]) -> Self {
        let mut perm = [0u8; MAX_RANK];
        perm[..p.len()].copy_from_slice(p);
        Self { perm, rank: p.len() as u8 }
    }
    #[must_use] pub fn is_identity(&self) -> bool {
        (0..self.rank as usize).all(|i| self.perm[i] as usize == i)
    }
    #[must_use] pub fn perm(&self) -> &[u8] { &self.perm[..self.rank as usize] }
}
```

Add to `ContractionKey` (after `batch`, before the precision group `wdt`, matching the geometry grouping):

```rust
    /// Storage-order class of the lhs operand (identity = canonical row-major).
    pub lhs_order: LayoutOrder,
    /// Storage-order class of the rhs operand (identity = canonical row-major).
    pub rhs_order: LayoutOrder,
```

Update every existing `ContractionKey { .. }` literal in the crate (notably `derive_contraction` at `:636`/`:648`) to set `lhs_order: LayoutOrder::identity(rank), rhs_order: LayoutOrder::identity(rank)` for now (Task 2 populates real values).

- [ ] **Step 4: Extend the token codec additively**

In the contraction `to_token` path, after the batch `/b<class>` component and before the precision group, emit — ONLY when non-identity — `/ol<digits>` for `lhs_order` and `/or<digits>` for `rhs_order`, where `<digits>` is the `perm()` values concatenated (e.g. `[1,0]` → `"10"`). Mirror `from_token` to parse the optional `/ol`/`/or` components back into `LayoutOrder::from_perm`. Prefixes `ol`/`or` do not collide with `b` (batch) or `st`/`rm` (precision) per the codec's prefix discipline (`structure_key.rs:211`).

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test -p baracuda-kernel-vocab contraction_layout_order_token_is_additive`
Expected: PASS.

- [ ] **Step 6: Regression — existing key tokens unchanged**

Run: `cargo test -p baracuda-kernel-vocab`
Expected: PASS (all pre-existing contraction/key token tests green — identity default emits no new component).

- [ ] **Step 7: Commit**

```bash
git add crates/baracuda-kernel-vocab/src/structure_key.rs
git commit -m "feat(vocab): LayoutOrder + additive lhs/rhs order on ContractionKey"
```

---

### Task 2: `classify_mat_layout` — derive layout from strides, stop rejecting non-row-major

**Files:**
- Modify: `crates/baracuda-kernel-vocab/src/structure_key.rs` (`derive_contraction` `:587`; the `dense()` closure `:616` and `dense3()` `:657`)
- Test: same-file test module

**Interfaces:**
- Consumes: `LayoutOrder` (Task 1), `OperandDesc`, `AxisRole` slice.
- Produces: `fn classify_mat_layout(od: &OperandDesc, roles: &[AxisRole]) -> Option<LayoutOrder>` — `Some(order)` if the operand is a **packed** permutation (each axis unit-or-extent-product stride, i.e. a pure transpose/reorder of a contiguous tensor); `None` if genuinely non-packed (→ declines to sub-spec D). `derive_contraction` returns `Some(ContractionKey)` with real `lhs_order`/`rhs_order` for packed transposed operands.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn derive_contraction_accepts_transposed_rhs() {
    // rhs stored [N,K] read as role [K,N]: K unit-stride, N strided by k → transposed.
    // lhs canonical [M,K].
    let lhs = OperandDesc::new(0, &[8, 16], &[16, 1], ElementKind::F32, 256); // [M,K] row-major
    let rhs = OperandDesc::new(1, &[4, 16], &[1, 4], ElementKind::F32, 256);  // [K,N] but N strided by k=4, K unit → transposed store
    let out = OperandDesc::new(2, &[8, 4], &[4, 1], ElementKind::F32, 256);   // [M,N] row-major
    let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
    let c = key.contraction.expect("transposed rhs must still be a contraction");
    assert!(c.lhs_order.is_identity(), "canonical lhs stays identity");
    assert_eq!(c.rhs_order.perm(), &[1, 0], "transposed rhs order = [1,0]");
}

#[test]
fn derive_contraction_declines_nonpacked() {
    // rhs with a stride that is neither unit nor an extent-product of the other axis
    // (a genuine non-packed slice) → declines to sub-spec D (None).
    let lhs = OperandDesc::new(0, &[8, 16], &[16, 1], ElementKind::F32, 256);
    let rhs = OperandDesc::new(1, &[16, 4], &[9, 1], ElementKind::F32, 256); // K-stride 9 ≠ n(=4), N-stride 1 → non-packed
    let out = OperandDesc::new(2, &[8, 4], &[4, 1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
    assert!(key.contraction.is_none(), "non-packed operand declines in v1 (sub-spec D)");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p baracuda-kernel-vocab derive_contraction_accepts_transposed_rhs derive_contraction_declines_nonpacked`
Expected: FAIL — `derive_contraction`'s `dense()` returns `None` for the transposed rhs (current behavior).

- [ ] **Step 3: Implement `classify_mat_layout` and rewire `derive_contraction`**

```rust
/// Classify a contraction operand's storage order from its strides. Returns the
/// permutation `perm` such that role/logical axis `d` reads storage axis `perm[d]`,
/// iff the operand is a PACKED permutation of a contiguous tensor (every axis'
/// |stride| equals the product of the extents storage-inner to it). `None` if
/// genuinely non-packed (arbitrary strides → sub-spec D).
fn classify_mat_layout(od: &OperandDesc, _roles: &[AxisRole]) -> Option<LayoutOrder> {
    let rank = od.rank as usize;
    // Storage order = axes sorted by descending |stride| (outermost first).
    let mut axes: Vec<usize> = (0..rank).collect();
    axes.sort_by_key(|&d| core::cmp::Reverse(od.strides[d].abs()));
    // Verify packed: walking storage-inner→outer, |stride| must equal the running
    // extent product.
    let mut acc: i64 = 1;
    for &d in axes.iter().rev() {
        if od.strides[d].abs() != acc { return None; } // non-packed → decline
        acc = acc.saturating_mul(od.shape[d]);
    }
    // perm[logical d] = storage position of axis d. Here logical == operand axis
    // index; storage position is `axes.iter().position(|&a| a == d)`.
    let mut perm = [0u8; MAX_RANK];
    for d in 0..rank {
        perm[d] = axes.iter().position(|&a| a == d).unwrap() as u8;
    }
    Some(LayoutOrder { perm, rank: od.rank })
}
```

Rewire `derive_contraction`: replace the `!dense(lhs) || !dense(rhs) || !dense(out)` rejection with `classify_mat_layout(lhs, &axes.lhs)?` / `classify_mat_layout(rhs, &axes.rhs)?` (the `?` propagates the `None` decline for non-packed operands). Keep `out` required-canonical in sub-spec A: `if !dense(out) { return None; }` (output views are sub-spec B). Set the new fields:

```rust
lhs_order,
rhs_order,
```

(Note: `derive_contraction` currently has no `axes` — the `ContractionAxes` roles. Thread the op's `ContractionAxes` into `derive_contraction` OR derive the storage order structurally without roles as above, which is role-agnostic. The role-agnostic form above needs no `axes` argument — prefer it to avoid changing the signature; roles enter at the plan/emitter layer.)

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p baracuda-kernel-vocab derive_contraction_accepts_transposed_rhs derive_contraction_declines_nonpacked`
Expected: PASS.

- [ ] **Step 5: Regression**

Run: `cargo test -p baracuda-kernel-vocab`
Expected: PASS (canonical operands classify to identity → byte-identical keys).

- [ ] **Step 6: Commit**

```bash
git add crates/baracuda-kernel-vocab/src/structure_key.rs
git commit -m "feat(vocab): classify_mat_layout — derive packed transpose, decline non-packed"
```

---

### Task 3: `validate_contraction_roles` predicate

**Files:**
- Modify: `crates/baracuda-kernelgen/src/plan.rs` (near the contraction scope at `:355`)
- Test: `plan.rs` test module

**Interfaces:**
- Consumes: `ContractionAxes` (`ir.rs:1375`), `AxisRole` (`ir.rs:1359`).
- Produces: `fn validate_contraction_roles(axes: &ContractionAxes, lhs_rank: usize, rhs_rank: usize) -> Result<(), String>`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn role_legality() {
    use crate::ir::{ContractionAxes, AxisRole::*};
    // canonical constructors pass
    assert!(validate_contraction_roles(&ContractionAxes::matmul(), 2, 2).is_ok());
    assert!(validate_contraction_roles(&ContractionAxes::batched_matmul(), 3, 3).is_ok());
    // transposed role order passes (rhs [N,K] → roles [FreeN, ContractedK])
    let t = ContractionAxes { lhs: vec![FreeM, ContractedK], rhs: vec![FreeN, ContractedK] };
    assert!(validate_contraction_roles(&t, 2, 2).is_ok());
    // batch axis in the middle passes
    let mid = ContractionAxes { lhs: vec![FreeM, Batch, ContractedK], rhs: vec![Batch, ContractedK, FreeN] };
    assert!(validate_contraction_roles(&mid, 3, 3).is_ok());
    // illegal: two FreeM
    let bad_m = ContractionAxes { lhs: vec![FreeM, FreeM], rhs: vec![ContractedK, FreeN] };
    assert!(validate_contraction_roles(&bad_m, 2, 2).is_err());
    // illegal: FreeN on lhs
    let bad_n = ContractionAxes { lhs: vec![FreeN, ContractedK], rhs: vec![ContractedK, FreeN] };
    assert!(validate_contraction_roles(&bad_n, 2, 2).is_err());
    // illegal: mismatched batch count
    let bad_b = ContractionAxes { lhs: vec![Batch, FreeM, ContractedK], rhs: vec![ContractedK, FreeN] };
    assert!(validate_contraction_roles(&bad_b, 3, 2).is_err());
    // illegal: two K (multi-group deferred)
    let bad_k = ContractionAxes { lhs: vec![FreeM, ContractedK, ContractedK], rhs: vec![ContractedK, ContractedK, FreeN] };
    assert!(validate_contraction_roles(&bad_k, 3, 3).is_err());
    // illegal: role-vector length != rank
    assert!(validate_contraction_roles(&ContractionAxes::matmul(), 3, 2).is_err());
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p baracuda-kernelgen role_legality`
Expected: FAIL — `validate_contraction_roles` undefined.

- [ ] **Step 3: Implement the predicate**

```rust
pub(crate) fn validate_contraction_roles(
    axes: &crate::ir::ContractionAxes,
    lhs_rank: usize,
    rhs_rank: usize,
) -> Result<(), String> {
    use crate::ir::AxisRole::*;
    if axes.lhs.len() != lhs_rank || axes.rhs.len() != rhs_rank {
        return Err("role vector length must equal operand rank".into());
    }
    let count = |v: &[crate::ir::AxisRole], r: crate::ir::AxisRole| v.iter().filter(|&&x| x == r).count();
    if count(&axes.lhs, FreeM) != 1 { return Err("lhs must have exactly one FreeM".into()); }
    if count(&axes.lhs, FreeN) != 0 { return Err("lhs must not carry FreeN".into()); }
    if count(&axes.rhs, FreeN) != 1 { return Err("rhs must have exactly one FreeN".into()); }
    if count(&axes.rhs, FreeM) != 0 { return Err("rhs must not carry FreeM".into()); }
    // single K-group in v1
    if count(&axes.lhs, ContractedK) != 1 || count(&axes.rhs, ContractedK) != 1 {
        return Err("v1: exactly one ContractedK per operand (multi-group deferred)".into());
    }
    // batch correspondence
    if count(&axes.lhs, Batch) != count(&axes.rhs, Batch) {
        return Err("lhs and rhs must share the same batch-axis count".into());
    }
    Ok(())
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p baracuda-kernelgen role_legality`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/baracuda-kernelgen/src/plan.rs
git commit -m "feat(plan): general validate_contraction_roles predicate"
```

---

### Task 4: Lift the plan gates + wire role validation

**Files:**
- Modify: `crates/baracuda-kernelgen/src/plan.rs` (`:1043` view gate; `:366` role pin)
- Test: `plan.rs` test module

**Interfaces:**
- Consumes: `validate_contraction_roles` (Task 3).
- Produces: `build_plan` admits a non-Identity `View` on a `Access::Contraction` operand and validates roles via the general predicate.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn contraction_admits_permute_view_on_rhs() {
    // Build a matmul OpDef with a Permute view on rhs (transpose) — previously
    // panicked at the plan.rs:1043 Elementwise-only gate.
    let op = matmul_opdef_with_rhs_view(crate::ir::View::Permute { perm: vec![1, 0] });
    let key = /* transposed-rhs key from Task 2 fixture */;
    // Should build without panic and route to Schedule::Contraction.
    let plan = build_plan(&op, &key);
    assert!(matches!(plan.schedule, Schedule::Contraction));
}

#[test]
#[should_panic(expected = "exactly one FreeM")]
fn contraction_rejects_illegal_roles() {
    let op = matmul_opdef_with_axes(crate::ir::ContractionAxes {
        lhs: vec![crate::ir::AxisRole::FreeM, crate::ir::AxisRole::FreeM],
        rhs: vec![crate::ir::AxisRole::ContractedK, crate::ir::AxisRole::FreeN],
    });
    let _ = build_plan(&op, &canonical_matmul_key());
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p baracuda-kernelgen contraction_admits_permute_view_on_rhs contraction_rejects_illegal_roles`
Expected: FAIL — the first panics at the `:1043` gate; the second panics with the old `:366` message, not the role-predicate message.

- [ ] **Step 3: Lift the view gate for Contraction**

At `plan.rs:1043` (`assert_valid_views`), change the assertion so a non-Identity view is admitted when `matches!(op.access, Access::Elementwise | Access::Contraction { .. })`. Keep the other non-Elementwise classes rejected (out of this sub-spec). Keep the multi-output guard (`:1051`) unchanged.

- [ ] **Step 4: Replace the role pin with the general predicate**

At `plan.rs:366`, replace the `ax == matmul || ax == batched_matmul` assertion with:

```rust
let (lhs_rank, rhs_rank) = (axes.lhs.len(), axes.rhs.len());
if let Err(msg) = validate_contraction_roles(axes, lhs_rank, rhs_rank) {
    panic!("contraction roles: {msg}");
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test -p baracuda-kernelgen contraction_admits_permute_view_on_rhs contraction_rejects_illegal_roles`
Expected: PASS.

- [ ] **Step 6: Regression — existing contraction plans unchanged**

Run: `cargo test -p baracuda-kernelgen contraction`
Expected: PASS (canonical matmul/batched cells still route to `Schedule::Contraction`; roles pass the general predicate).

- [ ] **Step 7: Commit**

```bash
git add crates/baracuda-kernelgen/src/plan.rs
git commit -m "feat(plan): admit Permute view on Contraction operands + general role gate"
```

---

### Task 5: Stride-binding emitter in `emit_contraction`

**Files:**
- Modify: `crates/baracuda-kernelgen/src/cuda.rs` (`emit_contraction` `:3269`; backstop `assert_views_lowerable` `:1757`)
- Test: `cuda.rs` emission-golden module + a `tests/` golden

**Interfaces:**
- Consumes: `ContractionAxes` roles, each operand's `View::Permute` (from `op.views`), the `m/n/k/B` launch args, `ContractionKey.lhs_order`/`rhs_order`.
- Produces: `fn operand_stride_binding(roles: &[AxisRole], order: &LayoutOrder, bcast: AxisMask, extents: &ExtentSyms) -> String` returning the address expression `Σ coord(role)·strideExpr(role)`.

- [ ] **Step 1: Write the failing test — canonical byte-identical, transposed swapped**

```rust
#[test]
fn emit_contraction_canonical_byte_identical() {
    // The canonical matmul cell must emit source byte-identical to the pre-change
    // emitter. Compare against a checked-in golden captured from HEAD.
    let src = emit_contraction_source(&canonical_matmul_plan());
    assert_eq!(src, include_str!("goldens/contract_canonical_f32.cu"),
        "canonical contraction emission must be byte-identical (Global Constraint)");
    // The lhs/rhs address expressions are the row-major special case.
    assert!(src.contains("in0[") && src.contains("mm * k + kk"));
    assert!(src.contains("in1[") && src.contains("kk * n + col"));
}

#[test]
fn emit_contraction_transposed_rhs_binding() {
    // rhs order [1,0] (storage [N,K], K inner): stride(N)=k, stride(K)=1 →
    // address col*k + kk.
    let src = emit_contraction_source(&transposed_rhs_matmul_plan());
    assert!(src.contains("col * k + kk"), "transposed rhs binding: {src}");
    assert!(!src.contains("kk * n + col"), "must NOT use the canonical rhs binding");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p baracuda-kernelgen emit_contraction_canonical_byte_identical emit_contraction_transposed_rhs_binding`
Expected: FAIL — the transposed test fails (emitter is still hardcoded row-major); create the canonical golden in Step 3 first.

- [ ] **Step 3: Capture the canonical golden, then implement the binding**

First capture the current canonical emission to `crates/baracuda-kernelgen/src/goldens/contract_canonical_f32.cu` (this freezes byte-identity). Then implement the binding. For an operand with `roles` and storage `order`, compute per-role stride as the extent-product of the axes storage-inner to that role's axis:

```rust
// extents: role → its extent symbol ("m"/"n"/"k"/"b" launch args).
// order.perm()[d] = storage position of logical axis d (lower = outer).
fn operand_stride_binding(
    roles: &[AxisRole], order: &LayoutOrder, bcast: AxisMask, ext: &dyn Fn(AxisRole) -> &'static str,
) -> String {
    let rank = roles.len();
    let mut terms = Vec::new();
    for d in 0..rank {
        if bcast.is_set(d as u8) { continue; } // stride-0 broadcast axis: drop term
        // storage position of axis d; stride = product of extents of axes with a
        // GREATER storage position (i.e. storage-inner).
        let sp = order.perm()[d] as usize;
        let factors: Vec<&str> = (0..rank)
            .filter(|&e| (order.perm()[e] as usize) > sp && !bcast.is_set(e as u8))
            .map(|e| ext(roles[e]))
            .collect();
        let stride = if factors.is_empty() { "1".to_string() } else { factors.join(" * ") };
        let coord = match roles[d] {
            AxisRole::FreeM => "mm", AxisRole::FreeN => "col",
            AxisRole::ContractedK => "kk", AxisRole::Batch => "b",
        };
        terms.push(if stride == "1" { coord.to_string() } else { format!("{coord} * {stride}") });
    }
    if terms.is_empty() { "0".into() } else { terms.join(" + ") }
}
```

Replace the three hardcoded offsets in `emit_contraction` (`in0[{lb}mm*k+kk]`, `in1[{rb}kk*n+col]`, `out[{ob}mm*n+col]`) with `operand_stride_binding(...)` for lhs and rhs; **out stays canonical** (`mm * n + col`, output views are sub-spec B). Verify the canonical case reproduces `mm * k + kk` / `kk * n + col` EXACTLY (it must, or the golden fails).

- [ ] **Step 4: Extend the backstop**

At `cuda.rs:1757` (`assert_views_lowerable`), admit the contraction binding path (a non-Identity view on a Contraction operand is now lowerable via `operand_stride_binding`).

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test -p baracuda-kernelgen emit_contraction_canonical_byte_identical emit_contraction_transposed_rhs_binding`
Expected: PASS (canonical byte-identical against the golden; transposed uses the swapped binding).

- [ ] **Step 6: Full-suite byte-identity regression**

Run: `cargo test -p baracuda-kernelgen`
Expected: PASS — every pre-existing contraction emission golden unchanged.

- [ ] **Step 7: Commit**

```bash
git add crates/baracuda-kernelgen/src/cuda.rs crates/baracuda-kernelgen/src/goldens/contract_canonical_f32.cu
git commit -m "feat(cuda): extent-product stride-binding for contraction operand layouts"
```

---

### Task 6: Generalize the CPU oracle + the differential matrix

**Files:**
- Modify: `crates/baracuda-kernelgen/src/oracle.rs` (`eval_contraction` `:1284`)
- Test: `crates/baracuda-kernelgen/tests/contraction_layout_oracle.rs` (new)

**Interfaces:**
- Consumes: the emitter (Task 5), `ContractionAxes`, layout orders, `oracle::evaluate` (`:1053`).
- Produces: `eval_contraction` reads roles + storage order + batch + broadcast to index its reference loop, K ascending, matching the emitter's accumulation order.

- [ ] **Step 1: Write the failing differential test (transposed rhs, rank-2)**

```rust
#[test]
fn oracle_matches_emitter_transposed_rhs_f32() {
    // Same logical matmul, rhs fed transposed. The oracle reference (general
    // indexing) must equal a from-scratch [M,K]·[K,N] matmul, K ascending.
    let (m, k, n) = (5, 7, 3);
    let lhs = fill_row_major(&[m, k]);          // [M,K]
    let rhs_t = fill_transposed(&[k, n]);       // logical [K,N], stored [N,K]
    let plan = transposed_rhs_plan(m, k, n);
    let got = oracle::evaluate(&plan, &operands, &[lhs.clone(), rhs_t.clone()], &[]);
    let want = naive_matmul_kascending(&lhs, &rhs_t.as_logical_kn(), m, k, n);
    assert_eq!(got[0].as_f32(), want, "oracle transposed-rhs reference");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p baracuda-kernelgen --test contraction_layout_oracle oracle_matches_emitter_transposed_rhs_f32`
Expected: FAIL — `eval_contraction` is rank-2-canonical-only (`:1282` "no batch/transpose").

- [ ] **Step 3: Generalize `eval_contraction`**

Rewrite the reference loop to index each operand via its `(roles, storage order)` — for output `(batch b, m, n)`, accumulate `Σ_k lhs_at(b,m,k) · rhs_at(b,k,n)` where `lhs_at`/`rhs_at` compute the flat offset via the SAME extent-product binding as the emitter (a Rust mirror of `operand_stride_binding`), K ascending, `f32`/`f64` accumulator per `AccumSpec::WideFloat`. Drop broadcast axes' contribution (stride 0). Keep the epilogue-over-`Reduced(0)` handling unchanged.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p baracuda-kernelgen --test contraction_layout_oracle oracle_matches_emitter_transposed_rhs_f32`
Expected: PASS.

- [ ] **Step 5: Add the full differential matrix**

Add parametrized tests over `{lhs_order ∈ (identity, [1,0])} × {rhs_order ∈ (identity, [1,0])} × {rank-2, rank-3 batched} × {broadcast batch on/off}`, each asserting `oracle::evaluate` on the plan equals a from-scratch reference. (These validate the emitter's binding indirectly through the plan interpreter; a full device run is Task 7.)

- [ ] **Step 6: Run the matrix**

Run: `cargo test -p baracuda-kernelgen --test contraction_layout_oracle`
Expected: PASS (all cells).

- [ ] **Step 7: Commit**

```bash
git add crates/baracuda-kernelgen/src/oracle.rs crates/baracuda-kernelgen/tests/contraction_layout_oracle.rs
git commit -m "feat(oracle): generalize eval_contraction to layout orders + differential matrix"
```

---

### Task 7: On-device validation of headline cells (RTX 4070, via gpu-run)

**Files:**
- Create: `crates/baracuda-kernelgen/ondevice/contract_layout_validate.cu` (mirrors the existing `contract_bias_batched_validate.cu` harness)
- Test: `crates/baracuda-kernelgen/tests/contract_layout_ondevice_smoke.rs` (`#[ignore]`, opt-in)

**Interfaces:**
- Consumes: the emitted kernels (Task 5) + the CPU oracle (Task 6) as the reference.

- [ ] **Step 1: Write the ignored on-device smoke test**

```rust
#[test]
#[ignore = "on-device: requires the RTX 4070 via gpu-run"]
fn contract_transposed_rhs_qkt_matches_oracle_on_device() {
    // Transposed-rhs QKᵀ (rank-3 batched) + GQA broadcast-KV: emit, compile with
    // nvcc, run on the 4070, memcmp the device output against the CPU oracle.
    // Assert bit-diff == 0 across the §A.7 headline cells.
    run_ondevice_contract_layout_matrix(); // builds+launches contract_layout_validate.cu
}
```

- [ ] **Step 2: Run to verify it fails / is skipped**

Run: `pwsh scripts/gpu-run.ps1 -Project baracuda -- cargo test -p baracuda-kernelgen --test contract_layout_ondevice_smoke -- --ignored`
Expected: FAIL (harness `.cu` not yet written) — and confirm the run acquired the `gpu-run` lock (contention log shows the `baracuda` entry).

- [ ] **Step 3: Write the `.cu` harness + launcher**

Author `contract_layout_validate.cu` mirroring `contract_bias_batched_validate.cu`: fill device operands (including a transposed-stored rhs and a batch-broadcast KV with stride 0), launch the generated kernels, copy back, and print per-cell `bit_diff`. The Rust side compiles it (matched MSVC/nvcc env per `cuda-box-local-validation`) and compares against `oracle::evaluate`.

- [ ] **Step 4: Run on device**

Run: `pwsh scripts/gpu-run.ps1 -Project baracuda -- cargo test -p baracuda-kernelgen --test contract_layout_ondevice_smoke -- --ignored`
Expected: PASS — report the POSITIVE passed-count (an all-`#[ignore]` run that prints `0 passed` is a skip-pass, NOT green; gate on the count). `bit_diff == 0` for every headline cell; add a negative control (wrong binding → non-zero diff) to prove the test bites.

- [ ] **Step 5: Sanitizer pass**

Run: `pwsh scripts/gpu-run.ps1 -Project baracuda -- <compute-sanitizer over the harness>`
Expected: 0 errors across memcheck/initcheck/racecheck/synccheck.

- [ ] **Step 6: Commit**

```bash
git add crates/baracuda-kernelgen/ondevice/contract_layout_validate.cu crates/baracuda-kernelgen/tests/contract_layout_ondevice_smoke.rs
git commit -m "test(ondevice): transposed/batched/broadcast contraction bit-exact on RTX 4070"
```

---

## Self-Review

**Spec coverage:**
- §A.2 role legality → Task 3 ✓
- §A.3 layout class + `classify_mat_layout` + additive codec → Tasks 1, 2 ✓
- §A.4 stride-binding emitter (+ worked QKᵀ) → Task 5 ✓
- §A.5 lifted gates (`:1043`, `:366`) + backstop → Tasks 4, 5 ✓
- §A.6 correct-first (no coalescing schedule here; variant deferred) + single-K (Task 3 predicate) ✓
- §A.7 oracle matrix + on-device → Tasks 6, 7 ✓
- §A.8 files → all six files touched across the tasks ✓
- Output-stays-canonical (`out` required-dense in T2, `out` binding unchanged in T5) ✓

**Placeholder scan:** No "TBD"/"handle edge cases". The two helper fixtures (`matmul_opdef_with_rhs_view`, `fill_transposed`) are named test scaffolding the implementer writes in-task; their behavior is specified by the assertions that consume them.

**Type consistency:** `LayoutOrder` (T1) is consumed by `classify_mat_layout` (T2, returns `Option<LayoutOrder>`), stored on `ContractionKey` (T1), read by `operand_stride_binding` (T5) and the oracle mirror (T6). `validate_contraction_roles` signature identical in T3 (def) and T4 (call). `ContractedK`/`FreeM`/`FreeN`/`Batch` roles used consistently.

**Out of scope (guarded):** output views (B), rank-change reshape (C), arbitrary/negative strides (D — `classify_mat_layout` returns `None`, T2 declines-test pins it), multi-K (T3 rejects), TF32/tensor-core (roadmap).
