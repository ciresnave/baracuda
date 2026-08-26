# Convention: group UNCOVERED coverage-matrix rows by LIFETIME

**Scope:** any project that keeps a CI-coverage matrix / audit listing what CI does and does not exercise (crates, features, targets, platforms). Portable — the worked examples are Baracuda's and Fuel's, the rule is not.

**Status:** proposed portable convention. **Merged from two independent derivations and sharpened three times under measurement:** Baracuda's `2026-08-20-ci-coverage-matrix.md` (first split of UNCOVERED; the discriminator; the closing-trap), Fuel's working agreement (multi-cause split; install-over-exclude; the drift mechanism), and two cost distinctions Fuel measured against its own repo (INSTALLABLE ≠ AFFORDABLE, then AFFORDABLE-by-config ≠ UNAVOIDABLE-by-construction). Each round found the previous split repeating its own defect one level down. **Adopt by reference, not by copy** — this file is the canonical home; copying is the divergence mechanism (two copies drift on independent edits).

---

## The rule

An **UNCOVERED** row is not one status. Classify it by its **LIFETIME — the edit that would change it from uncovered to covered** — because the lifetime is the *action*. Five kinds, each naming its own remedy:

| kind | what it is | the edit that closes it | default action |
|------|-----------|-------------------------|----------------|
| **OMISSION** | builds in the current CI env; simply isn't built | a **workflow** edit | close it |
| **INSTALLABLE** | needs a tool/SDK installable on the runner, **cheap** to install+build | a **runner-setup** edit | **install — don't exclude** (before the list drifts) |
| **COST-EXCLUDED** | installable, expensive **by configuration** — a feature/scope you could change, or a cost a cache/faster runner would drop | a **config/cost-model** edit, or a decision to pay | record with the cost; revisit when config or cost changes — a "won't", not a "can't" |
| **UNAVOIDABLE-COST** | expensive **by construction** — a mandatory dep whose build script always runs (`cargo check` runs build scripts), so no configuration removes it | a **code/dependency-structure** edit (make the dep optional; gate the codegen) — a setup step alone will *never* cover it | record as structural; the lever is code, not config or budget |
| **PLATFORM** | needs a runner **class** you lack (OS/arch target; a device needed at **runtime**) | **acquiring a runner** of that class | add a runner, or record uncovered-by-design |

## The recurring lesson: keep splitting until each group has ONE answer

Every coarser version of this taxonomy filed two different answers under one label, and the defect was always the same shape one level down:

- **two-way** (fixable / not) filed "no runner will *ever* have this" beside "no runner has it *yet*";
- **three-way** filed "installable and *cheap*" beside "installable and *ruinous*";
- **four-way** filed "ruinous *by config* (a cache would fix it)" beside "ruinous *by construction* (a mandatory build script; nothing to buy)".

So the stopping rule is mechanical: **group by the edit that closes the row, and split any group whose members have different right answers.** COST-EXCLUDED's answer is "pay it or wait for it to get cheaper"; UNAVOIDABLE-COST's is "there is nothing to buy — change the code or record it." Different answers → different rows. INSTALLABLE would tell a reader of an UNAVOIDABLE-COST row "add a setup step and you're covered," which is false at any budget — that misdirection is the cost of stopping one row too early.

## Why it matters — the failure modes

1. **The label misleads the reader.** Stamping everything UNCOVERED asserts a shared status that isn't there; a *partially*-covered dimension is worse than a wholly uncovered one because the covered part reads as "handled" (a project built 3 of 5 features — `--features metal` present read as "features tested" — while `--features vulkan` broke for 72 min, all green).
2. **The exclusion list grows unopposed.** Each exclusion is *locally* reasonable, so the list drifts into "workspace minus everything with a toolchain" with no step at which anyone objects. INSTALLABLE defaults to install-over-exclude to stop the drift at the first increment.
3. **A "won't" or a structural cost masquerades as a "can't."** Recording COST-EXCLUDED or UNAVOIDABLE-COST as PLATFORM freezes a decision that a cost drop (for the former) or a code change (for the latter) should re-open. The capability exists; only the price or the dependency graph is in the way.

## The discriminator (mechanical)

For each uncovered row: **what is missing, can the runner get it, what does getting it cost, and is that cost configurable or structural?**

- Nothing missing → **OMISSION.**
- Missing but installable and cheap → **INSTALLABLE.**
- Installable but expensive, and the expense is **removable by configuration** (a feature, a scoped build, a cache) → **COST-EXCLUDED.**
- Installable but expensive **by construction** — a mandatory dependency whose build script always runs, so no feature combination avoids it → **UNAVOIDABLE-COST.**
- A runner *class* is missing (OS/arch, or a device at **runtime**) → **PLATFORM.**

Read **build scripts, dependency edges, and whether the dep is `optional`** — not the crate's name. A dependency's `optional = true` vs mandatory is the exact line that separates COST-EXCLUDED from UNAVOIDABLE-COST, and it is one grep of the manifest — but grep the *line*, not a count: a `-c` of a crate name catches the mention in a `# comment` explaining the dep is absent and reports the opposite of the truth.

**Worked example (Baracuda + Fuel, four of the five kinds in two repos):**
- Baracuda's ~61 lazy-libloading `-sys`/wrapper crates: no toolkit needed → **OMISSION** (now covered by a `--all-features` clippy step). The audit first called all 68 "uncovered because nvcc"; only 7 invoke nvcc.
- Baracuda's 7 nvcc crates: nvcc *installs* on ubuntu, so not PLATFORM; but `baracuda-kernels-sys` forges ~426 kernels (~56 min) pulled by workspace membership → **COST-EXCLUDED** (a scoped build or a kernel cache would change the cost).
- Fuel's `fuel-cuda-backend`: `baracuda-kernels-sys` is a **mandatory** dep (`optional` absent), and `cargo check` runs its build script, so **no feature combination compiles it without the forge** → **UNAVOIDABLE-COST**, not COST-EXCLUDED. (Fuel's sibling `fuel-dispatch --features baracuda-types` is a *different target* that pulls no forge — cheap and already covered; the two were fused in one file and only the second is a live question.)
- GPU-**runtime** validation (capture-replay, compute-sanitizer, kernel execution): no GPU runner → **PLATFORM.**

## The trap when you add a step

The new step can be **vacuously green**: if it pulls a COST-EXCLUDED / UNAVOIDABLE-COST / PLATFORM crate whose build script *skips* (rather than fails) when the device/toolkit is absent, it passes while compiling nothing — *in exactly the environment that cannot detect it.* Guard with **assert-the-compiled-set, not the exit code** (`cargo build --message-format json`).

## How to apply

1. Label every UNCOVERED row by the discriminator.
2. Close OMISSIONs.
3. INSTALLABLE: **install over exclude** while it's cheap.
4. COST-EXCLUDED: record the cost, prefer reducing it (scope the build; add a cache) over excluding, revisit when config or cost changes — never relabel PLATFORM.
5. UNAVOIDABLE-COST: record as structural; the only real lever is a code/dep change (make the dep optional, gate the codegen). Do not promise a setup step covers it.
6. PLATFORM: record with the reason, a *declared blindness*.
7. Assert the **compiled set** of any step you add.
8. Prefer a self-maintaining form (`--all-features`) so new items auto-join coverage — backed by a guard that fails when a new item is neither covered nor recorded as one of the cost/platform kinds.

## One-line form

> **UNCOVERED is not one status — classify it by the edit that would close it: a workflow edit (OMISSION), a cheap install (INSTALLABLE), a config or cost change (COST-EXCLUDED — a "won't"), a code change (UNAVOIDABLE-COST — expensive by construction, nothing to buy), or a runner you don't have (PLATFORM). Keep splitting until each row has one answer; filing them under a single "can't" is how a project ends up testing "the workspace minus everything with a toolchain" and calling it a hardware limit.**
