# Convention: group UNCOVERED coverage-matrix rows by LIFETIME

**Scope:** any project that keeps a CI-coverage matrix / audit listing what CI does and does not exercise (crates, features, targets, platforms). Portable — the worked examples are Baracuda's and Fuel's, the rule is not.

**Status:** proposed portable convention, for cross-project adoption. **Merged from two independent derivations plus a cost correction:** Baracuda's `2026-08-20-ci-coverage-matrix.md` (which first split UNCOVERED and supplied the discriminator + the closing-trap), Fuel's working agreement (the multi-cause split, install-over-exclude, and the drift mechanism), and a cost refinement surfaced by Fuel's `fuel-cuda-backend` case against Baracuda's kernel forge (INSTALLABLE ≠ AFFORDABLE). Each carried a level the others lacked; this is the union.

---

## The rule

An **UNCOVERED** row is not one status. Classify it by its **LIFETIME — the edit that would change it from uncovered to covered** — because the lifetime is what tells you the *action*. Four kinds, and each names its own remedy:

| kind | what it needs | the edit that closes it | default action |
|------|---------------|-------------------------|----------------|
| **OMISSION** | nothing missing — it builds in the current CI environment and simply isn't built | a **workflow** edit (add the step) | close it |
| **INSTALLABLE** | a tool/SDK/toolkit that is installable on the runner **and cheap to install+build** (protoc, oneMKL, AOCL, nvcc for a light crate) | a **runner-setup** edit (install it) + the step | **install it — do not exclude** |
| **COST-EXCLUDED** | installable, but the install+build is **prohibitively expensive per run** (e.g. an unconditional kernel forge dragged in by workspace membership) | the **cost dropping** (caching, sccache, a shared build, faster runners) — or a deliberate decision to pay | record as *excluded-for-cost* — a defensible "won't", never dressed as a "can't" |
| **PLATFORM** | a runner **class you do not have** (an OS/arch target; a device — GPU/accelerator — needed at **runtime**) | **acquiring a runner** of that class | add a runner, or record uncovered-by-design |

Only PLATFORM is a fact about the world you can't edit away. The other three are all actionable or decidable — and the whole point is that they routinely get filed under one "can't test this" heading, where nobody revisits them.

## Why fewer categories are not enough — the defect recurs at each level

- A **two-way** split (fixable / not) files "no runner will *ever* have this hardware" next to "no runner has this installed *yet*" — different lifetimes, one label.
- A **three-way** split (OMISSION / INSTALLABLE / PLATFORM) then files "installable and *cheap*" next to "installable and *ruinous*" — same remedy *shape* (install + build), opposite answers. Fuel excludes `fuel-cuda-backend` though the toolkit installs fine, because building it is not free; that exclusion is a **cost** decision, and its lifetime is the cost, not the install.

The recurring lesson: **group by the edit that closes the row, and keep splitting until each group has a single answer.** A category whose members have different right answers is still a conflation.

## Why it matters — three failure modes

1. **The label misleads the reader.** A matrix that stamps everything UNCOVERED tells the reader they share a status they don't; a *partially*-covered dimension is worse than a wholly uncovered one, because the covered part reads as "handled." (A project whose CI built three of five features had `--features metal` present — which read as "features are tested" — while untested `--features vulkan` broke for 72 minutes with every check green.)
2. **The exclusion list grows unopposed.** Each exclusion is *locally* reasonable — "this needs a toolchain, skip it" — so the list drifts into "workspace minus everything with a toolchain" **with no step at which anyone objects.** This is why INSTALLABLE defaults to install-over-exclude: it stops the drift at the first increment.
3. **A "won't pay" masquerades as a "can't."** Recording a COST-EXCLUDED row as PLATFORM freezes a decision that should be revisited every time the cost model changes (a build cache lands, runners get faster). The capability exists; only the price is in the way.

## The discriminator (mechanical, not a judgement call)

For each uncovered row, ask **what is missing, can the runner get it, and what does getting it cost?**

- Nothing missing — builds in the current env → **OMISSION.**
- A tool/SDK is missing but **installable and cheap** → **INSTALLABLE.**
- Installable but the build is **ruinous per run** → **COST-EXCLUDED.**
- A runner *class* is missing — different OS/arch, or a device needed at **runtime** → **PLATFORM.**

Read the **build scripts and dependency edges**, not the crate's name. **Worked example (Baracuda, corrected twice as this convention sharpened):**
- First pass: "68 crates uncovered because they compile `.cu` via nvcc." **Wrong** — only 7 invoke nvcc; the other ~61 are lazy-libloading and build with no toolkit at all → **OMISSION** (now covered by a `--all-features` clippy step).
- Second pass: "the 7 need CUDA, so PLATFORM." **Also wrong** — nvcc *installs* on an ubuntu runner, so the 7 are not platform-bound to build.
- Correct: the 7 nvcc crates are **COST-EXCLUDED** — `baracuda-kernels-sys` alone forges ~426 kernels (~56 min) that workspace membership drags in, so a build job is installable but not free. Only the GPU-**runtime** validation (capture-replay, compute-sanitizer, kernel execution) is genuinely **PLATFORM**. Three of the four kinds appear in one repo, and each earlier flattening cost a coverage decision that had a real, different answer.

## The trap when you close an OMISSION or add an INSTALLABLE step

The new step can be **vacuously green.** If it pulls a COST-EXCLUDED or PLATFORM crate whose build script *skips* (rather than fails) when the device/toolkit is absent, the step passes while compiling nothing — *in exactly the environment that cannot detect it.* Guard with **assert-the-compiled-set, not the exit code** (`cargo build --message-format json` names what actually compiled): a step that pulled in a skipped forge is distinguishable from one that compiled the intended code only by reading what compiled.

## How to apply

1. Label every UNCOVERED row `OMISSION` / `INSTALLABLE` / `COST-EXCLUDED` / `PLATFORM` by the discriminator.
2. Close OMISSIONs (add the step).
3. For INSTALLABLE, **prefer installing over excluding** whenever the tool installs cheaply — an exclusion is a permanent coverage loss dressed as a convenience, and the list drifts.
4. For COST-EXCLUDED, record it as *excluded-for-cost* with the measured cost, and revisit when the cost model changes (a build cache, faster runners) — never relabel it PLATFORM.
5. Record each PLATFORM with its reason, a *declared blindness* a future reader won't "fix" with a step that then skips-green.
6. For any step you add, assert its **compiled set** so it can't go vacuous.
7. Prefer a self-maintaining form (`--all-features` over an enumerated list) so a new item auto-joins coverage — backed by a guard that fails when a new item is neither covered nor recorded as COST-EXCLUDED/PLATFORM.

## One-line form

> **UNCOVERED is not one status — classify it by the edit that would close it: a workflow edit (OMISSION), a cheap install (INSTALLABLE — and install-over-exclude before the list drifts), a cost that hasn't dropped yet (COST-EXCLUDED — a "won't pay", not a "can't"), or a runner you don't have (PLATFORM). Filing the first three under the last is how a project ends up testing "the workspace minus everything with a toolchain" and calling it a hardware limit.**
