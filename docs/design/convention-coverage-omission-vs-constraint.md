# Convention: a coverage matrix must split OMISSION from CONSTRAINT

**Scope:** any project that keeps a CI-coverage matrix / audit listing what CI does and does not exercise (crates, features, targets, platforms). Portable — the worked example is Baracuda's, the rule is not.

**Status:** proposed portable convention, for cross-project adoption. Origin: Baracuda's `2026-08-20-ci-coverage-matrix.md`, which conflated the two and corrected it under review.

---

## The rule

Every row a coverage matrix marks **UNCOVERED** is one of two kinds, and they must be labelled differently:

- **CONSTRAINT — "needs-X-can't."** The code *cannot* be exercised in the available CI environment: it requires hardware, a toolkit, a credential, or a service that no runner has. Uncovered by a fact about the world. **Not fixable by adding a step** — a step would fail (or worse, silently skip). Correct disposition: *record it as uncovered-by-design, with the reason.*
- **OMISSION — "could-do-doesn't."** The code *can* be exercised in the available environment and simply isn't. Uncovered by a gap in the workflow. **Fixable** — add the step. Correct disposition: *close it, or the matrix is documenting a gap instead of holding a gate.*

## Why the distinction is load-bearing, not cosmetic

A matrix that stamps both **UNCOVERED** tells the reader they have the same status. They do not: one is actionable and one is not, and **lumping them hides which.** The reader budgets around a hardware constraint and never notices that most of the "constraint" was actually omission they could have closed this afternoon.

The failure mode is concrete. A partially-covered dimension is **more dangerous than a wholly uncovered one**, because the covered part signals to a reader that the dimension is *handled*. (Live instance: a project whose CI built three of five build-features had `--features metal` present in the workflow — which read as "features are tested" — while the untested `--features vulkan` broke a binding for 72 minutes with every check green. The presence of *some* feature steps is exactly what stopped anyone looking for the missing one.)

## The discriminator (mechanical, not a judgement call)

Ask of each uncovered row: **does exercising it require the absent thing to BUILD, or merely to RUN — and is it actually built at all?**

- Its build script invokes the missing toolkit (nvcc/cc against unavailable headers), or it link-depends on the absent library → **CONSTRAINT.**
- It builds without the missing thing (e.g. resolves the library lazily at runtime, or is pure-source) and just isn't in any CI step → **OMISSION.**

Read the *build scripts and dependency edges*, do not assume from the crate's name or subject. Baracuda's audit first wrote "68 crates uncovered because they compile `.cu` via nvcc"; measuring the build scripts showed **only 7 actually invoke nvcc** — the other ~61 are lazy-libloading and were OMISSION mislabelled as CONSTRAINT. A whole CI-buildable surface was hidden behind a wrong word.

## The trap when you DO add the omitted step

An OMISSION step can be **vacuously green**. If the step pulls a CONSTRAINT crate whose build script *skips* (rather than fails) when the toolkit is absent, the step passes while compiling nothing — and it does so *in exactly the environment that cannot tell the difference*, because that environment is the one without the toolkit. Guard against it: **assert the compiled set, not the exit code** (`cargo build --message-format json` names what actually compiled). A step that pulls in a skipped forge is distinguishable from one that compiled the intended code only by reading what compiled.

## How to apply

1. Label every UNCOVERED row `CONSTRAINT` or `OMISSION` by the discriminator above.
2. Close the OMISSIONs (add the step) or state why not — an OMISSION left open is a gate you decided not to build, and should read that way, not as a hardware limit.
3. Record each CONSTRAINT with its reason, so the absence is a *declared blindness* a future reader won't "fix" by adding a step that then skips-green.
4. If you add an OMISSION step, assert its **compiled set** so it can't go vacuous.
5. Prefer a self-maintaining form where possible (e.g. `--all-features` over an enumerated feature list) so a newly-added item auto-joins coverage instead of silently rejoining the uncovered set — and back it with a guard that fails when a new item is neither covered nor recorded as CONSTRAINT.

## One-line form

> **UNCOVERED is two words wearing one label. "Can't build it here" is a constraint you record; "don't build it here" is an omission you close. A matrix that can't tell them apart is telling the reader the fixable gap is a law of physics.**
