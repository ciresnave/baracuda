# API freeze — public-surface inventory (first pass)

Backs the ROADMAP item "API freeze + 1.0 stability review". Measured at
`origin/main` `fc1a97de` (alpha.94). This is an **inventory**, not the
categorisation the roadmap item asks for: nothing here decides stable /
`#[doc(hidden)]` / remove.

## Method (and its limit)

`git ls-files` over `crates/*/src/**/*.rs`, counting lines that start with
`pub` followed by `fn|struct|enum|trait|type|const|static|mod|union|use`
(optionally `unsafe`/`async`), plus `#[doc(hidden)]`, `#[deprecated` and
`pub(crate)` lines. **It undercounts**: it sees neither `pub` struct fields nor
items emitted by macros, and it counts `pub use` re-exports as items. Treat the
numbers as a ranking, not a census. 74 crates matched.

## Where the surface is

| Crate | `pub` items | `doc(hidden)` | `deprecated` |
|---|---:|---:|---:|
| baracuda-kernels-sys | 5775 | 0 | 0 |
| baracuda-kernels | 3466 | 1 | 7 |
| baracuda-cuda-sys | 3270 | 0 | 0 |
| baracuda-driver | 570 | 2 | 0 |
| baracuda-runtime | 381 | 0 | 0 |
| baracuda-cudnn-sys | 210 | 0 | 0 |
| baracuda-cutlass-kernels-sys | 174 | 0 | 0 |
| baracuda-cudnn | 159 | 0 | 0 |
| 66 other crates | 1 – 154 each | 50 across 7 crates | 2 |

Workspace total by this method: **17 276 items, 53 `doc(hidden)`, 9
`deprecated`, 119 `pub(crate)`**. Three crates (`-kernels-sys`, `-kernels`,
`-cuda-sys`) hold 12 511 of them (72%). Only `-kernels-sys` and `-kernels` are
ours to decide; `-cuda-sys` mirrors the CUDA driver API.

Reading: the freeze decision is overwhelmingly about the two kernels crates, and
`doc(hidden)` is barely used (1 in `-kernels`), so "what is hidden today" is
close to "nothing" — every `pub` item there is currently a promise.

## The `Element` vs `DeviceRepr + Copy` split (roadmap sub-item)

Resolved by reading, not by a change:

- `Element` is **not defined in baracuda.** It is `unpopped_vocab::Element`,
  re-exported through `baracuda-kernels-types` (`lib.rs:40`,
  `pub use unpopped_vocab::*`).
- In `unpopped-vocab` (`element.rs`): `Element: KernelDtype + sealed::Sealed`
  and `KernelDtype: DeviceRepr + sealed + Copy + 'static`. So `Element` is a
  **sealed subtrait** of the `DeviceRepr + Copy + 'static` bound, not a rival.
- Baracuda's use matches that layering: carriers (`TensorRef`, `TensorMut`,
  `MatrixRef`, `MatrixMut`, `VectorRef` in `baracuda-kernels-types`) and the
  layout plans (`ContiguizePlan`, `WriteSlicePlan`, `CastSubBytePlan`,
  `GgufMmvqBatchedPlan`) bound on `DeviceRepr + Copy + 'static` so sub-byte /
  packed types can ride them; compute plans bound on `T: Element`
  (919 lines in `baracuda-cutlass`, `baracuda-kernels`,
  `baracuda-kernels-bench`).
- Two crates define their own scalar traits (`CudnnDataType`,
  `SolverScalar`), both `DeviceRepr + Copy + 'static`.

**Consequence for the freeze:** the split is intentional and one half of it is
Unpopped's. Unifying "under a new umbrella trait" would be an Unpopped change.
The baracuda-side action is only to document the layering, and to treat
`Element` / `KernelDtype` as an Unpopped-owned stability dependency of every
`Plan<T>` signature.

## Breaking-change policy (draft input, not adopted)

Facts already fixed by CireSnave's standing rules (`C:\Projects\CLAUDE.md` §9),
which a policy must not contradict:

- A pushed change changes the version; a breaking change moves the major.
- **Pre-1.0 the major is the second number** (0.5.x -> 0.6.0, never 1.0.0).
  This workspace is `0.0.1-alpha.N` lockstep, where N is the moving counter.
- One version number per project, except crates that must match another
  project's version (`baracuda-cuda-emit` / `-cuda-parse` follow `unpopped`).
- The PM allocates the number at gate time.

Open questions a policy needs answered (none decided here): what counts as
breaking at the C-ABI layer of `-kernels-sys` (symbol added / removed / struct
layout), and whether adding a variant to a `#[non_exhaustive]` enum is breaking.

## Not done

- No per-item stable / hidden / remove classification.
- `-kernels-sys` symbol-level stability (5775 items) is untouched.
- The `0.1.0-beta.0` cut is a CireSnave/PM decision, not made here.
