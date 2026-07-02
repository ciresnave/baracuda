# On-device numeric validation — general reduction path (item 03)

`reduce_validate.cu` launches the **generated** general-path reduction kernels
(`_reduce_{tag}_ax{hex}[_kd]`) with small hand-checkable shapes and diffs against a
CPU reference on the GPU. It validates the paths the adversarial review flagged as
catchable **only** on device:

- the emitter↔host **ABI** — `shape[]` / `s0[]` / `so[]` indexing and `n_out`;
- the **keepdim ⇒ `so` by input axis** vs **collapse ⇒ `so` by kept position** split;
- **NaN propagation** in the `Max` `has`-flag fold (torch.amax semantics);
- multi-axis, middle-axis (two kept axes), and reduce-all (kept empty).

## Run

Windows: run from a Visual Studio dev shell so `nvcc` finds `cl.exe`
(`Enter-VsDevShell`), or use an x64 Native Tools prompt.

```
# 1. generate the catalog .cu (includes the 6 general-path reduction cells)
cargo run -p baracuda-kernelgen --bin kernelgen -- <outdir>
# 2. copy this file next to the generated .cu, compile for sm_89, run
cp crates/baracuda-kernelgen/ondevice/reduce_validate.cu <outdir>/
nvcc -arch=sm_89 <outdir>/reduce_validate.cu -o <outdir>/reduce_validate
<outdir>/reduce_validate
```

Expected: `ALL PASSED` (bit-exact, `maxerr 0`; NaN propagated).

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **all 6 cases PASS**.

> This is a manual harness (needs `nvcc` + generated `.cu`), not wired into
> `cargo test`. The `#include`d kernel names track the catalog cells in
> `bin/kernelgen.rs`; update both together.
