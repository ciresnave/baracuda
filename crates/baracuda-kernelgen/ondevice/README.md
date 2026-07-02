# On-device validation — shared-interior DAG emitter (item 02)

`dag_validate.cu` launches the **generated** diamond kernels — `out = g / (g + 1)`
with `g = a * b`, the shared product hoisted to one `tmp` — and diffs against a
host oracle. It validates the one thing that is catchable only on device: that the
DAG rewrite (emit a shared value once, reference it twice) is a **no-op on the
computed values**, and that the hoisted-`tmp` source compiles and runs.

Two cells exercise both hoist paths:

- `baracuda_gen_diamond_f32_scalar` — `float tmp0 = (in0[i]*in1[i]); out[i] = (tmp0 / (tmp0 + 1.0));`
- `baracuda_gen_diamond_f32_co_v4` — per-lane scoped block `{ float tmp0 = (v0.x*v1.x); vo.x = (tmp0 / (tmp0 + 1.0)); }` (no cross-lane name collision).

## Run

Windows: run from a Visual Studio dev shell so `nvcc` finds `cl.exe`
(`Enter-VsDevShell`), or use an x64 Native Tools prompt.

```sh
# 1. generate the catalog .cu (includes the two diamond cells)
cargo run -p baracuda-kernelgen --bin kernelgen -- <outdir>
# 2. copy this file next to the generated .cu, compile for sm_89, run
cp crates/baracuda-kernelgen/ondevice/dag_validate.cu <outdir>/
nvcc -O3 -arch=sm_89 <outdir>/dag_validate.cu -o <outdir>/dag_validate
<outdir>/dag_validate
```

Expected: `ALL PASSED` (`maxerr 0` — bit-exact; dedup changes text, not values).

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **both cases PASS**
(scalar and per-lane vectorized hoist, bit-exact; PTX `.entry` present under a
headerless `-ptx` compile).

> Manual harness (needs `nvcc` + generated `.cu`), not wired into `cargo test`.
> The `#include`d kernel names track the diamond cells in `bin/kernelgen.rs`;
> update both together. The **fused-reduction epilogue** dedup (Softmax's shared
> `exp(x-max)`), the DAG-based contract flops count, and the `region_to_op` seam
> hash-cons are the item-02 follow-up (see `docs/planning/foundational/`).
