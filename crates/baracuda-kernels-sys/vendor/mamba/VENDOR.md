# Vendored: state-spaces/mamba — SSD chunk-scan (Phase 50) + selective_scan (Phase 50b)

This directory contains the attribution + provenance metadata for the
**Mamba** family of state-space model kernels authored by Tri Dao +
Albert Gu. Two distinct CUDA op families are covered:

1. **Mamba-2 SSD chunk-scan** (Phase 50) — the chunk-scan + GEMM
   reformulation that maps the selective-SSM recurrence onto matmul-
   friendly primitives. Used by Mamba-2 8B, Codestral-Mamba,
   Falcon-Mamba, Zamba2.
2. **Mamba-1 selective_scan** (Phase 50b) — the original selective
   state-space scan with parallel-scan-friendly fused operator. Used by
   Mamba-7B and the broader Mamba-1 family of shipping models.

## Provenance

- **Upstream**: <https://github.com/state-spaces/mamba>
- **License**: Apache-2.0 (see `LICENSE` next to this file).
- **Papers**:
  - Mamba-1 — "Mamba: Linear-Time Sequence Modeling with Selective
    State Spaces", Gu + Dao, 2023
    (<https://arxiv.org/abs/2312.00752>).
  - Mamba-2 — "Transformers are SSMs", Dao + Gu, 2024
    (<https://arxiv.org/abs/2405.21060>).
- **Vendored**: 2026-05-28 (Phase 50 + Phase 50b on the same date).

## Scope: what we kept

Per baracuda's vendoring discipline, this VENDOR.md + LICENSE +
AUTHORS files preserve attribution. The CUDA implementations live
under `crates/baracuda-kernels-sys/kernels/` (at the baracuda kernel
root, not inside `vendor/`) because baracuda **hand-ports** the
upstream kernels to fit baracuda's dispatch / FFI pattern:

- `kernels/ssd/ssd_chunk_scan_fp.cu` — Phase 50, Mamba-2 SSD FW.
  Hand-port of the upstream Triton SSD reference
  (`mamba_ssm/ops/triton/ssd_chunk_*.py`).
- `kernels/ssd/ssd_chunk_scan_backward_fp.cu` — Phase 50, Mamba-2
  SSD BW.
- `kernels/ssd/selective_scan_fp.cu` — Phase 50b, Mamba-1
  selective_scan FW. Hand-port of upstream
  `csrc/selective_scan/selective_scan_fwd_kernel.cuh`.
- `kernels/ssd/selective_scan_backward_fp.cu` — Phase 50b, Mamba-1
  selective_scan BW. Hand-port of upstream
  `csrc/selective_scan/selective_scan_bwd_kernel.cuh`.

### SSD (Phase 50) algorithm sketch

Upstream's `mamba_ssm/ops/triton/ssd_chunk_*.py` provides the SSD
algorithm decomposition into three phases:

1. **Intra-chunk** — compute outputs within each chunk using the
   semi-separable matrix form (a fused GEMM-like contraction).
2. **State** — compute per-chunk "summary" state by reducing the
   chunk's local recurrence.
3. **Inter-chunk** — propagate chunk states across chunks via a
   sequential recurrence and combine with intra outputs.

Phase 50 ships the dense, fixed-chunk-size CUDA port with
`(B, L, H, D, N)` shape contract. Variable-length sequences,
paged state, and dynamic chunk size are deferred.

### selective_scan (Phase 50b) algorithm sketch

For per-channel `(b, d)`:

```text
For t = 0..L-1:
    dA  = exp(delta[b, t, d] · A[d, n])           [per (d, n)]
    dBu = delta[b, t, d] · B[b, t, n] · u[b, t, d]
    h[b, d, n] = dA · h[b, d, n] + dBu
    y[b, t, d] = Σ_n h[b, d, n] · C[b, t, n]
If D given:  y[b, t, d] += D[d] · u[b, t, d]
If z given:  y[b, t, d] *= silu(z[b, t, d])  (= z · sigmoid(z))
```

Optional knobs:

- `delta_bias: [D]` — added to `delta` before optional softplus.
- `delta_softplus: bool` — applies `log1p(exp(·))` with the standard
  `delta > 20 → delta` overflow short-circuit.

Phase 50b ships **real-only dtypes** (`f32`, `f16`, `bf16`) and the
**dense fixed-length** kernel. Complex selective_scan is reserved in
the upstream code path but no shipping LLM uses it; variable-length
sequences (`cu_seqlens`) + paged state defer to the broader serving
overhaul.

## Scope: what we removed

- **Triton kernels** — baracuda doesn't bring in Triton; we hand-port
  the algorithm to CUDA.
- **PyTorch C++ extension glue** — neither `csrc/selective_scan/` nor
  the SSD Triton bindings are imported as-is. Phase 50b exposes the
  selective_scan FW + BW through flat C FFI
  (`baracuda_kernels_selective_scan_*`); Rust callers reach for
  `SelectiveScanPlan` / `SelectiveScanBackwardPlan`.
- **Python `mamba_ssm/` package** — pure Python orchestration that
  Rust callers compose themselves via baracuda's Plan types.
- **Hopper / sm_90a fast paths** (`csrc/selective_scan/hopper/`) —
  not Phase 50/50b scope; sm_80/sm_89 is target hardware.
- **`causal_conv1d` git submodule** — vendored separately under
  `vendor/causal-conv1d/` (different upstream repo, different
  license: BSD-3-Clause).

## Out of scope for Phase 50b

- **Complex selective_scan**: no shipping Mamba-1 LLM uses it.
- **Variable-length sequences (`cu_seqlens`)**.
- **Paged SSM state** (analog of paged-attention KV cache).
- **Hybrid Mamba + Attention architectures** (Jamba, Zamba) — those
  are caller-side orchestration over baracuda primitives.

## License attribution

baracuda's workspace ships under MIT/Apache-2.0 dual. The Mamba SSD
and selective_scan algorithms originate with Tri Dao + Albert Gu
under Apache-2.0, retained verbatim in `LICENSE` next to this README.
The hand-ported CUDA kernel files under `kernels/ssd/*.cu` carry a
short attribution header pointing at this `VENDOR.md`.
