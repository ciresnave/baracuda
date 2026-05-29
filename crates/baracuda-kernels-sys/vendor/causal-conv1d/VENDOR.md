# Vendored: Dao-AILab causal-conv1d (Phase 50)

This directory contains the attribution + provenance metadata for
Tri Dao's `causal-conv1d` primitive, the depthwise-causal 1-D
convolution kernel used by Mamba / Mamba-2 between the input
projection and the SSM block.

## Provenance

- **Upstream**: <https://github.com/Dao-AILab/causal-conv1d>
- **License**: BSD 3-Clause (see `LICENSE` next to this file).
- **Vendored**: 2026-05-28.

## Scope: what we kept

Per baracuda's vendoring discipline, this VENDOR.md + LICENSE +
AUTHORS files preserve attribution. The CUDA implementation lives
under `crates/baracuda-kernels-sys/kernels/conv/causal_conv1d_*.cu`
(at the baracuda kernel root, not inside `vendor/`) because Phase 50
**hand-ports** the upstream algorithm into a single bespoke kernel
that fits baracuda's existing dispatch / launcher pattern, rather
than vendoring the upstream `csrc/causal_conv1d.cu` verbatim.

The algorithmic contract is identical to upstream:

- **Depthwise**: per-channel filter, no cross-channel mixing.
- **Causal**: output at position `t` depends only on `x[t-w+1 .. t]`
  (no future leakage; missing positions zero-padded).
- **Width**: `w ∈ {2, 3, 4}` only (matches upstream's fast path).
  Wider widths require a different kernel and are deferred.
- **Optional bias**: per-channel additive bias.
- **Optional activation**: SiLU (the default Mamba uses) or none.
  Configurable via descriptor.

## Scope: what we removed

- **PyTorch C++ extension glue** (`csrc/causal_conv1d.cpp`, `setup.py`,
  Python bindings) — baracuda exposes the kernel via flat C FFI
  through `baracuda_kernels_causal_conv1d_*` symbols.
- **Variable-length / update-state paths** (`causal_conv1d_update`,
  `causal_conv1d_fwd_varlen`) — Phase 50 ships dense FW + BW only.
  Decode-step `update` is a follow-up.
- **Width-5+ kernels** — defer until a concrete downstream caller
  asks for it.

## Future scope

- **Width-5+** for non-Mamba consumers (TCN, WaveNet, etc.).
- **`causal_conv1d_update`** decode-step kernel for incremental
  inference (single new position, KV-cache-style state).
- **Variable-length batching** (`cu_seqlens` style).

## License attribution

baracuda's workspace ships under MIT/Apache-2.0 dual. The vendored
algorithm originates with Tri Dao under BSD 3-Clause, retained
verbatim in `LICENSE` next to this README. The hand-ported CUDA
kernel files under `kernels/conv/causal_conv1d_*.cu` carry a
short attribution header pointing at this `VENDOR.md`.
