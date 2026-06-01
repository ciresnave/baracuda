# FlashAttention v2 — saved-tensor contract (forward → backward)

This guide describes the load-bearing handoff between baracuda's FA2
forward and backward kernels for callers implementing differentiable
attention (autograd frameworks, training loops, gradient-checkpointing
schedulers, etc.).

## TL;DR

```text
                   q, k, v  (operand tensors)
                       │
                       ▼
           ┌─────────────────────────┐
           │  FA2 forward            │
           │  (..._sdpa_<dt>_run_v2) │
           └─────────────────────────┘
                       │
            writes  ▼          ▼
                  out         lse  (f32, [b, h, q])
                       │       │
                       └───┬───┘
                           │  ── SAVE both for backward ──
                           ▼
           ┌─────────────────────────┐
           │  FA2 backward           │
           │  (..._backward_<dt>_run)│
           │                         │
           │  reads: q, k, v, out,   │
           │         dout, lse       │
           │  writes: dq, dk, dv     │
           └─────────────────────────┘
```

LSE (log-sum-exp per query row) is the load-bearing **saved tensor**:
FA2's BW kernel uses it as the numerically-stable softmax normalizer to
reconstruct `P_ij = exp(S_ij − lse_i)` without re-materializing the
attention matrix. Passing a different LSE (e.g. one recomputed from
scratch in the BW pass, or one from a different FW invocation)
silently corrupts the gradient.

## Pre-allocation

```rust
let n_lse_f32 = unsafe {
    baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_lse_size(
        batch, num_heads, seq_q,
    )
};
let mut dev_lse: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_lse_f32)?;
```

The helper returns the element count in f32 elements (multiply by 4 for
bytes). The varlen path has the equivalent
`baracuda_kernels_fa2_sdpa_varlen_lse_size(batch, num_heads, total_q)`.

LSE is **always f32** regardless of the operand dtype. f16 and bf16
operands both produce f32 LSE — FA2 accumulates softmax in f32 to
preserve range across long sequences.

## Forward — write LSE

The v2 forward (`..._sdpa_<dt>_run_v2`) is the recommended entry point
because it carries the full FA2 feature surface (ALiBi, sliding window,
softcap) — the BW kernel reads those same parameters, so using v2 keeps
the FW and BW configurations aligned.

```rust
let fw_status = unsafe {
    baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_f16_run_v2(
        batch, num_heads, num_heads_k, seq_q, seq_k, head_dim,
        softmax_scale,
        is_causal as i32,
        alibi_slopes_ptr,   // *const c_void; null = no ALiBi
        alibi_batch_stride, // 0 = per-head layout; h = per-batch-per-head
        window_size_left,   // -1 = unbounded
        window_size_right,  // -1 = unbounded; causal forces this to 0
        softcap,            // 0.0 = no softcap (typical Gemma-2: 30.0)
        q_ptr, k_ptr, v_ptr,
        out_ptr,
        dev_lse.as_slice_mut().as_raw().0 as *mut c_void,  // ← LSE here
        ws_ptr, ws_bytes,
        stream.as_raw() as *mut c_void,
    )
};
assert_eq!(fw_status, 0);
```

v1 (`..._sdpa_<dt>_run`) also writes LSE — same f32 buffer — but
defaults all ALiBi / sliding-window / softcap params to their disabled
values. If you don't need those features, v1 is fine.

## Save for backward

Stash these alongside the operand tensors that your autograd framework
already saves for backward:

| Field          | dtype     | shape                        | source                          |
|----------------|-----------|------------------------------|---------------------------------|
| `q`, `k`, `v`  | operand   | [B, H{,_k}, S{q,k}, D]      | forward inputs                  |
| `out`          | operand   | [B, H, Sq, D]                | forward output                  |
| `lse`          | **f32**   | [B, H, Sq]                   | **forward output (this guide)** |
| `softmax_scale`| f32       | scalar                       | forward parameter               |
| `is_causal`    | bool      | scalar                       | forward parameter               |
| `alibi_slopes` | f32       | [H] or [B, H] (or absent)    | forward parameter               |
| `window_*`     | i32       | scalar                       | forward parameter               |
| `softcap`      | f32       | scalar                       | forward parameter               |

`alibi_slopes`, sliding-window bounds, and softcap MUST be replayed
into the BW call identically. Mismatched feature flags between FW and
BW produce silently-wrong gradients.

## Backward — read the saved LSE

```rust
let bw_ws_bytes = unsafe {
    baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_backward_workspace_size(
        batch, num_heads, seq_q, head_dim,
    )
};
let mut dev_bw_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, bw_ws_bytes)?;

let bw_status = unsafe {
    baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_backward_f16_run(
        batch, num_heads, num_heads_k, seq_q, seq_k, head_dim,
        softmax_scale,
        is_causal as i32,
        alibi_slopes_ptr,    // identical to FW
        alibi_batch_stride,  // identical to FW
        window_size_left,    // identical to FW
        window_size_right,   // identical to FW
        softcap,             // identical to FW
        q_ptr, k_ptr, v_ptr,
        out_ptr,
        dout_ptr,
        dev_lse.as_slice().as_raw().0 as *const c_void,  // ← saved LSE
        dq_ptr, dk_ptr, dv_ptr,
        dev_bw_ws.as_slice_mut().as_raw().0 as *mut c_void,
        bw_ws_bytes,
        stream.as_raw() as *mut c_void,
    )
};
assert_eq!(bw_status, 0);
```

## Supported dtypes and head_dims

| Path | dtypes  | head_dim                                   |
|------|---------|--------------------------------------------|
| FW   | f16, bf16 | {32, 64, 96, 128, 160, 192, 224, 256, 512} |
| BW   | f16, bf16 | {32, 64, 96, 128, 192, 256}                |

BW does not support hd160 / hd224 / hd512 — FA2's BW algorithm has
structural constraints (`kBlockKSmem=32` atom_layout for hd160/224;
`kBlockM≥64` static_assert for hd512) that upstream FA2 and the Candle
fork both confirm. Callers needing BW at those head_dims must fall back
to the bespoke `SdpaBackwardPlan` (Phase 6 / Milestone 6.6, d_k ≤ 128).

## Determinism note

FA2's BW kernel uses `atomicAdd` into the `dq_accum` workspace, so
gradients are **NOT** bit-stable run-to-run. The bespoke 3-kernel BW
pipeline (`baracuda-kernels`'s default for f32 / f64 / d ≤ 128 f16+bf16
when FA2 isn't selected) IS deterministic. Use the bespoke path when
gradient bit-reproducibility matters (e.g. unit tests that compare
training runs).

## Varlen

The packed-batch (varlen) FA2 surface — `..._varlen_run`,
`..._varlen_backward_<dt>_run` — has the same LSE saved-tensor
contract, but the LSE buffer's layout is different:

```text
varlen LSE: f32 [num_heads, total_q + 128 * batch]
            (unpadded; FA2-internal 128-element-per-batch padding)
```

Use `baracuda_kernels_fa2_sdpa_varlen_lse_size(batch, num_heads, total_q)`
to pre-allocate, save it the same way, pass it back to the BW launcher.

## See also

- `crates/baracuda-kernels-sys/src/lib.rs` — the
  `baracuda_kernels_fa2_sdpa_f16_run` and `..._backward_f16_run`
  docstrings carry the load-bearing `# LSE saved-tensor contract`
  sections.
- `crates/baracuda-kernels/src/attention/flash_sdpa.rs`,
  `flash_sdpa_backward.rs` — the safe-plan layer (`FlashSdpaPlan`,
  `FlashSdpaBackwardPlan`) which wraps the same FFI with typed
  `TensorRef` / `TensorMut`.
- `crates/baracuda-kernels-sys/vendor/flash-attention/VENDOR.md` —
  upstream attribution + the head_dim limitation deep-dive.
