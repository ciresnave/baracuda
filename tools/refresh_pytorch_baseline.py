"""Generate PyTorch reference timings for baracuda's bench suite.

Phase 73.1 — replaces the abandoned tch-rs in-process integration with a
frozen-on-disk baseline. Runs each op + shape + dtype combo under
PyTorch's CUDA path with CUDA-event timing (same methodology as
baracuda-kernels-bench's `measure_median_ns`), dumps medians to a JSON
file checked into the repo under
`crates/baracuda-kernels-bench/bench-baselines/`.

The bench harness reads this JSON at process start and emits the PyTorch
median into the `pytorch_ns` column of `PhaseTwentyNineRow`. No PyTorch
dep at build or test time — refresh is an out-of-band step, run when
PyTorch updates materially or when adding new ops to the bench suite.

Usage:
    python tools/refresh_pytorch_baseline.py [--ops OP1,OP2,...]
                                              [--output PATH]
                                              [--samples N] [--inner N]

If `--output` is omitted, the script picks a filename based on detected
device + PyTorch version + CUDA version:

    bench-baselines/pytorch_<device_slug>_<torch_version>_cu<cuda>.json

Example:
    bench-baselines/pytorch_rtx4070_2.11.0_cu130.json

Notes on timing methodology:

  - 10-launch warmup (matches baracuda's `warmup()`).
  - `samples` independent batches, each containing `inner` launches.
  - Wall-clock time around `torch.cuda.synchronize()` for each batch.
  - Report the median of per-batch averages (matches baracuda's
    `measure_median_ns`).

Defaults (`samples=11`, `inner=50`) are conservative — same as bench
harness — and add ~550 launches per (op, shape, dtype) cell. For the
~10 ops × ~10 shapes × ~3 dtypes covered in Phase 29 that's ~165k
launches, roughly 3-5 minutes on RTX 4070.

Ops covered (Phase 73.2 fanout):

  - gemm:        `torch.matmul`                            × {f32, f16, bf16}
  - softmax:     `F.softmax(..., dim=-1)`                  × {f32, f16}
  - layernorm:   `F.layer_norm(..., normalized_shape=[H])` × {f32, f16}
  - rmsnorm:     manual `x / sqrt(mean(x^2) + eps) * w`    × {f32, f16, bf16}
  - reduce_sum:  `torch.sum(x, dim=-1)`                    × {f32}
  - reduce_max:  `torch.amax(x, dim=-1)`                   × {f32}
  - reduce_mean: `torch.mean(x, dim=-1)`                   × {f32}
  - add:         `a + b` elementwise                       × {f32, f16}
  - mul:         `a * b` elementwise                       × {f32, f16}
  - relu:        `F.relu(x)`                               × {f32, f16}
  - gelu:        `F.gelu(x)`           (exact, erf-based)  × {f32, f16}
  - conv2d:      `F.conv2d(x, w, padding=k//2)`            × {f32, f16}
  - maxpool2d:   `F.max_pool2d(x, k, stride, pad)`         × {f32, f16}
  - flash_sdpa_gqa:
                 `F.scaled_dot_product_attention(q,k,v, is_causal=True)`
                                                           × {f16, bf16}

mmvq (GGUF-quantized matrix-vector) intentionally skipped — PyTorch has
no direct equivalent op; baseline would need a separate design.

Shape constants mirror `crates/baracuda-kernels-bench/src/lib.rs`
(CROSS_GEMM_*, CROSS_SEQLEN_SWEEP, CROSS_HIDDEN_SWEEP, CONV2D_SWEEP,
POOL_SWEEP, sdpa_gqa.rs constants).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import torch


# ---------------------------------------------------------------------
# Shape sweep — kept in sync with `baracuda-kernels-bench/src/lib.rs`.
# ---------------------------------------------------------------------

# Matches `CROSS_GEMM_M_SWEEP`. `M=1` covers decode, `M=32 / 128` covers
# prefill at LLM-typical batch sizes.
GEMM_M_SWEEP: tuple[int, ...] = (1, 32, 128)
# Matches `CROSS_GEMM_KN_SWEEP`. 7B hidden = 4096; smaller 2048 picks up
# the K-bound regime.
GEMM_KN_SWEEP: tuple[int, ...] = (2048, 4096)

# Dtypes to cover per op. `f32` everywhere; `f16` / `bf16` for ops with
# tensor-core paths.
GEMM_DTYPES: tuple[tuple[str, torch.dtype], ...] = (
    ("f32", torch.float32),
    ("f16", torch.float16),
    ("bf16", torch.bfloat16),
)

# `CROSS_SEQLEN_SWEEP` / `CROSS_HIDDEN_SWEEP` from lib.rs — used by
# softmax / layernorm / rmsnorm / reductions.
NORM_R_SWEEP: tuple[int, ...] = (512, 2048, 4096)
NORM_H_SWEEP: tuple[int, ...] = (1024, 4096)

# Norm-family dtypes. softmax + layernorm cover f32 / f16; rmsnorm
# also covers bf16 (the bench has the wider sweep).
NORM_DTYPES: tuple[tuple[str, torch.dtype], ...] = (
    ("f32", torch.float32),
    ("f16", torch.float16),
)
RMSNORM_DTYPES: tuple[tuple[str, torch.dtype], ...] = (
    ("f32", torch.float32),
    ("f16", torch.float16),
    ("bf16", torch.bfloat16),
)

# `ELT_SWEEP` from elementwise.rs.
ELT_NUMELS: tuple[int, ...] = (1 << 20, 1 << 24)
ELT_DTYPES: tuple[tuple[str, torch.dtype], ...] = (
    ("f32", torch.float32),
    ("f16", torch.float16),
)

# `CONV2D_SWEEP` from lib.rs — (n, c_in, c_out, hw, k) tuples. Only
# square shapes (h == w) are bench-covered.
CONV2D_SWEEP: tuple[tuple[int, int, int, int, int], ...] = (
    (1, 64, 64, 56, 3),
    (1, 128, 128, 28, 3),
    (1, 256, 256, 14, 3),
)
CONV2D_DTYPES = NORM_DTYPES  # f32, f16

# `POOL_SWEEP` from lib.rs — (n, c, h, w, k, stride, pad).
POOL_SWEEP: tuple[tuple[int, int, int, int, int, int, int], ...] = (
    (1, 64, 56, 56, 3, 2, 1),
    (1, 128, 28, 28, 3, 2, 1),
    (1, 256, 14, 14, 3, 2, 1),
)
POOL_DTYPES = NORM_DTYPES  # f32, f16

# `sdpa_gqa.rs` constants.
SDPA_BATCH: int = 1
SDPA_NUM_Q_HEADS: int = 32
SDPA_KV_HEAD_SWEEP: tuple[int, ...] = (32, 8, 4, 1)
SDPA_SEQ_LEN: int = 2048
SDPA_HEAD_DIM: int = 128
SDPA_DTYPES: tuple[tuple[str, torch.dtype], ...] = (
    ("f16", torch.float16),
    ("bf16", torch.bfloat16),
)


# ---------------------------------------------------------------------
# Timing helpers — match baracuda's measure_median_ns methodology.
# ---------------------------------------------------------------------

WARMUP_LAUNCHES: int = 10


def time_median_ns(
    launch: Callable[[], None],
    samples: int,
    inner: int,
) -> float:
    """Median per-launch wall-clock ns across `samples` batches of `inner`."""
    # Warmup — let GPU clock + caches settle. Match `warmup()` in
    # `baracuda-kernels-bench/src/lib.rs`.
    for _ in range(WARMUP_LAUNCHES):
        launch()
    torch.cuda.synchronize()

    per_launch_ns: list[float] = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(inner):
            launch()
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        per_launch_ns.append((end - start) / inner)

    return statistics.median(per_launch_ns)


# ---------------------------------------------------------------------
# Per-op generators — yield (op, shape, dtype, launch_fn).
# ---------------------------------------------------------------------


def gemm_cases() -> Iterable[tuple[str, str, str, Callable[[], None]]]:
    """All (op, shape, dtype, launch) tuples for GEMM."""
    device = torch.device("cuda")
    for kn in GEMM_KN_SWEEP:
        for m in GEMM_M_SWEEP:
            n = kn
            k = kn
            shape = f"M{m}_N{n}_K{k}"
            for dtype_name, dtype in GEMM_DTYPES:
                # Bind loop vars into the closure (Python late-binding gotcha).
                # Tensors live for the duration of the cell — torch keeps them
                # alive via the closure capture; freed at next cell.
                a = torch.ones((m, k), dtype=dtype, device=device)
                b = torch.ones((k, n), dtype=dtype, device=device)
                launch = lambda a=a, b=b: torch.matmul(a, b)
                yield ("gemm", shape, dtype_name, launch)


def softmax_cases() -> Iterable[tuple[str, str, str, Callable[[], None]]]:
    """`F.softmax(x, dim=-1)` over CROSS_SEQLEN × CROSS_HIDDEN."""
    device = torch.device("cuda")
    for rows in NORM_R_SWEEP:
        for cols in NORM_H_SWEEP:
            shape = f"R{rows}_C{cols}"
            for dtype_name, dtype in NORM_DTYPES:
                x = torch.randn((rows, cols), dtype=dtype, device=device)
                launch = lambda x=x: torch.nn.functional.softmax(x, dim=-1)
                yield ("softmax", shape, dtype_name, launch)


def log_softmax_cases() -> Iterable[tuple[str, str, str, Callable[[], None]]]:
    """`F.log_softmax(x, dim=-1)` over CROSS_SEQLEN × CROSS_HIDDEN.

    Phase 73.4 addition — sibling to `softmax_cases`. Same shapes, same
    dtypes; the bench harness uses the same `R{rows}_C{cols}` shape key
    so the lookup matches `softmax_vs_cudnn.rs`'s LogSoftmax pass.
    """
    device = torch.device("cuda")
    for rows in NORM_R_SWEEP:
        for cols in NORM_H_SWEEP:
            shape = f"R{rows}_C{cols}"
            for dtype_name, dtype in NORM_DTYPES:
                x = torch.randn((rows, cols), dtype=dtype, device=device)
                launch = lambda x=x: torch.nn.functional.log_softmax(x, dim=-1)
                yield ("log_softmax", shape, dtype_name, launch)


def layernorm_cases() -> Iterable[tuple[str, str, str, Callable[[], None]]]:
    """`F.layer_norm(x, normalized_shape=[H])`."""
    device = torch.device("cuda")
    for rows in NORM_R_SWEEP:
        for hidden in NORM_H_SWEEP:
            shape = f"R{rows}_H{hidden}"
            for dtype_name, dtype in NORM_DTYPES:
                x = torch.randn((rows, hidden), dtype=dtype, device=device)
                w = torch.ones((hidden,), dtype=dtype, device=device)
                b = torch.zeros((hidden,), dtype=dtype, device=device)
                launch = lambda x=x, w=w, b=b, h=hidden: torch.nn.functional.layer_norm(
                    x, normalized_shape=[h], weight=w, bias=b
                )
                yield ("layernorm", shape, dtype_name, launch)


def rmsnorm_cases() -> Iterable[tuple[str, str, str, Callable[[], None]]]:
    """`y = x / sqrt(mean(x^2) + eps) * w`. PyTorch 2.11 may or may not
    have `F.rms_norm` (it became stable in 2.4). Use the manual form for
    cross-version determinism."""
    device = torch.device("cuda")
    eps = 1e-5
    for rows in NORM_R_SWEEP:
        for hidden in NORM_H_SWEEP:
            shape = f"R{rows}_H{hidden}"
            for dtype_name, dtype in RMSNORM_DTYPES:
                x = torch.randn((rows, hidden), dtype=dtype, device=device)
                w = torch.ones((hidden,), dtype=dtype, device=device)
                def launch(x=x, w=w):
                    # Match baracuda's RMSNorm: pop. mean of x^2 along
                    # last axis; cast to f32 accumulator if dtype is half.
                    upcast = x.float() if x.dtype != torch.float32 else x
                    rms = upcast.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
                    y = (upcast / rms).to(x.dtype) * w
                    return y
                yield ("rmsnorm", shape, dtype_name, launch)


def reduce_cases() -> Iterable[tuple[str, str, str, Callable[[], None]]]:
    """`torch.sum/amax/amin/mean/prod/var/std/norm/logsumexp(x, dim=-1)` over
    CROSS_SEQLEN × CROSS_HIDDEN."""
    device = torch.device("cuda")
    fns: tuple[tuple[str, Callable[[torch.Tensor], torch.Tensor]], ...] = (
        ("reduce_sum", lambda t: torch.sum(t, dim=-1)),
        ("reduce_max", lambda t: torch.amax(t, dim=-1)),
        ("reduce_mean", lambda t: torch.mean(t, dim=-1)),
        # Phase 73.4: Prod completes the basic reduce set.
        ("reduce_prod", lambda t: torch.prod(t, dim=-1)),
        # Phase 73.6: Min + statistical / L2 reductions.
        ("reduce_min", lambda t: torch.amin(t, dim=-1)),
        # Bessel-corrected sample variance / std (correction=1) to match
        # baracuda's `correction: 1` in the bench.
        ("reduce_var", lambda t: torch.var(t, dim=-1, unbiased=True)),
        ("reduce_std", lambda t: torch.std(t, dim=-1, unbiased=True)),
        ("reduce_norm2", lambda t: torch.linalg.vector_norm(t, ord=2, dim=-1)),
        ("reduce_logsumexp", lambda t: torch.logsumexp(t, dim=-1)),
    )
    for op_name, fn in fns:
        for rows in NORM_R_SWEEP:
            for hidden in NORM_H_SWEEP:
                shape = f"R{rows}_H{hidden}"
                x = torch.randn((rows, hidden), dtype=torch.float32, device=device)
                launch = lambda x=x, fn=fn: fn(x)
                yield (op_name, shape, "f32", launch)


def elementwise_cases() -> Iterable[tuple[str, str, str, Callable[[], None]]]:
    """Comprehensive elementwise coverage — binary ops, activations, math
    unaries. Phase 73.5 extended the original 4 ops (add/mul/relu/gelu)
    with the full set listed below."""
    device = torch.device("cuda")
    binary_ops: tuple[tuple[str, Callable], ...] = (
        ("add", lambda a, b: a + b),
        ("mul", lambda a, b: a * b),
        # Phase 73.5: additional binaries.
        ("sub", lambda a, b: a - b),
        ("div", lambda a, b: a / b),
        ("maximum", lambda a, b: torch.maximum(a, b)),
        ("minimum", lambda a, b: torch.minimum(a, b)),
        ("pow", lambda a, b: torch.pow(a, b)),
    )
    unary_ops: tuple[tuple[str, Callable], ...] = (
        ("relu", lambda x: torch.nn.functional.relu(x)),
        # baracuda's gelu uses the exact erf-based formulation; match PyTorch
        # default (approximate='none').
        ("gelu", lambda x: torch.nn.functional.gelu(x, approximate="none")),
        # Phase 73.4: Silu (Llama-family) + classical Tanh/Sigmoid.
        ("silu", lambda x: torch.nn.functional.silu(x)),
        ("tanh", lambda x: torch.tanh(x)),
        ("sigmoid", lambda x: torch.sigmoid(x)),
        # Phase 73.5: additional activations.
        ("mish", lambda x: torch.nn.functional.mish(x)),
        ("hardswish", lambda x: torch.nn.functional.hardswish(x)),
        ("hardsigmoid", lambda x: torch.nn.functional.hardsigmoid(x)),
        ("hardtanh", lambda x: torch.nn.functional.hardtanh(x)),
        ("leaky_relu", lambda x: torch.nn.functional.leaky_relu(x)),
        ("elu", lambda x: torch.nn.functional.elu(x)),
        ("selu", lambda x: torch.nn.functional.selu(x)),
        ("relu6", lambda x: torch.nn.functional.relu6(x)),
        ("softplus", lambda x: torch.nn.functional.softplus(x)),
        ("softsign", lambda x: torch.nn.functional.softsign(x)),
        ("gelu_tanh", lambda x: torch.nn.functional.gelu(x, approximate="tanh")),
        # Phase 73.5: basic math unaries.
        ("abs", lambda x: torch.abs(x)),
        ("neg", lambda x: torch.neg(x)),
        ("sign", lambda x: torch.sign(x)),
        ("reciprocal", lambda x: torch.reciprocal(x)),
        ("sqrt", lambda x: torch.sqrt(x)),
        ("rsqrt", lambda x: torch.rsqrt(x)),
        ("square", lambda x: torch.square(x)),
        ("exp", lambda x: torch.exp(x)),
        ("log", lambda x: torch.log(x)),
        ("sin", lambda x: torch.sin(x)),
        ("cos", lambda x: torch.cos(x)),
        ("erf", lambda x: torch.erf(x)),
    )
    for n in ELT_NUMELS:
        shape = f"N{n}"
        for dtype_name, dtype in ELT_DTYPES:
            a = torch.ones((n,), dtype=dtype, device=device)
            b = torch.ones((n,), dtype=dtype, device=device)
            for op_name, fn in binary_ops:
                launch = lambda a=a, b=b, fn=fn: fn(a, b)
                yield (op_name, shape, dtype_name, launch)
            for op_name, fn in unary_ops:
                launch = lambda a=a, fn=fn: fn(a)
                yield (op_name, shape, dtype_name, launch)


def conv2d_cases() -> Iterable[tuple[str, str, str, Callable[[], None]]]:
    """`F.conv2d(x, w, padding=k//2)` over CONV2D_SWEEP. NCHW layout."""
    device = torch.device("cuda")
    for (n, c_in, c_out, hw, k) in CONV2D_SWEEP:
        shape = f"N{n}_Cin{c_in}_Cout{c_out}_HW{hw}_K{k}"
        for dtype_name, dtype in CONV2D_DTYPES:
            x = torch.randn((n, c_in, hw, hw), dtype=dtype, device=device)
            weight = torch.randn((c_out, c_in, k, k), dtype=dtype, device=device)
            launch = lambda x=x, w=weight, p=k // 2: torch.nn.functional.conv2d(
                x, w, padding=p
            )
            yield ("conv2d", shape, dtype_name, launch)


def maxpool2d_cases() -> Iterable[tuple[str, str, str, Callable[[], None]]]:
    """`F.max_pool2d(x, kernel_size=k, stride=s, padding=p)` over POOL_SWEEP."""
    device = torch.device("cuda")
    for (n, c, h, w, k, s, p) in POOL_SWEEP:
        shape = f"N{n}_C{c}_H{h}_W{w}_K{k}_S{s}"
        for dtype_name, dtype in POOL_DTYPES:
            x = torch.randn((n, c, h, w), dtype=dtype, device=device)
            launch = lambda x=x, k=k, s=s, p=p: torch.nn.functional.max_pool2d(
                x, kernel_size=k, stride=s, padding=p
            )
            yield ("maxpool2d", shape, dtype_name, launch)


def avgpool2d_cases() -> Iterable[tuple[str, str, str, Callable[[], None]]]:
    """`F.avg_pool2d(x, kernel_size=k, stride=s, padding=p, count_include_pad=True)`
    over POOL_SWEEP. `count_include_pad=True` matches both PyTorch's default
    AND the cuDNN `AverageCountIncludePadding` mode the bench uses, so the
    three-way comparison is apples-to-apples on the denominator semantics."""
    device = torch.device("cuda")
    for (n, c, h, w, k, s, p) in POOL_SWEEP:
        shape = f"N{n}_C{c}_H{h}_W{w}_K{k}_S{s}"
        for dtype_name, dtype in POOL_DTYPES:
            x = torch.randn((n, c, h, w), dtype=dtype, device=device)
            launch = lambda x=x, k=k, s=s, p=p: torch.nn.functional.avg_pool2d(
                x, kernel_size=k, stride=s, padding=p, count_include_pad=True
            )
            yield ("avgpool2d", shape, dtype_name, launch)


# `concat.rs` shape sweep — (BH, Ka, Kb, D) tuples. KV-cache decode shape
# + mid-sequence joins. Must match CONCAT_SWEEP in benches/concat.rs.
CONCAT_SWEEP: tuple[tuple[int, int, int, int], ...] = (
    (32, 2047, 1, 128),
    (32, 1024, 1024, 128),
    (32, 512, 512, 128),
)
CONCAT_DTYPES = NORM_DTYPES  # f32, f16


EMBEDDING_SWEEP: tuple[tuple[int, int, int], ...] = (
    (32000, 4096, 1),
    (32000, 4096, 2048),
    (8192, 1024, 512),
)


def embedding_cases() -> Iterable[tuple[str, str, str, Callable[[], None]]]:
    """`F.embedding(input_ids, weight)` over EMBEDDING_SWEEP. Indices are
    int32 in `[0, vocab)`; weight is float."""
    device = torch.device("cuda")
    for (vocab, hidden, num) in EMBEDDING_SWEEP:
        shape = f"V{vocab}_D{hidden}_N{num}"
        # Pre-build indices once per shape; reuse across dtypes.
        idx = torch.randint(0, vocab, (num,), dtype=torch.int32, device=device).long()
        for dtype_name, dtype in NORM_DTYPES:  # f32, f16
            weight = torch.randn((vocab, hidden), dtype=dtype, device=device)
            launch = lambda w=weight, i=idx: torch.nn.functional.embedding(i, w)
            yield ("embedding", shape, dtype_name, launch)


def concat_cases() -> Iterable[tuple[str, str, str, Callable[[], None]]]:
    """`torch.cat([a, b], dim=1)` over CONCAT_SWEEP — KV-cache concat
    pattern. Shape label matches `BH{bh}_Ka{ka}_Kb{kb}_D{d}` from the
    bench file."""
    device = torch.device("cuda")
    for (bh, ka, kb, d) in CONCAT_SWEEP:
        shape = f"BH{bh}_Ka{ka}_Kb{kb}_D{d}"
        for dtype_name, dtype in CONCAT_DTYPES:
            a = torch.randn((bh, ka, d), dtype=dtype, device=device)
            b = torch.randn((bh, kb, d), dtype=dtype, device=device)
            launch = lambda a=a, b=b: torch.cat([a, b], dim=1)
            yield ("concat", shape, dtype_name, launch)


def sdpa_cases() -> Iterable[tuple[str, str, str, Callable[[], None]]]:
    """`F.scaled_dot_product_attention(q, k, v, is_causal=True)` with
    GQA broadcasting. PyTorch supports GQA natively by accepting K/V
    with fewer head groups than Q (broadcast happens inside SDPA).

    Shape format matches `sdpa_gqa.rs`:
    `Hq{NUM_Q_HEADS}_Hkv{num_kv}_Q{SEQ_LEN}_D{HEAD_DIM}`.
    """
    device = torch.device("cuda")
    nq = SDPA_NUM_Q_HEADS
    seq = SDPA_SEQ_LEN
    d = SDPA_HEAD_DIM
    batch = SDPA_BATCH
    for num_kv in SDPA_KV_HEAD_SWEEP:
        if nq % num_kv != 0:
            continue
        shape = f"Hq{nq}_Hkv{num_kv}_Q{seq}_D{d}"
        for dtype_name, dtype in SDPA_DTYPES:
            q = torch.randn((batch, nq, seq, d), dtype=dtype, device=device)
            # K/V physical = (batch, num_kv, seq, d); SDPA expects them in
            # the same num-heads dim as Q. For GQA we expand via repeat_interleave
            # so the underlying allocation matches what PyTorch's reference
            # implementation would do (no zero-stride broadcast).
            k = torch.randn((batch, num_kv, seq, d), dtype=dtype, device=device)
            v = torch.randn((batch, num_kv, seq, d), dtype=dtype, device=device)
            if num_kv != nq:
                k_exp = k.repeat_interleave(nq // num_kv, dim=1)
                v_exp = v.repeat_interleave(nq // num_kv, dim=1)
            else:
                k_exp, v_exp = k, v
            launch = lambda q=q, k=k_exp, v=v_exp: (
                torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                )
            )
            yield ("flash_sdpa_gqa", shape, dtype_name, launch)


# Op registry — name → cases generator. Extend per-op for fanout.
OP_REGISTRY: dict[str, Callable[[], Iterable[tuple[str, str, str, Callable[[], None]]]]] = {
    "gemm": gemm_cases,
    "softmax": softmax_cases,
    "log_softmax": log_softmax_cases,
    "layernorm": layernorm_cases,
    "rmsnorm": rmsnorm_cases,
    "reduce": reduce_cases,
    "elementwise": elementwise_cases,
    "conv2d": conv2d_cases,
    "maxpool2d": maxpool2d_cases,
    "avgpool2d": avgpool2d_cases,
    "concat": concat_cases,
    "embedding": embedding_cases,
    "sdpa": sdpa_cases,
}


# ---------------------------------------------------------------------
# Device + version metadata.
# ---------------------------------------------------------------------


def _device_slug() -> str:
    """Short identifier for the active CUDA device. Used in default output filename."""
    name = torch.cuda.get_device_name(0)
    # "NVIDIA GeForce RTX 4070 ..." -> "rtx4070"
    m = re.search(r"RTX\s*(\d{4})", name)
    if m:
        return f"rtx{m.group(1)}"
    # Fallback: lowercase, strip non-alnum
    return re.sub(r"[^a-z0-9]+", "", name.lower())[:20]


def _torch_version_short() -> str:
    """`'2.11.0+cu130'` → `'2.11.0'`."""
    return torch.__version__.split("+", 1)[0]


def _cuda_version_short() -> str:
    """`'13.0'` → `'130'` for filename use."""
    v = torch.version.cuda or "unknown"
    return v.replace(".", "")


def _default_output_path() -> Path:
    """Default output filename derived from device + torch + CUDA version."""
    base = Path(__file__).resolve().parent.parent / "crates" / "baracuda-kernels-bench" / "bench-baselines"
    return base / (
        f"pytorch_{_device_slug()}_{_torch_version_short()}_cu{_cuda_version_short()}.json"
    )


def _metadata(samples: int, inner: int) -> dict[str, object]:
    """Self-describing block written into the JSON header."""
    return {
        "schema_version": 1,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "sample_count": samples,
        "inner_iters": inner,
        "warmup_launches": WARMUP_LAUNCHES,
        "methodology": (
            "wall-clock around torch.cuda.synchronize() per batch; "
            "median of per-batch averages. Matches baracuda's "
            "measure_median_ns in baracuda-kernels-bench/src/lib.rs."
        ),
    }


# ---------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--ops",
        default=",".join(OP_REGISTRY.keys()),
        help=f"Comma-separated ops to refresh. Default: all ({','.join(OP_REGISTRY)}).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Default: bench-baselines/pytorch_<device>_<torch>_cu<cuda>.json",
    )
    p.add_argument("--samples", type=int, default=11, help="Outer batch count (default 11).")
    p.add_argument("--inner", type=int, default=50, help="Inner launches per batch (default 50).")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() == False — install a CUDA-enabled PyTorch wheel.", file=sys.stderr)
        return 2

    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    for op in ops:
        if op not in OP_REGISTRY:
            print(f"ERROR: unknown op '{op}'. Known: {','.join(OP_REGISTRY)}", file=sys.stderr)
            return 2

    output_path = args.output or _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print(f"Ops: {','.join(ops)}")
    print(f"Output: {output_path}")
    print(f"Samples: {args.samples}  inner: {args.inner}")
    print()

    results: list[dict[str, object]] = []
    total_cells = 0
    for op in ops:
        cases = list(OP_REGISTRY[op]())
        total_cells += len(cases)
    print(f"Total cells: {total_cells}")
    print()

    cell_index = 0
    for op in ops:
        for (op_label, shape, dtype_name, launch) in OP_REGISTRY[op]():
            cell_index += 1
            print(f"[{cell_index}/{total_cells}] {op_label}/{dtype_name}/{shape} ...", end=" ", flush=True)
            try:
                median_ns = time_median_ns(launch, args.samples, args.inner)
            except RuntimeError as e:
                print(f"FAILED: {e}", flush=True)
                continue
            print(f"{median_ns:.1f} ns", flush=True)
            results.append(
                {
                    "op": op_label,
                    "shape": shape,
                    "dtype": dtype_name,
                    "median_ns": round(median_ns, 3),
                }
            )

    # Merge with the existing baseline, replacing only the (op, shape, dtype)
    # keys that this run measured. This lets `--ops <subset>` refresh just
    # those ops without wiping coverage of the ones we didn't ask for.
    merged: dict[tuple[str, str, str], dict[str, object]] = {}
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            for entry in existing.get("results", []):
                key = (entry["op"], entry["shape"], entry["dtype"])
                merged[key] = entry
        except (json.JSONDecodeError, KeyError) as e:
            print(f"WARN: ignoring existing {output_path} ({e}); writing fresh", file=sys.stderr)
    for entry in results:
        key = (entry["op"], entry["shape"], entry["dtype"])
        merged[key] = entry
    payload = {
        "metadata": _metadata(args.samples, args.inner),
        "results": list(merged.values()),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"Wrote {len(merged)} cells to {output_path} ({len(results)} refreshed this run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
