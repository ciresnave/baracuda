#!/usr/bin/env python3
"""Pure-Python (no GPU, no dependencies) check of the Sinkhorn-Knopp adjoint.

Forward, as in mHC.cu: K = exp(H); repeat `iters` times: divide each row by its sum, then each column by its sum.
For y = x / sum(x) the exact adjoint is  dx = (dy - sum(dy * y)) / sum(x)  (a constant is subtracted).

mHC.cu `sinkhorn_knopp_backward` instead does  d = d - y * sum(d * y)  (the subtracted term is multiplied by y)
and has no 1/sum. This script differentiates the same forward numerically and prints the error of both.
"""
import math

N = 4
_state = 12345


def uniform(lo, hi):
    """Deterministic LCG in [lo, hi): reproducible test data, not for security."""
    global _state
    _state = (_state * 1103515245 + 12345) % 2**31
    return lo + (hi - lo) * _state / 2**31


def forward(K, iters):
    M = [r[:] for r in K]
    for _ in range(iters):
        for i in range(N):
            s = sum(M[i]); M[i] = [v / s for v in M[i]]
        for j in range(N):
            s = sum(M[i][j] for i in range(N))
            for i in range(N): M[i][j] /= s
    return M


def checkpoints(K, iters):
    tile = [r[:] for r in K]; chk = []; rin = []; cin = []
    for _ in range(iters):
        r = [1 / sum(tile[i]) for i in range(N)]
        tile = [[tile[i][j] * r[i] for j in range(N)] for i in range(N)]; chk.append([x[:] for x in tile]); rin.append(r)
        c = [1 / sum(tile[i][j] for i in range(N)) for j in range(N)]
        tile = [[tile[i][j] * c[j] for j in range(N)] for i in range(N)]; cin.append(c)
    return chk, rin, cin


def col_dot(d, M):
    return [sum(d[i][j] * M[i][j] for i in range(N)) for j in range(N)]


def row_dot(d, M):
    return [sum(d[i][j] * M[i][j] for j in range(N)) for i in range(N)]


def step_upstream(d, A):
    # mHC.cu: column step then row step, each  d -= y * sum(d * y), no 1/sum; y = the checkpoint A.
    dot = col_dot(d, A)
    d = [[d[i][j] - A[i][j] * dot[j] for j in range(N)] for i in range(N)]
    dot = row_dot(d, A)
    return [[d[i][j] - A[i][j] * dot[i] for j in range(N)] for i in range(N)]


def step_exact(d, A, c, r):
    B = [[A[i][j] * c[j] for j in range(N)] for i in range(N)]
    dot = col_dot(d, B)
    d = [[c[j] * (d[i][j] - dot[j]) for j in range(N)] for i in range(N)]
    dot = row_dot(d, A)
    return [[r[i] * (d[i][j] - dot[i]) for j in range(N)] for i in range(N)]


def adjoint(K, G, iters, upstream_style):
    chk, rin, cin = checkpoints(K, iters)
    d = [r[:] for r in G]
    for it in range(iters - 1, -1, -1):
        if upstream_style:
            d = step_upstream(d, chk[it])
        else:
            d = step_exact(d, chk[it], cin[it], rin[it])
    return d


for iters in (1, 2, 20):
    K = [[math.exp(uniform(-0.3, 0.3)) for _ in range(N)] for _ in range(N)]
    G = [[uniform(-1, 1) for _ in range(N)] for _ in range(N)]
    loss = lambda K: sum(forward(K, iters)[i][j] * G[i][j] for i in range(N) for j in range(N))
    e = 1e-6; fd = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            Kp = [r[:] for r in K]; Kp[i][j] += e
            Km = [r[:] for r in K]; Km[i][j] -= e
            fd[i][j] = (loss(Kp) - loss(Km)) / (2 * e)
    err = lambda a: max(abs(a[i][j] - fd[i][j]) for i in range(N) for j in range(N))
    print(f"iters={iters:2d}  max|adjoint - finite difference|: exact {err(adjoint(K, G, iters, False)):.2e}"
          f"   mHC.cu style {err(adjoint(K, G, iters, True)):.2e}   (scale {max(abs(v) for r in fd for v in r):.2f})")
