#!/usr/bin/env python3
"""Pure-Python (no GPU, no dependencies) check of the Sinkhorn-Knopp adjoint.

Forward, as in mHC.cu: K = exp(H); repeat `iters` times: divide each row by its sum, then each column by its sum.
For y = x / sum(x) the exact adjoint is  dx = (dy - sum(dy * y)) / sum(x)  (a constant is subtracted).

mHC.cu `sinkhorn_knopp_backward` instead does  d = d - y * sum(d * y)  (the subtracted term is multiplied by y)
and has no 1/sum. This script differentiates the same forward numerically and prints the error of both.
"""
import math, random

N = 4
random.seed(1)


def forward(K, iters):
    M = [r[:] for r in K]
    for _ in range(iters):
        for i in range(N):
            s = sum(M[i]); M[i] = [v / s for v in M[i]]
        for j in range(N):
            s = sum(M[i][j] for i in range(N))
            for i in range(N): M[i][j] /= s
    return M


def adjoint(K, G, iters, upstream_style):
    tile = [r[:] for r in K]; chk = []; rin = []; cin = []
    for _ in range(iters):
        r = [1 / sum(tile[i]) for i in range(N)]
        tile = [[tile[i][j] * r[i] for j in range(N)] for i in range(N)]; chk.append([x[:] for x in tile]); rin.append(r)
        c = [1 / sum(tile[i][j] for i in range(N)) for j in range(N)]
        tile = [[tile[i][j] * c[j] for j in range(N)] for i in range(N)]; cin.append(c)
    d = [r[:] for r in G]
    for it in range(iters - 1, -1, -1):
        A, c, r = chk[it], cin[it], rin[it]
        if upstream_style:
            # mHC.cu: column step then row step, each  d -= y * sum(d * y), no 1/sum; y = the checkpoint A.
            dot = [sum(d[i][j] * A[i][j] for i in range(N)) for j in range(N)]
            d = [[d[i][j] - A[i][j] * dot[j] for j in range(N)] for i in range(N)]
            dot = [sum(d[i][j] * A[i][j] for j in range(N)) for i in range(N)]
            d = [[d[i][j] - A[i][j] * dot[i] for j in range(N)] for i in range(N)]
        else:
            B = [[A[i][j] * c[j] for j in range(N)] for i in range(N)]
            dot = [sum(d[i][j] * B[i][j] for i in range(N)) for j in range(N)]
            d = [[c[j] * (d[i][j] - dot[j]) for j in range(N)] for i in range(N)]
            dot = [sum(d[i][j] * A[i][j] for j in range(N)) for i in range(N)]
            d = [[r[i] * (d[i][j] - dot[i]) for j in range(N)] for i in range(N)]
    return d


for iters in (1, 2, 20):
    K = [[math.exp(random.uniform(-0.3, 0.3)) for _ in range(N)] for _ in range(N)]
    G = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(N)]
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
