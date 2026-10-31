"""Value of looking: how much a detection now lowers the probability that a pedestrian's predicted position is off by
more than ERR_TOL at deadlines 0.5 ... 5.5 s ahead."""
import math

import numpy as np
from numba import njit

from .constants import ERR_TOL, FOCAL_PX, HEIGHT_EDGES_PX, N_HORIZON, PERSON_M, SIG_LAT
from .tracker import range_sigma

N_QUAD = 101


@njit(cache=True)
def height_bin(d):
    """Bin of the apparent person height (px) at distance d."""
    h = FOCAL_PX * PERSON_M / max(d, 0.3)
    if h < HEIGHT_EDGES_PX[0]:
        return 0
    if h < HEIGHT_EDGES_PX[1]:
        return 1
    if h < HEIGHT_EDGES_PX[2]:
        return 2
    return 3


@njit(cache=True)
def recall_at(d, recall):
    return recall[height_bin(d)]


@njit(cache=True)
def bin_of(x, edges):
    for b in range(edges.size - 1):
        if x < edges[b + 1]:
            return b
    return edges.size - 2


@njit(cache=True)
def p_exceed(s00, s01, s11, r):
    """P(|e| > r) for e ~ N(0, [[s00, s01], [s01, s11]]), Simpson rule over the minor axis."""
    tr = s00 + s11
    det = s00 * s11 - s01 * s01
    disc = math.sqrt(max(tr * tr / 4.0 - det, 0.0))
    l1 = tr / 2.0 + disc
    l2 = max(tr / 2.0 - disc, 1e-12)
    b = min(r / math.sqrt(l2), 8.0)
    h = math.pi / (N_QUAD - 1)
    acc = 0.0
    for j in range(N_QUAD):
        ph = -0.5 * math.pi + j * h
        z = b * math.sin(ph)
        inner = max(r * r - l2 * z * z, 0.0)
        w = 1.0 if (j == 0 or j == N_QUAD - 1) else (4.0 if j % 2 == 1 else 2.0)
        acc += w * math.exp(-0.5 * z * z) * math.erf(math.sqrt(inner / (2.0 * l1))) * b * math.cos(ph)
    inside = acc * h / 3.0 / math.sqrt(2.0 * math.pi)
    return min(max(1.0 - inside, 0.0), 1.0)


@njit(cache=True)
def pos_cov(S, dt, q):
    """Position block of covariance S predicted dt ahead."""
    a = S[0, 0] + dt * (S[0, 2] + S[2, 0]) + dt * dt * S[2, 2] + dt ** 3 / 3.0 * q
    b = S[0, 1] + dt * (S[0, 3] + S[2, 1]) + dt * dt * S[2, 3]
    c = S[1, 1] + dt * (S[1, 3] + S[3, 1]) + dt * dt * S[3, 3] + dt ** 3 / 3.0 * q
    return a, b, c


@njit(cache=True)
def full_cov(S, dt, q):
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    Q = np.zeros((4, 4))
    Q[0, 0] = Q[1, 1] = dt ** 3 / 3 * q
    Q[0, 2] = Q[2, 0] = Q[1, 3] = Q[3, 1] = dt ** 2 / 2 * q
    Q[2, 2] = Q[3, 3] = dt * q
    return F @ S @ F.T.copy() + Q


@njit(cache=True)
def tracked_value(Pi, age, d, th, q, r, rate):
    """Value of a look at a tracked pedestrian, before the weight h of its distance to the route.
    Pi: covariance after the last detection, age s ago; d, th: predicted distance and bearing; r: detection
    probability; rate: current processing rate of the camera, which discounts later deadlines."""
    S_now = full_cov(Pi, age, q)
    sr = range_sigma(d)
    c_, s_ = math.cos(th), math.sin(th)
    Sz = np.empty((2, 2))
    Sz[0, 0] = S_now[0, 0] + c_ * c_ * sr * sr + s_ * s_ * SIG_LAT * SIG_LAT
    Sz[1, 1] = S_now[1, 1] + s_ * s_ * sr * sr + c_ * c_ * SIG_LAT * SIG_LAT
    Sz[0, 1] = S_now[0, 1] + c_ * s_ * (sr * sr - SIG_LAT * SIG_LAT)
    Sz[1, 0] = S_now[1, 0] + c_ * s_ * (sr * sr - SIG_LAT * SIG_LAT)
    Kg = S_now[:, :2].copy() @ np.linalg.inv(Sz)
    S_post = S_now - Kg @ S_now[:2, :].copy()
    v = 0.0
    for j in range(N_HORIZON):
        sj = 0.5 + j
        a0, b0, c0 = pos_cov(S_now, sj, q)
        a1, b1, c1 = pos_cov(S_post, sj, q)
        v += math.exp(-rate * r * sj) * (p_exceed(a0, b0, c0, ERR_TOL) - p_exceed(a1, b1, c1, ERR_TOL))
    return r * v
