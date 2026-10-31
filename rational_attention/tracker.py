"""Constant-velocity Kalman filter per pedestrian. State [x, y, vx, vy]; tracks are indexed by local person id."""
import math

import numpy as np
from numba import njit

from .constants import DT, SIG_LAT, SIGMA_V0


@njit(cache=True)
def range_sigma(d):
    """Detection noise along the line of sight (m) at distance d."""
    return 0.01 * d if d <= 3.0 else 0.05 * d


@njit(cache=True)
def meas_cov(th, d):
    """Detection noise covariance for a pedestrian at bearing th and distance d."""
    sr = range_sigma(d)
    c_, s_ = math.cos(th), math.sin(th)
    R = np.empty((2, 2))
    R[0, 0] = c_ * c_ * sr * sr + s_ * s_ * SIG_LAT * SIG_LAT
    R[1, 1] = s_ * s_ * sr * sr + c_ * c_ * SIG_LAT * SIG_LAT
    R[0, 1] = c_ * s_ * (sr * sr - SIG_LAT * SIG_LAT)
    R[1, 0] = R[0, 1]
    return R


@njit(cache=True)
def predict(X, Pm, tl, i, k, q):
    """Advance track i from its last update (tick tl[i]) to tick k."""
    if k <= tl[i]:
        return
    dt = (k - tl[i]) * DT
    X[i, 0] += X[i, 2] * dt
    X[i, 1] += X[i, 3] * dt
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    Q = np.zeros((4, 4))
    Q[0, 0] = Q[1, 1] = dt ** 3 / 3 * q
    Q[0, 2] = Q[2, 0] = Q[1, 3] = Q[3, 1] = dt ** 2 / 2 * q
    Q[2, 2] = Q[3, 3] = dt * q
    Ft = F.T.copy()
    Pm[i] = F @ Pm[i].copy() @ Ft + Q
    tl[i] = k


@njit(cache=True)
def update(X, Pm, i, zx, zy, R):
    P = Pm[i].copy()
    S = P[:2, :2].copy() + R
    Si = np.linalg.inv(S)
    K = P[:, :2].copy() @ Si
    y0, y1 = zx - X[i, 0], zy - X[i, 1]
    for r in range(4):
        X[i, r] += K[r, 0] * y0 + K[r, 1] * y1
    H = np.zeros((2, 4))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    Pm[i] = (np.eye(4) - K @ H) @ P


@njit(cache=True)
def observe(mask, k, x0, y0, X, Pm, tl, init, tick_off, ent_view, ent_det, ent_pl, ent_zx, ent_zy, ent_d, q):
    """Run the detector on the cameras in mask at tick k and update or start the tracks of the people it finds."""
    for e in range(tick_off[k], tick_off[k + 1]):
        v = ent_view[e]
        if v < 0 or ((mask >> v) & 1) == 0 or not ent_det[e]:
            continue
        pl = ent_pl[e]
        zx, zy = ent_zx[e], ent_zy[e]
        R = meas_cov(math.atan2(zy - y0, zx - x0), ent_d[e])
        if not init[pl]:
            init[pl] = True
            tl[pl] = k
            X[pl, 0], X[pl, 1], X[pl, 2], X[pl, 3] = zx, zy, 0.0, 0.0
            for a in range(4):
                for b in range(4):
                    Pm[pl, a, b] = 0.0
            Pm[pl, 0, 0], Pm[pl, 0, 1], Pm[pl, 1, 0], Pm[pl, 1, 1] = R[0, 0], R[0, 1], R[1, 0], R[1, 1]
            Pm[pl, 2, 2] = SIGMA_V0 * SIGMA_V0
            Pm[pl, 3, 3] = SIGMA_V0 * SIGMA_V0
        else:
            predict(X, Pm, tl, pl, k, q)
            update(X, Pm, pl, zx, zy, R)
