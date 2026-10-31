"""Attention scheduler. Every tick it values a look through each camera, then picks the set of detector calls whose
value most exceeds the price times their cost. The price follows the spending: it rises when spending is above the
budget and falls when it is below."""
import math

import numpy as np
from numba import njit

from .constants import (DT, H_EDGES, HALF_VIEW, L_EDGES, MASK, N_HORIZON, N_VIEWS, OPTION_SETS, PATH_LOOKAHEAD,
                        PRICE_STEP, RANGE_M, RATE_TAU)
from .tracker import observe, predict
from .value import bin_of, pos_cov, recall_at, tracked_value


@njit(cache=True)
def check_deadlines(k, X, Pm, tl, init, tick_off, ent_pl, ent_x, ent_y, kd_of, conf_idx_of, err, q):
    """Tracking error of every conflict pedestrian whose deadline is tick k (inf if never detected)."""
    for e in range(tick_off[k], tick_off[k + 1]):
        pl = ent_pl[e]
        if kd_of[pl] == k:
            if init[pl]:
                predict(X, Pm, tl, pl, k, q)
                err[conf_idx_of[pl]] = math.hypot(X[pl, 0] - ent_x[e], X[pl, 1] - ent_y[e])
            else:
                err[conf_idx_of[pl]] = np.inf


@njit(cache=True)
def view_of(th, heading):
    a = th - heading
    a = (a + math.pi) % (2 * math.pi) - math.pi
    return 0 if abs(a) <= HALF_VIEW else (1 if a > 0 else 2)


@njit(cache=True)
def simulate(B, cost, q, P, heading, tick_off, ent_view, ent_det, ent_pl, ent_x, ent_y, ent_zx, ent_zy, ent_d, n_local,
             kd_of, n_conf, conf_idx_of, recall, h_tab, lam_a, lam_b, h_u, rbar, l_new, mu0):
    """Drive one route with budget B (cost units per s) and start price mu0.
    Returns the tracking error of each conflict pedestrian at its deadline, the calls made of each kind, the mean and
    final log price, and the cost units spent."""
    return _run(B, cost, q, P, heading, tick_off, ent_view, ent_det, ent_pl, ent_x, ent_y, ent_zx, ent_zy, ent_d,
                n_local, kd_of, n_conf, conf_idx_of, recall, h_tab, lam_a, lam_b, h_u, rbar, l_new, mu0,
                np.zeros(0, np.int64), np.zeros((0, 6)), np.zeros(0), np.zeros((0, 0, 2)), np.zeros((0, 0, 3)))


@njit(cache=True)
def simulate_recorded(B, cost, q, P, heading, tick_off, ent_view, ent_det, ent_pl, ent_x, ent_y, ent_zx, ent_zy, ent_d,
                      n_local, kd_of, n_conf, conf_idx_of, recall, h_tab, lam_a, lam_b, h_u, rbar, l_new, mu0):
    """simulate, also returning per tick: the chosen option set, the camera values (tracked part, then total), the log
    price, and each track's predicted position and position covariance (s_xx, s_xy, s_yy; NaN if not tracked)."""
    K = tick_off.size - 1
    rec_set = np.zeros(K, np.int64)
    rec_val = np.zeros((K, 6))
    rec_price = np.zeros(K)
    rec_pos = np.full((K, n_local, 2), np.nan)
    rec_cov = np.full((K, n_local, 3), np.nan)
    err, used, lp_mean, lp_end, spent = _run(B, cost, q, P, heading, tick_off, ent_view, ent_det, ent_pl, ent_x, ent_y,
                                             ent_zx, ent_zy, ent_d, n_local, kd_of, n_conf, conf_idx_of, recall, h_tab,
                                             lam_a, lam_b, h_u, rbar, l_new, mu0, rec_set, rec_val, rec_price,
                                             rec_pos, rec_cov)
    return err, used, lp_mean, lp_end, spent, rec_set, rec_val, rec_price, rec_pos, rec_cov


@njit(cache=True)
def _run(B, cost, q, P, heading, tick_off, ent_view, ent_det, ent_pl, ent_x, ent_y, ent_zx, ent_zy, ent_d, n_local,
         kd_of, n_conf, conf_idx_of, recall, h_tab, lam_a, lam_b, h_u, rbar, l_new, mu0, rec_set, rec_val, rec_price,
         rec_pos, rec_cov):
    record = rec_set.size > 0
    K = tick_off.size - 1
    X = np.zeros((n_local, 4))
    Pm = np.zeros((n_local, 4, 4))
    tl = np.zeros(n_local, np.int64)
    init = np.zeros(n_local, np.bool_)
    err = np.full(n_conf, np.nan)
    used = np.zeros(MASK.size, np.int64)
    logmu = math.log(mu0)
    rate = np.full(N_VIEWS, B / 3.0)                # processing rate of each camera (calls per s), smoothed
    m = np.zeros(N_VIEWS)                           # expected undetected conflict pedestrians in each camera
    V = np.zeros(N_VIEWS)
    looks = np.zeros(N_VIEWS, np.int64)
    net = np.zeros(MASK.size)
    on_time = np.zeros(N_HORIZON)                   # share of pedestrians found s before their deadline
    for j in range(N_HORIZON):                      # that are then tracked in time
        on_time[j] = 1.0 - l_new[bin_of(0.5 + j, L_EDGES)]
    logmu_sum = 0.0
    spent_sum = 0.0
    for v in range(N_VIEWS):
        m[v] = math.exp(lam_a[v]) / h_u
    for k in range(K):
        x0, y0 = P[k, 0], P[k, 1]
        n_trk = 0
        V[:] = 0.0
        # value of looking at the tracked pedestrians in range
        for i in range(n_local):
            if not init[i]:
                continue
            dt = (k - tl[i]) * DT
            cx = X[i, 0] + X[i, 2] * dt
            cy = X[i, 1] + X[i, 3] * dt
            d = math.hypot(cx - x0, cy - y0)
            if d > RANGE_M:
                continue
            n_trk += 1
            dpath = 1e9
            for j in range(PATH_LOOKAHEAD + 1):
                kk = min(k + 5 * j, K - 1)
                dpath = min(dpath, math.hypot(cx - P[kk, 0], cy - P[kk, 1]))
            h = h_tab[bin_of(dpath, H_EDGES)]
            if h <= 0.0:
                continue
            th = math.atan2(cy - y0, cx - x0)
            v = view_of(th, heading[k])
            V[v] += h * tracked_value(Pm[i], dt, d, th, q, recall_at(d, recall), rate[v])
        if record:
            rec_val[k, :3] = V
        # value of looking for pedestrians not yet tracked
        xn = math.log(1.0 + n_trk)
        for v in range(N_VIEWS):
            lam = math.exp(lam_a[v] + lam_b[v] * xn)
            if k > 0:
                m[v] = m[v] * math.exp(-h_u * DT) + lam * DT
            acc = 0.0
            for j in range(N_HORIZON):
                s = 0.5 + j
                acc += h_u * math.exp(-(h_u + rate[v] * rbar) * s) * on_time[j]
            V[v] += m[v] * rbar * acc
        if record:
            rec_val[k, 3:] = V
            rec_price[k] = logmu
        # best set of calls: value of the cameras covered minus price times cost
        mu = math.exp(logmu)
        for a in range(MASK.size):
            val = 0.0
            for v in range(N_VIEWS):
                if (MASK[a] >> v) & 1:
                    val += V[v]
            net[a] = val - mu * cost[a]
        best, bi = 0.0, 0
        for ci in range(OPTION_SETS.shape[0]):
            tot = 0.0
            for j in range(3):
                a = OPTION_SETS[ci, j]
                if a >= 0:
                    tot += net[a]
            if tot > best:
                best, bi = tot, ci
        if record:
            rec_set[k] = bi
        spent = 0.0
        looks[:] = 0
        for j in range(3):
            a = OPTION_SETS[bi, j]
            if a < 0:
                continue
            observe(MASK[a], k, x0, y0, X, Pm, tl, init, tick_off, ent_view, ent_det, ent_pl, ent_zx, ent_zy, ent_d, q)
            for v in range(N_VIEWS):
                if (MASK[a] >> v) & 1:
                    m[v] *= 1.0 - rbar
                    looks[v] += 1
            used[a] += 1
            spent += cost[a]
        check_deadlines(k, X, Pm, tl, init, tick_off, ent_pl, ent_x, ent_y, kd_of, conf_idx_of, err, q)
        if record:
            for i in range(n_local):
                if init[i]:
                    dt = (k - tl[i]) * DT
                    rec_pos[k, i, 0] = X[i, 0] + X[i, 2] * dt
                    rec_pos[k, i, 1] = X[i, 1] + X[i, 3] * dt
                    rec_cov[k, i, 0], rec_cov[k, i, 1], rec_cov[k, i, 2] = pos_cov(Pm[i], dt, q)
        for v in range(N_VIEWS):
            rate[v] += (looks[v] / DT - rate[v]) * (DT / RATE_TAU)
        logmu += PRICE_STEP * (spent - B * DT)
        logmu_sum += logmu
        spent_sum += spent
    return err, used, logmu_sum / K, logmu, spent_sum
