"""Calibration on the training days.
Value model, from a run with a fixed camera schedule (5 calls/s, one left and one right look every 10 calls):
  h       conflicts per s within the next 6 s for a tracked pedestrian, by distance to the route
  lam     arrivals per s of conflict pedestrians in each camera's view, lam = exp(lam_a + lam_b log(1 + n))
          with n the number of tracked pedestrians in range
  h_u     1 / mean time from a conflict pedestrian's first appearance to its deadline
  rbar    mean detection probability of conflict pedestrians while they are not yet tracked
  l_new   share of late conflicts among pedestrians first detected 0-1, 1-2, ... s before the deadline
Start price, per cost table and budget: median over the training routes of the route-mean log price."""
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numba import njit

from .constants import DT, ERR_TOL, H_EDGES, HORIZON_TICKS, L_EDGES, PATH_LOOKAHEAD, Q_ACC, RANGE_M
from .detector import load_detector
from .replay import run_days
from .routes import Day
from .scheduler import check_deadlines
from .tracker import observe
from .value import bin_of, recall_at

NMAX = 300
SCHEDULE = np.array([1, 0, 0, 0, 0, 2, 0, 0, 0, 0], np.int64)      # camera of each call: L, F, F, F, F, R, F, ...
FIXED_RATE = 5.0


@njit(cache=True)
def record_route(q, P, tick_off, ent_view, ent_det, ent_pl, ent_x, ent_y, ent_zx, ent_zy, ent_d, n_local, kd_of,
                 n_conf, conf_idx_of, recall, hcnt, hpos, n_hist, arr, rsum, tau):
    """Drive one route with the fixed schedule and add to the counts behind h, lam, h_u and rbar.
    Returns the tracking errors at the deadlines and the tick each pedestrian was first detected."""
    K = tick_off.size - 1
    X = np.zeros((n_local, 4))
    Pm = np.zeros((n_local, 4, 4))
    tl = np.zeros(n_local, np.int64)
    init = np.zeros(n_local, np.bool_)
    seen = np.zeros(n_local, np.bool_)
    first_det = np.full(n_local, -1, np.int64)
    err = np.full(n_conf, np.nan)
    credit = 1.0
    slot = 0
    for k in range(K):
        x0, y0 = P[k, 0], P[k, 1]
        n_trk = 0
        for i in range(n_local):
            if not init[i]:
                continue
            dt = (k - tl[i]) * DT
            cx = X[i, 0] + X[i, 2] * dt
            cy = X[i, 1] + X[i, 3] * dt
            if math.hypot(cx - x0, cy - y0) > RANGE_M:
                continue
            n_trk += 1
            if k % 10 == 0:
                dpath = 1e9
                for j in range(PATH_LOOKAHEAD + 1):
                    kk = min(k + 5 * j, K - 1)
                    dpath = min(dpath, math.hypot(cx - P[kk, 0], cy - P[kk, 1]))
                b = bin_of(dpath, H_EDGES)
                hcnt[b] += 1
                if kd_of[i] > k and kd_of[i] <= k + HORIZON_TICKS:
                    hpos[b] += 1
        nb = min(n_trk, NMAX - 1)
        n_hist[nb] += 1
        for e in range(tick_off[k], tick_off[k + 1]):
            pl = ent_pl[e]
            if ent_view[e] >= 0 and kd_of[pl] > k:
                if not seen[pl]:
                    arr[ent_view[e], nb] += 1
                    tau[0] += (kd_of[pl] - k) * DT
                    tau[1] += 1
                if not init[pl]:
                    rsum[0] += recall_at(ent_d[e], recall)
                    rsum[1] += 1
            if ent_view[e] >= 0:
                seen[pl] = True
        if k > 0:
            credit += FIXED_RATE * DT
        nproc = 0
        while credit >= 1.0 - 1e-9 and nproc < 3:
            observe(1 << SCHEDULE[slot % SCHEDULE.size], k, x0, y0, X, Pm, tl, init, tick_off, ent_view, ent_det,
                    ent_pl, ent_zx, ent_zy, ent_d, q)
            credit -= 1.0
            nproc += 1
            slot += 1
        for i in range(n_local):
            if init[i] and first_det[i] < 0:
                first_det[i] = k
        check_deadlines(k, X, Pm, tl, init, tick_off, ent_pl, ent_x, ent_y, kd_of, conf_idx_of, err, q)
    return err, first_det


def record_day(path):
    recall = load_detector()["recall"]
    D = Day(path)
    rec = dict(hcnt=np.zeros(H_EDGES.size - 1, np.int64), hpos=np.zeros(H_EDGES.size - 1, np.int64),
               n=np.zeros(NMAX, np.int64), arr=np.zeros((3, NMAX), np.int64), r=np.zeros(2), tau=np.zeros(2))
    lcnt = np.zeros(L_EDGES.size - 1)
    llate = np.zeros(L_EDGES.size - 1)
    for r in range(D.n):
        d = D.route(r)
        err, fd = record_route(Q_ACC, d["P"], d["tick_off"], d["ent_view"], d["ent_det"], d["ent_pl"], d["ent_x"],
                               d["ent_y"], d["ent_zx"], d["ent_zy"], d["ent_d"], d["n_local"], d["kd_of"],
                               d["n_conf"], d["conf_idx_of"], recall, rec["hcnt"], rec["hpos"], rec["n"], rec["arr"],
                               rec["r"], rec["tau"])
        for pl in range(d["n_local"]):
            kd = d["kd_of"][pl]
            if kd >= 0 and 0 <= fd[pl] < kd:
                b = int(np.searchsorted(L_EDGES, (kd - fd[pl]) * DT, side="right") - 1)
                lcnt[b] += 1
                llate[b] += not (err[d["conf_idx_of"][pl]] <= ERR_TOL)
    return rec, lcnt, llate


def poisson_fit(T, A):
    """A_n ~ Poisson(T_n exp(a + b log(1 + n))); Newton's method on (a, b)."""
    x = np.log1p(np.arange(T.size))
    w = T > 0
    x, T, A = x[w], T[w], A[w]
    a, b = math.log(max(A.sum(), 1) / T.sum()), 0.0
    for _ in range(100):
        mu = T * np.exp(a + b * x)
        g = np.array([(A - mu).sum(), ((A - mu) * x).sum()])
        H = np.array([[mu.sum(), (mu * x).sum()], [(mu * x).sum(), (mu * x * x).sum()]])
        step = np.linalg.solve(H, g)
        a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-10:
            break
    return float(a), float(b)


def value_model(paths):
    with ProcessPoolExecutor(max_workers=len(paths)) as ex:
        res = list(ex.map(record_day, paths))
    hcnt = sum(r[0]["hcnt"] for r in res)
    hpos = sum(r[0]["hpos"] for r in res)
    n = sum(r[0]["n"] for r in res)
    arr = sum(r[0]["arr"] for r in res)
    rr = sum(r[0]["r"] for r in res)
    tau = sum(r[0]["tau"] for r in res)
    lcnt = sum(r[1] for r in res)
    llate = sum(r[2] for r in res)
    lam = [poisson_fit(n * DT, arr[v]) for v in range(3)]
    return dict(h_edges_m=H_EDGES[:-1].tolist(),
                h_per_s=np.where(hcnt > 0, hpos / np.maximum(hcnt, 1) / (HORIZON_TICKS * DT), 0.0).tolist(),
                lam_a=[l_[0] for l_ in lam], lam_b=[l_[1] for l_ in lam], h_u=float(tau[1] / tau[0]),
                rbar=float(rr[0] / rr[1]), l_edges_s=L_EDGES[:-1].tolist(),
                l_new=np.where(lcnt > 0, llate / np.maximum(lcnt, 1), 1.0).tolist())


def start_price(paths, cost, B, model, log_guess, passes=1):
    """A pass starts every training day at the guess; the result is the median route-mean log price."""
    lp = log_guess
    for _ in range(passes):
        lp = float(np.median(run_days(paths, cost, B, model, lp)["log_price"]))
    return lp
