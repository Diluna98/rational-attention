"""Robot routes replayed through a recorded day. Each route is a shortest path between two random walkable cells,
driven at 1 m/s from a random start time. For every tick and every pedestrian in the scene it stores the camera that
sees them (or none), whether the detector would find them, a noisy detected position, and which pedestrians come
within CONFLICT_M of the robot (conflicts) with the tick by which they must be tracked (deadline)."""
import math
from pathlib import Path

import numpy as np
from numba import njit
from scipy.sparse.csgraph import connected_components, dijkstra

from .atc import load_tracks
from .constants import (ALIGN_MAX, CELL, CONFLICT_M, DT, HALF_VIEW, MIN_AGE_S, MIN_K, MIN_ROUTE_M, OX, OY, RANGE_M,
                        ROUTES_PER_DAY, SEED, SIG_LAT, STEP_M, WARN_TICKS)
from .detector import load_detector
from .mapping import blocked, build_graph, on_obstacles, resample, string_pull
from .value import height_bin

FIELDS = ("P", "heading", "tick_off", "ent_view", "ent_det", "ent_pl", "ent_x", "ent_y", "ent_zx", "ent_zy", "ent_d",
          "kd_of", "conf_idx_of", "elig")


@njit(cache=True)
def sightings(P, f0, offsets, fr_p, fr_x, fr_y, birth_t, T0, occ, n_persons, thr, rho_frame, seed):
    """Per pedestrian sample along the route: tick, person, camera (-1 if out of range or behind an obstacle),
    detected or not, distance; then the conflict pedestrians, their first conflict tick, and whether they count."""
    np.random.seed(seed)
    K = P.shape[0]
    e0, e1 = offsets[f0], offsets[f0 + K]
    n = e1 - e0
    ent_k = np.empty(n, np.int32)
    ent_p = np.empty(n, np.int32)
    ent_view = np.empty(n, np.int8)
    ent_det = np.zeros(n, np.bool_)
    ent_d = np.empty(n, np.float32)
    tick_off = np.empty(K + 1, np.int64)
    conf_k = np.full(n_persons, -1, np.int32)
    z = np.zeros(n_persons)
    z_f = np.full(n_persons, -1, np.int64)
    j = 0
    for k in range(K):
        tick_off[k] = j
        f = f0 + k
        x0, y0 = P[k, 0], P[k, 1]
        if k < K - 1:
            hd = math.atan2(P[k + 1, 1] - y0, P[k + 1, 0] - x0)
        else:
            hd = math.atan2(y0 - P[k - 1, 1], x0 - P[k - 1, 0])
        for e in range(offsets[f], offsets[f + 1]):
            p = fr_p[e]
            dx, dy = fr_x[e] - x0, fr_y[e] - y0
            d = math.hypot(dx, dy)
            a = math.atan2(dy, dx) - hd
            a = (a + math.pi) % (2 * math.pi) - math.pi
            if abs(a) <= HALF_VIEW:
                cam = 0
            elif a > 0:
                cam = 1
            else:
                cam = 2
            vis = d <= RANGE_M and not blocked(occ, x0, y0, fr_x[e], fr_y[e])
            # detection score: AR(1) per person over the 10 Hz frames
            if z_f[p] < 0:
                z[p] = np.random.standard_normal()
            else:
                r = rho_frame ** (f - z_f[p])
                z[p] = r * z[p] + math.sqrt(1.0 - r * r) * np.random.standard_normal()
            z_f[p] = f
            ent_k[j] = k
            ent_p[j] = p
            ent_view[j] = cam if vis else -1
            ent_det[j] = vis and not (z[p] < thr[height_bin(d)])
            ent_d[j] = d
            if d < CONFLICT_M and conf_k[p] < 0:
                conf_k[p] = k
            j += 1
    tick_off[K] = j
    m = 0
    for e in range(n):
        p = ent_p[e]
        if conf_k[p] >= 0:
            m += 1
            conf_k[p] = -conf_k[p] - 2
    cp = np.empty(m, np.int32)
    ck = np.empty(m, np.int32)
    i = 0
    for e in range(n):
        p = ent_p[e]
        if conf_k[p] <= -2:
            cp[i] = p
            ck[i] = -conf_k[p] - 2
            conf_k[p] = -1
            i += 1
    elig = np.zeros(m, np.bool_)
    for i in range(m):
        age = T0 + (f0 + ck[i]) * DT - birth_t[cp[i]]
        elig[i] = ck[i] >= MIN_K and age >= MIN_AGE_S
    return ent_k, ent_p, ent_view, ent_det, ent_d, tick_off, cp, ck, elig


def sample_routes(rng, G, cells, big, route_free, T0, T1, n_frames):
    routes = []
    while len(routes) < ROUTES_PER_DAY:
        s, g = rng.choice(big, 2, replace=False)
        _, pred = dijkstra(G, indices=s, return_predecessors=True)
        node, path = g, [g]
        while node != s:
            node = pred[node]
            path.append(node)
        pts = (cells[path[::-1]][:, ::-1] + 0.5) * CELL + np.array([OX, OY])
        P, L = resample(string_pull(pts, route_free), STEP_M)
        if L < MIN_ROUTE_M:
            continue
        dur = (P.shape[0] - 1) * DT
        if T1 - T0 - dur - 1.0 <= 0:
            continue
        ts = T0 + rng.uniform(0, T1 - T0 - dur - 1.0)
        f0 = int(math.floor((ts - T0) / DT))
        if f0 + P.shape[0] + 1 > n_frames:
            continue
        routes.append((P, ts, f0, L))
    return routes


def prepare_day(tracks_npz, seed_index, maps, det, out):
    """Sample ROUTES_PER_DAY routes on one day and store everything the replay needs in out (npz).
    Returns a short record, or None if the day's tracks do not line up with the map."""
    t, pidx, x, y, n_persons = load_tracks(tracks_npz)
    share = on_obstacles(maps["occ"], x, y)
    if share > ALIGN_MAX:
        return None
    G, cells = build_graph(maps["route_free"])
    _, cc = connected_components(G, directed=False)
    big = np.flatnonzero(cc == np.bincount(cc).argmax())
    los = maps["los"]
    T0, T1 = float(t[0]), float(t[-1])
    # one sample per person per 10 Hz frame
    f_all = np.floor((t - T0) / DT).astype(np.int64)
    u, fi = np.unique(f_all * n_persons + pidx, return_index=True)
    fr_f = u // n_persons
    fr_p = (u % n_persons).astype(np.int64)
    fr_x, fr_y = x[fi].copy(), y[fi].copy()
    offsets = np.searchsorted(fr_f, np.arange(fr_f[-1] + 2)).astype(np.int64)
    _, first = np.unique(pidx, return_index=True)
    birth_t = t[first]
    routes = sample_routes(np.random.default_rng(SEED + seed_index), G, cells, big, maps["route_free"], T0, T1,
                           offsets.size - 2)
    store = {k: [] for k in FIELDS}
    meta = dict(n_local=[], n_conf=[], length_m=[], t_start=[])
    for ri, (P, ts, f0, L) in enumerate(routes):
        seed = SEED + 100000 * (seed_index + 1) + ri
        ent_k, ent_p, ent_view, ent_det, ent_d, tick_off, cp, ck, elig = sightings(
            P, f0, offsets, fr_p, fr_x, fr_y, birth_t, T0, los, n_persons, det["thr"], det["rho_frame"], seed)
        e0 = offsets[f0]
        ent_x = fr_x[e0:e0 + ent_k.size].copy()
        ent_y = fr_y[e0:e0 + ent_k.size].copy()
        K = P.shape[0]
        heading = np.empty(K)
        heading[:-1] = np.arctan2(np.diff(P[:, 1]), np.diff(P[:, 0]))
        heading[-1] = heading[-2]
        # detected positions: noise along and across the line of sight
        nrng = np.random.default_rng(SEED + 7919 * (100000 * (seed_index + 1) + ri + 1))
        nr, nt = nrng.standard_normal(ent_k.size), nrng.standard_normal(ent_k.size)
        th = np.arctan2(ent_y - P[ent_k, 1], ent_x - P[ent_k, 0])
        sr = np.where(ent_d <= 3.0, 0.01 * ent_d, 0.05 * ent_d)
        ent_zx = ent_x + nr * sr * np.cos(th) - nt * SIG_LAT * np.sin(th)
        ent_zy = ent_y + nr * sr * np.sin(th) + nt * SIG_LAT * np.cos(th)
        # local person ids; deadline = WARN_TICKS before the conflict, if the person is in the scene then
        uniq, ent_pl = np.unique(ent_p, return_inverse=True)
        local_of = {int(p): j for j, p in enumerate(uniq)}
        kd_of = np.full(uniq.size, -1, np.int32)
        conf_idx_of = np.full(uniq.size, -1, np.int32)
        present = np.zeros(cp.size, bool)
        for ci, (p, kc) in enumerate(zip(cp, ck)):
            pl = local_of[int(p)]
            conf_idx_of[pl] = ci
            kd = int(kc) - WARN_TICKS
            if kd >= 0:
                present[ci] = bool(np.any(ent_p[tick_off[kd]:tick_off[kd + 1]] == p))
            if elig[ci] and present[ci]:
                kd_of[pl] = kd
        for k_, v_ in (("P", P), ("heading", heading), ("tick_off", tick_off), ("ent_view", ent_view),
                       ("ent_det", ent_det), ("ent_pl", ent_pl), ("ent_x", ent_x), ("ent_y", ent_y),
                       ("ent_zx", ent_zx), ("ent_zy", ent_zy), ("ent_d", ent_d), ("kd_of", kd_of),
                       ("conf_idx_of", conf_idx_of), ("elig", elig & present)):
            store[k_].append(np.asarray(v_))
        meta["n_local"].append(uniq.size)
        meta["n_conf"].append(cp.size)
        meta["length_m"].append(L)
        meta["t_start"].append(ts)
    arrays = {}
    for k_, lst in store.items():
        arrays[k_] = np.concatenate(lst)
        arrays[k_ + "_off"] = np.r_[0, np.cumsum([a.shape[0] for a in lst])].astype(np.int64)
    for k_, v_ in meta.items():
        arrays[k_] = np.array(v_)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    return dict(routes=len(routes), km=float(np.sum(meta["length_m"]) / 1000), on_obstacles=share,
                conflicts=int(np.concatenate(store["elig"]).sum()))



def prepare_job(job):
    """(day, seed index, tracks file, maps, output file) -> (day, record of prepare_day)."""
    day, seed, tracks, maps, out = job
    return day, prepare_day(tracks, seed, maps, load_detector(), out)


class Day:
    """Prepared routes of one day."""
    def __init__(self, path):
        self.z = dict(np.load(path))
        self.n = int(self.z["n_local"].size)

    def route(self, r):
        z = self.z
        out = {k: z[k][z[k + "_off"][r]:z[k + "_off"][r + 1]] for k in FIELDS}
        out["P"] = np.ascontiguousarray(out["P"], dtype=np.float64)
        for k in ("ent_x", "ent_y", "ent_zx", "ent_zy", "ent_d", "heading"):
            out[k] = np.ascontiguousarray(out[k], dtype=np.float64)
        out["tick_off"] = out["tick_off"].astype(np.int64)
        out["ent_pl"] = out["ent_pl"].astype(np.int32)
        out["n_local"] = int(z["n_local"][r])
        out["n_conf"] = int(z["n_conf"][r])
        out["length_m"] = float(z["length_m"][r])
        return out

    def order(self):
        """Routes in order of start time (one robot through the day)."""
        return np.argsort(self.z["t_start"], kind="stable")
