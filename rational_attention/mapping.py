"""Map of the ATC shopping centre: obstacles for line of sight and the walkable cells used to plan robot routes.
Obstacles are the occupancy grid's dark pixels, minus pixels that many tracked people walked through."""
import math

import numpy as np
from numba import njit
from PIL import Image
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix

from .atc import load_tracks, map_files
from .constants import BLOCK_PX, CELL, CELL_PX, LOS_CLOSE_M, MIN_IDS, OCC_MAX_VALUE, OX, OY, RES, ROUTE_CLEAR_M


def distinct_counts(pidx, ix, iy, nx, ny):
    """Number of different people seen in each grid cell."""
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    key = np.unique(pidx[ok] * (nx * ny) + iy[ok].astype(np.int64) * nx + ix[ok])
    return np.bincount(key % (nx * ny), minlength=nx * ny).reshape(ny, nx)


def disk(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return xx * xx + yy * yy <= r * r


def build_map(atc_dir, tracks_npz):
    """occ: obstacles (pixels); los: obstacles for line of sight (closed by LOS_CLOSE_M); route_free: cells a robot
    route may use (walked by >= MIN_IDS people and >= ROUTE_CLEAR_M from obstacles)."""
    pgm, yaml = map_files(atc_dir)
    meta = {l.split(":")[0].strip(): l.split(":", 1)[1].strip() for l in yaml.read_text().split("\n") if ":" in l}
    assert float(meta["resolution"]) == RES and meta["origin"].startswith(f"[{OX}, {OY}"), meta
    img = np.array(Image.open(pgm))[::-1, :].copy()             # [iy, ix], y up
    _, pidx, x, y, _ = load_tracks(tracks_npz)
    NYP, NXP = img.shape
    NY, NX = NYP // CELL_PX, NXP // CELL_PX
    NYB, NXB = -(-NYP // BLOCK_PX), -(-NXP // BLOCK_PX)
    px = np.floor((x - OX) / RES).astype(np.int64)
    py = np.floor((y - OY) / RES).astype(np.int64)
    tracked = distinct_counts(pidx, px // CELL_PX, py // CELL_PX, NX, NY) >= MIN_IDS
    walked = distinct_counts(pidx, px // BLOCK_PX, py // BLOCK_PX, NXB, NYB) >= MIN_IDS
    walk_px = np.repeat(np.repeat(walked, BLOCK_PX, 0), BLOCK_PX, 1)[:NYP, :NXP]
    occ = (img < OCC_MAX_VALUE) & ~walk_px
    r = int(round(LOS_CLOSE_M / RES))
    los = ndi.binary_closing(np.pad(occ, r), structure=disk(r))[r:-r, r:-r] & ~walk_px
    clear = (ndi.distance_transform_edt(~los) * RES)[CELL_PX // 2::CELL_PX, CELL_PX // 2::CELL_PX][:NY, :NX]
    route_free = tracked & (clear >= ROUTE_CLEAR_M)
    return dict(occ=occ, los=np.ascontiguousarray(los), route_free=route_free)


def on_obstacles(occ, x, y):
    """Share of track samples on obstacle pixels (checks that a day's tracks are aligned with the map)."""
    NYP, NXP = occ.shape
    px = np.floor((x - OX) / RES).astype(np.int64)
    py = np.floor((y - OY) / RES).astype(np.int64)
    inmap = (px >= 0) & (px < NXP) & (py >= 0) & (py < NYP)
    return float(occ[py[inmap], px[inmap]].mean())


@njit(cache=True)
def blocked(occ, x0, y0, x1, y1):
    """True if an obstacle lies on the sight line; the last 0.25 m (the person) is not tested."""
    ny, nx = occ.shape
    gx0, gy0 = (x0 - OX) / RES, (y0 - OY) / RES
    gx1, gy1 = (x1 - OX) / RES, (y1 - OY) / RES
    L = math.hypot(gx1 - gx0, gy1 - gy0)
    n = int(L * 2.0) + 1
    skip = 0.25 / RES
    for i in range(1, n):
        s = i / n
        if (1.0 - s) * L <= skip:
            break
        px = int(math.floor(gx0 + s * (gx1 - gx0)))
        py = int(math.floor(gy0 + s * (gy1 - gy0)))
        if 0 <= px < nx and 0 <= py < ny and occ[py, px]:
            return True
    return False


def build_graph(free):
    """8-connected graph over the free cells, edge weights in m."""
    ny, nx = free.shape
    idx = -np.ones(free.shape, np.int64)
    cells = np.argwhere(free)
    idx[free] = np.arange(len(cells))
    rows, cols, w = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        cy, cx = cells[:, 0] + dy, cells[:, 1] + dx
        ok = (cy >= 0) & (cy < ny) & (cx >= 0) & (cx < nx)
        ok[ok] = free[cy[ok], cx[ok]]
        a, b = idx[cells[ok, 0], cells[ok, 1]], idx[cy[ok], cx[ok]]
        rows += [a, b]
        cols += [b, a]
        w += [np.full(a.size, CELL * math.hypot(dy, dx))] * 2
    G = csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(len(cells),) * 2)
    return G, cells


def seg_free(a, b, free):
    n = int(math.ceil(math.hypot(*(b - a)) / RES)) + 1
    pts = a + (b - a) * np.linspace(0, 1, n)[:, None]
    cx = np.floor((pts[:, 0] - OX) / CELL).astype(int)
    cy = np.floor((pts[:, 1] - OY) / CELL).astype(int)
    return bool(free[cy, cx].all())


def string_pull(pts, free):
    """Shorten a grid path by skipping points while the straight segment stays in free cells."""
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = i + 1
        while j + 1 < len(pts) and seg_free(pts[i], pts[j + 1], free):
            j += 1
        out.append(pts[j])
        i = j
    return np.array(out)


def resample(poly, step):
    seg = np.hypot(*np.diff(poly, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    q = np.arange(0.0, s[-1] + 1e-9, step)
    return np.c_[np.interp(q, s, poly[:, 0]), np.interp(q, s, poly[:, 1])], float(s[-1])
