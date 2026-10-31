"""Run the scheduler over prepared days. One robot drives a day's routes in order of start time and carries its price
from route to route."""
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .constants import DT, ERR_TOL, MASK, Q_ACC
from .detector import CONFIG, load_detector
from .routes import Day
from .scheduler import simulate


def load_json(name):
    return json.loads((CONFIG / name).read_text())


def load_value_model(path=CONFIG / "calibration.json"):
    c = json.loads(Path(path).read_text())
    return dict(h=np.array(c["h_per_s"]), lam_a=np.array(c["lam_a"]), lam_b=np.array(c["lam_b"]), h_u=c["h_u"],
                rbar=c["rbar"], l_new=np.array(c["l_new"]))


def cost_vector(device, kind):
    """Cost units of each call (F, L, R, FL, FR, LR, FLR); a one-camera call costs 1."""
    c2, c3 = load_json("devices.json")[device][kind]["relative"]
    return np.array([1.0, 1.0, 1.0, c2, c2, c2, c3])


def run_day(path, cost, B, model, log_price0):
    """Per route (in file order): late conflicts, conflicts, calls of each kind, units spent, mean log price."""
    D = Day(path)
    recall = load_detector()["recall"]
    out = dict(late=np.zeros(D.n, np.int64), conflicts=np.zeros(D.n, np.int64), calls=np.zeros((D.n, MASK.size), np.int64),
               units=np.zeros(D.n), log_price=np.zeros(D.n), seconds=np.zeros(D.n), km=np.zeros(D.n))
    lp = log_price0
    for r in D.order():
        d = D.route(r)
        err, used, lp_mean, lp, spent = simulate(
            B, cost, Q_ACC, d["P"], d["heading"], d["tick_off"], d["ent_view"], d["ent_det"], d["ent_pl"], d["ent_x"],
            d["ent_y"], d["ent_zx"], d["ent_zy"], d["ent_d"], d["n_local"], d["kd_of"], d["n_conf"],
            d["conf_idx_of"], recall, model["h"], model["lam_a"], model["lam_b"], model["h_u"], model["rbar"],
            model["l_new"], math.exp(lp))
        e = err[d["elig"]]
        out["late"][r] = int((~(e <= ERR_TOL)).sum())
        out["conflicts"][r] = e.size
        out["calls"][r] = used
        out["units"][r] = spent
        out["log_price"][r] = lp_mean
        out["seconds"][r] = (d["tick_off"].size - 1) * DT
        out["km"][r] = d["length_m"] / 1000
    return out


def run_days(paths, cost, B, model, log_price0, workers=None):
    """run_day on each day in parallel; arrays concatenated over days."""
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(run_day, paths, [cost] * len(paths), [B] * len(paths), [model] * len(paths),
                          [log_price0] * len(paths)))
    return {k: np.concatenate([r[k] for r in res]) for k in res[0]}
