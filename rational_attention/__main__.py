"""Command line:
  python -m rational_attention download  --atc DIR
  python -m rational_attention prepare   --atc DIR --days DIR
  python -m rational_attention calibrate --days DIR
  python -m rational_attention run       --days DIR --device xavier-nx-10w --cost energy --budget 5
"""
import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import atc
from .calibration import start_price, value_model
from .constants import MAP_DAY, N_CAMS
from .detector import CONFIG
from .mapping import build_map
from .replay import cost_vector, load_json, load_value_model, run_days
from .routes import prepare_job


def day_list(splits):
    days = load_json("days.json")
    return {d: v for s in splits for d, v in days[s].items()}


def cmd_download(a):
    parts = {v["part"] for v in day_list(("train", "test")).values()} | {1}
    atc.download(a.atc, parts)


def cmd_prepare(a):
    out = Path(a.days)
    out.mkdir(parents=True, exist_ok=True)
    map_path = out / "map.npz"
    if not map_path.exists():
        print("building the map", flush=True)
        np.savez_compressed(map_path, **build_map(a.atc, atc.day_file(a.atc, MAP_DAY, 1)))
    maps = {k: v.astype(bool) for k, v in np.load(map_path).items()}
    jobs = []
    for day, v in day_list(a.split).items():
        if (out / f"{day}.npz").exists():
            continue
        t0 = time.time()
        tracks = atc.day_file(a.atc, day, v["part"])
        print(f"{day}: tracks ready ({time.time() - t0:.0f} s)", flush=True)
        jobs.append((day, v["seed"], tracks, maps, out / f"{day}.npz"))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for day, rec in ex.map(prepare_job, jobs):
            print(f"{day}: " + ("not aligned with the map, skipped" if rec is None else
                               f"{rec['routes']} routes, {rec['km']:.1f} km, {rec['conflicts']} conflicts"), flush=True)


def paths(days_dir, split):
    return [str(Path(days_dir) / f"{d}.npz") for d in day_list((split,)) if (Path(days_dir) / f"{d}.npz").exists()]


def cmd_calibrate(a):
    train = paths(a.days, "train")
    model = value_model(train)
    (CONFIG / "calibration.json").write_text(json.dumps(model, indent=2) + "\n")
    print("value model: " + json.dumps(model), flush=True)
    model = load_value_model()
    starts = load_json("start_prices.json")
    for device in a.device or load_json("devices.json"):
        for kind in a.cost or ("time", "energy"):
            for B in a.budget or [float(b) for b in starts[device][kind]]:
                guess = starts.get(device, {}).get(kind, {}).get(f"{B:g}", -6.0)
                lp = start_price(train, cost_vector(device, kind), B, model, guess, a.passes)
                starts.setdefault(device, {}).setdefault(kind, {})[f"{B:g}"] = lp
                print(f"start price {device} {kind} {B:g}: log {lp:+.4f}", flush=True)
                (CONFIG / "start_prices.json").write_text(json.dumps(starts, indent=2) + "\n")


def cmd_run(a):
    test = paths(a.days, "test")
    model = load_value_model()
    dev = load_json("devices.json")[a.device]
    starts = load_json("start_prices.json")[a.device][a.cost]
    results = {}
    for B in a.budget:
        R = run_days(test, cost_vector(a.device, a.cost), B, model, starts[f"{B:g}"])
        secs, km, calls = R["seconds"].sum(), R["km"].sum(), R["calls"].sum(0)
        units = R["units"].sum() / secs
        row = dict(days=len(test), km=float(km), conflicts=int(R["conflicts"].sum()),
                   late_per_km=float(R["late"].sum() / km), units_per_s=float(units),
                   calls_per_s=float(calls.sum() / secs),
                   call_share={f"{n} camera{'s' if n > 1 else ''}": float(calls[N_CAMS == n].sum() / calls.sum())
                               for n in (1, 2, 3)})
        if a.cost == "energy":
            row["detector_power_w"] = units * dev["energy"]["one_camera_mj"] / 1000
            row["detector_energy_j_per_km"] = float(R["units"].sum() * dev["energy"]["one_camera_mj"] / 1000 / km)
        else:
            row["detector_busy_share"] = units * dev["time"]["one_camera_ms"] / 1000
        results[f"{B:g}"] = row
        print(f"{a.device} {a.cost} budget {B:g}: " + json.dumps(row), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.device}_{a.cost}.json").write_text(json.dumps(results, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(prog="python -m rational_attention")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("download", help="download the ATC map and tracking archives")
    p.add_argument("--atc", default="data/atc")
    p = sub.add_parser("prepare", help="build the map and sample the robot routes of each day")
    p.add_argument("--atc", default="data/atc")
    p.add_argument("--days", default="data/days")
    p.add_argument("--split", nargs="+", default=["train", "test"])
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("calibrate", help="fit the value model and the start prices on the training days")
    p.add_argument("--days", default="data/days")
    p.add_argument("--device", nargs="+")
    p.add_argument("--cost", nargs="+", choices=("time", "energy"))
    p.add_argument("--budget", nargs="+", type=float)
    p.add_argument("--passes", type=int, default=1)
    p = sub.add_parser("run", help="run the scheduler on the test days")
    p.add_argument("--days", default="data/days")
    p.add_argument("--device", default="xavier-nx-10w")
    p.add_argument("--cost", default="energy", choices=("time", "energy"))
    p.add_argument("--budget", nargs="+", type=float, default=[5.0])
    p.add_argument("--out", default="results")
    a = ap.parse_args()
    dict(download=cmd_download, prepare=cmd_prepare, calibrate=cmd_calibrate, run=cmd_run)[a.cmd](a)


if __name__ == "__main__":
    main()
