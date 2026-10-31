"""ATC shopping centre pedestrian tracking dataset (Brscic et al., IEEE THMS 2013):
https://dil.atr.jp/crest2010_HRI/ATC_dataset/ (free for research use)."""
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = "https://dil.atr.jp/crest2010_HRI/ATC_dataset/"


def download(atc_dir, parts):
    """Download the map and the tracking archives of the given parts (about 4-5 GB each)."""
    dl = Path(atc_dir) / "download"
    dl.mkdir(parents=True, exist_ok=True)
    for name in ["ATC-map.zip"] + [f"atc-tracking-part{p}.7z" for p in sorted(set(parts))]:
        out = dl / name
        if out.exists():
            continue
        print(f"downloading {name}", flush=True)
        tmp = out.with_suffix(out.suffix + ".part")
        with urllib.request.urlopen(URL + name) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 22)
        tmp.rename(out)


def map_files(atc_dir):
    """Paths of the occupancy grid (pgm) and its metadata (yaml), extracted from ATC-map.zip if needed."""
    ex = Path(atc_dir) / "extracted"
    pgm, yaml = ex / "localization_grid.pgm", ex / "localization_grid.yaml"
    if not (pgm.exists() and yaml.exists()):
        ex.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(Path(atc_dir) / "download" / "ATC-map.zip") as z:
            z.extractall(ex)
    return pgm, yaml


def seven_zip():
    exe = os.environ.get("SEVEN_ZIP") or shutil.which("7z") or shutil.which("7za")
    if exe is None and Path(r"C:\Program Files\7-Zip\7z.exe").exists():
        exe = r"C:\Program Files\7-Zip\7z.exe"
    if exe is None:
        raise RuntimeError("7-Zip not found: install it or set SEVEN_ZIP to the 7z executable")
    return exe


def day_file(atc_dir, day, part):
    """Tracks of one day as npz (t, id, x, y), converted from the day's CSV in its tracking archive on first use."""
    npz = Path(atc_dir) / "derived" / f"atc-{day}.npz"
    if npz.exists():
        return npz
    import pandas as pd
    ex = Path(atc_dir) / "extracted"
    csv = ex / f"atc-{day}.csv"
    if not csv.exists():
        archive = Path(atc_dir) / "download" / f"atc-tracking-part{part}.7z"
        subprocess.run([seven_zip(), "e", "-y", f"-o{ex}", str(archive), csv.name], check=True,
                       stdout=subprocess.DEVNULL)
    df = pd.read_csv(csv, header=None, names=["t", "id", "x", "y", "z", "v", "ang", "face"],
                     usecols=["t", "id", "x", "y"], dtype={"t": np.float64, "id": np.int64, "x": np.int32, "y": np.int32})
    df = df[df.id >= 0]
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, t=df.t.values, id=df.id.values, x=df.x.values, y=df.y.values)
    csv.unlink()
    return npz


def load_tracks(npz):
    """Time-sorted samples: t (s), person index, x, y (m), number of people."""
    d = np.load(npz)
    order = np.argsort(d["t"], kind="stable")
    t = d["t"][order]
    uid, pidx = np.unique(d["id"][order], return_inverse=True)
    return t, pidx.astype(np.int64), d["x"][order] / 1000.0, d["y"][order] / 1000.0, uid.size
