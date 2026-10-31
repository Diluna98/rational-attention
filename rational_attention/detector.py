"""Person detector model: detection probability by apparent height in the image, with misses that persist over time
(a per-person score follows an AR(1) process; a detection is a score above the bin's threshold)."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from .constants import DT

CONFIG = Path(__file__).resolve().parent.parent / "config"


def load_detector(path=CONFIG / "detector.json"):
    c = json.loads(Path(path).read_text())
    recall = np.array(c["recall"], np.float64)
    return dict(recall=recall, thr=norm.ppf(1.0 - recall), rho_frame=float(c["score_corr_1s"] ** DT))
