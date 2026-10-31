"""Live view of the scheduler on one route.
Map: the robot, its route and its three camera regions (a region lights up when the detector processes that camera),
pedestrians (dark grey in view, light grey out of view or hidden; orange edge: conflict still ahead), detections (+),
the robot's tracks in range (2-sigma ellipses; hollow circles when wider than 6 m) and each conflict's outcome at its
deadline (green: tracked in time, red: late).
Right: each camera's value of a look against the price of a one-camera call, the cameras processed so far, and the
price over time."""
import math
from pathlib import Path

import numpy as np

from .constants import ACTIONS, DT, ERR_TOL, HALF_VIEW, MASK, N_VIEWS, OPTION_SETS, OX, OY, Q_ACC, RANGE_M, RES
from .detector import load_detector
from .replay import cost_vector, load_json, load_value_model
from .routes import Day
from .scheduler import simulate_recorded

CAM_COLORS = ("#0f7f6f", "#6a3d9a", "#b2182b")          # front, left, right
CAM_NAMES = ("front", "left", "right")
CONFLICT_COLOR = "#e08214"
WINDOW_M = 15.0                                         # half width of the map view around the robot
MARK_TICKS = 20                                         # how long a deadline outcome stays on the map
MAX_ELLIPSE_M = 6.0


def busiest_route(D):
    """The route with the most conflicts."""
    off = D.z["elig_off"]
    return int(np.argmax([D.z["elig"][off[r]:off[r + 1]].sum() for r in range(D.n)]))


def record(days_dir, day, route, device, cost, B):
    D = Day(Path(days_dir) / f"{day}.npz")
    route = busiest_route(D) if route is None else route
    d = D.route(route)
    model = load_value_model()
    lp0 = load_json("start_prices.json")[device][cost][f"{B:g}"]
    out = simulate_recorded(B, cost_vector(device, cost), Q_ACC, d["P"], d["heading"], d["tick_off"], d["ent_view"],
                            d["ent_det"], d["ent_pl"], d["ent_x"], d["ent_y"], d["ent_zx"], d["ent_zy"], d["ent_d"],
                            d["n_local"], d["kd_of"], d["n_conf"], d["conf_idx_of"], load_detector()["recall"],
                            model["h"], model["lam_a"], model["lam_b"], model["h_u"], model["rbar"], model["l_new"],
                            math.exp(lp0))
    for k, v in zip(("err", "used", "lp_mean", "lp_end", "spent", "opt", "val", "log_price", "pos", "cov"), out):
        d[k] = v
    d.update(day=day, route=route, device=device, cost=cost, budget=B)
    return d


def ellipses(pos, cov):
    """Centres, widths, heights and angles (deg) of 2-sigma ellipses."""
    a, b, c = cov[:, 0], cov[:, 1], cov[:, 2]
    disc = np.sqrt(np.maximum((a - c) ** 2 / 4 + b * b, 0.0))
    l1 = np.maximum((a + c) / 2 + disc, 0.0)
    l2 = np.maximum((a + c) / 2 - disc, 0.0)
    ang = np.degrees(0.5 * np.arctan2(2 * b, a - c))
    return pos, np.minimum(4 * np.sqrt(l1), 4 * WINDOW_M), np.minimum(4 * np.sqrt(l2), 4 * WINDOW_M), ang


def animate(d, occ, save=None, speed=1.0, start=0.0, duration=None):
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import EllipseCollection
    from matplotlib.patches import Polygon, Wedge

    K = d["tick_off"].size - 1
    P, hd = d["P"], d["heading"]
    t = np.arange(K) * DT
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(3, 2, width_ratios=(1.5, 1), height_ratios=(1.3, 0.55, 1), hspace=0.55, wspace=0.12,
                          left=0.01, right=0.98, top=0.92, bottom=0.08)
    ax = fig.add_subplot(gs[:, 0])
    ax_v, ax_c, ax_p = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 1])
    fig.suptitle(f"day {d['day']}, route {d['route']}: {d['device']}, {d['cost']} costs, budget {d['budget']:g} units/s",
                 fontsize=10)

    # map
    ny, nx = occ.shape
    ax.imshow(np.where(occ, 0.72, 1.0), cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest",
              extent=(OX, OX + nx * RES, OY, OY + ny * RES))
    ax.plot(P[:, 0], P[:, 1], ls="--", color="0.45", lw=1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    wedges = [Wedge((0, 0), RANGE_M, 0, 1, facecolor=c, edgecolor=c, alpha=0.08, lw=0.8) for c in CAM_COLORS]
    for w in wedges:
        ax.add_patch(w)
    robot = Polygon(np.zeros((3, 2)), closed=True, facecolor="black", zorder=6)
    ax.add_patch(robot)
    people = ax.scatter([], [], s=22, zorder=4)
    dets = ax.scatter([], [], marker="+", s=40, lw=1.2, zorder=5)
    marks = ax.scatter([], [], s=90, marker="o", facecolors="none", lw=2.0, zorder=7)
    tracks = [None]
    vague = ax.scatter([], [], s=30, facecolors="none", edgecolors="#1f5fa8", lw=0.9, zorder=3)
    info = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left", fontsize=9, family="monospace",
                   bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9), zorder=10)
    ax.legend(handles=[plt.Line2D([], [], color=c, lw=6, alpha=0.5, label=n) for c, n in zip(CAM_COLORS, CAM_NAMES)],
              loc="lower right", fontsize=8, framealpha=0.9)

    # camera values against the price
    vmax = max(np.nanmax(d["val"][:, 3:]), np.exp(d["log_price"]).max()) * 3
    vmin = np.exp(d["log_price"]).min() / 30
    bars = ax_v.bar(range(N_VIEWS), np.zeros(N_VIEWS), color=CAM_COLORS, bottom=vmin)
    price_line = ax_v.axhline(np.exp(d["log_price"][0]), color="black", lw=1.2, label="price of a one-camera call")
    ax_v.set_yscale("log")
    ax_v.set_ylim(vmin, vmax)
    ax_v.set_xticks(range(N_VIEWS), CAM_NAMES)
    ax_v.set_title("value of a look", fontsize=9)
    ax_v.legend(loc="upper right", fontsize=8)

    # cameras processed
    cover = np.array([np.bitwise_or.reduce([MASK[a] for a in o if a >= 0] or [0]) for o in OPTION_SETS])
    looked = ((cover[d["opt"]][:, None] >> np.arange(N_VIEWS)) & 1).astype(bool)      # [tick, camera]
    rgb = np.array([matplotlib.colors.to_rgb(c) for c in CAM_COLORS])
    strip = np.where(looked.T[:, :, None], rgb[:, None, :], 1.0)                     # [camera, tick, rgb]
    shown = np.ones_like(strip)
    strip_img = ax_c.imshow(shown, aspect="auto", interpolation="nearest", extent=(0, K * DT, N_VIEWS - 0.5, -0.5))
    ax_c.set_yticks(range(N_VIEWS), CAM_NAMES, fontsize=8)
    ax_c.set_title("cameras processed", fontsize=9)

    # price
    ax_p.set_xlim(0, t[-1])
    lp = d["log_price"]
    ax_p.set_ylim(np.exp(lp.min()) * 0.9, np.exp(lp.max()) * 1.1)
    ax_p.set_title("price", fontsize=9)
    ax_p.set_xlabel("time (s)")
    (price_trace,) = ax_p.plot([], [], color="black", lw=1.2)

    err, kd_of, conf_idx_of = d["err"], d["kd_of"], d["conf_idx_of"]
    outcome = []                                            # (x, y, in time, last tick shown)
    tally = [0, 0, -1]                                      # deadlines so far, late so far, last tick counted

    def frame(k):
        x0, y0 = P[k]
        h = hd[k]
        ax.set_xlim(x0 - WINDOW_M, x0 + WINDOW_M)
        ax.set_ylim(y0 - WINDOW_M * 0.8, y0 + WINDOW_M * 0.8)
        deg = math.degrees(h)
        span = math.degrees(HALF_VIEW)
        for w, (a0, a1), on in zip(wedges, ((deg - span, deg + span), (deg + span, deg + 180), (deg - 180, deg - span)),
                                   looked[k]):
            w.set_center((x0, y0))
            w.set_theta1(a0)
            w.set_theta2(a1)
            w.set_alpha(0.32 if on else 0.06)
        c_, s_ = math.cos(h), math.sin(h)
        robot.set_xy([[x0 + 0.9 * c_, y0 + 0.9 * s_], [x0 - 0.5 * c_ - 0.45 * s_, y0 - 0.5 * s_ + 0.45 * c_],
                      [x0 - 0.5 * c_ + 0.45 * s_, y0 - 0.5 * s_ - 0.45 * c_]])
        e = np.arange(d["tick_off"][k], d["tick_off"][k + 1])
        pl = d["ent_pl"][e]
        view = d["ent_view"][e]
        people.set_offsets(np.c_[d["ent_x"][e], d["ent_y"][e]])
        people.set_facecolors(np.where(view[:, None] >= 0, [[0.33, 0.33, 0.33, 1]], [[0.75, 0.75, 0.75, 1]]))
        ahead = kd_of[pl] >= k
        people.set_edgecolors([CONFLICT_COLOR if a else "none" for a in ahead])
        people.set_linewidths(np.where(ahead, 1.8, 0.0))
        seen = (view >= 0) & d["ent_det"][e] & looked[k][np.maximum(view, 0)]
        dets.set_offsets(np.c_[d["ent_zx"][e][seen], d["ent_zy"][e][seen]] if seen.any() else np.zeros((0, 2)))
        dets.set_color([CAM_COLORS[v] for v in view[seen]])
        if tracks[0] is not None:
            tracks[0].remove()
        pos = d["pos"][k]
        ok = ~np.isnan(pos[:, 0])
        ok[ok] = np.hypot(pos[ok, 0] - x0, pos[ok, 1] - y0) <= RANGE_M
        cen, wd, ht, ang = ellipses(pos[ok], d["cov"][k][ok])
        tight = wd <= MAX_ELLIPSE_M
        tracks[0] = EllipseCollection(wd[tight], ht[tight], ang[tight], units="xy", offsets=cen[tight],
                                      offset_transform=ax.transData, facecolors="none", edgecolors="#1f5fa8",
                                      linewidths=0.9, zorder=3)
        ax.add_collection(tracks[0])
        vague.set_offsets(cen[~tight] if (~tight).any() else np.zeros((0, 2)))
        for kk in range(tally[2] + 1, k + 1):
            ee = np.arange(d["tick_off"][kk], d["tick_off"][kk + 1])
            for j in ee[kd_of[d["ent_pl"][ee]] == kk]:
                in_time = bool(err[conf_idx_of[d["ent_pl"][j]]] <= ERR_TOL)
                outcome.append((d["ent_x"][j], d["ent_y"][j], in_time, kk + MARK_TICKS))
                tally[0] += 1
                tally[1] += not in_time
        tally[2] = max(tally[2], k)
        outcome[:] = [o for o in outcome if o[3] >= k]
        marks.set_offsets(np.array([o[:2] for o in outcome]) if outcome else np.zeros((0, 2)))
        marks.set_edgecolors(["#1a9641" if o[2] else "#d7191c" for o in outcome])
        calls = " + ".join(ACTIONS[a] for a in OPTION_SETS[d["opt"][k]] if a >= 0) or "none"
        info.set_text(f"t {t[k]:5.1f} s   calls: {calls:<10s}\nconflicts {tally[0]}, late {tally[1]}")
        for b, v in zip(bars, d["val"][k, 3:]):
            b.set_height(max(v, vmin) - vmin)
        for b, on in zip(bars, looked[k]):
            b.set_edgecolor("black" if on else "none")
            b.set_linewidth(1.5)
        price_line.set_ydata([math.exp(d["log_price"][k])] * 2)
        shown[:, :k + 1] = strip[:, :k + 1]
        strip_img.set_data(shown)
        price_trace.set_data(t[:k + 1], np.exp(lp[:k + 1]))
        return ()

    step = max(1, int(round(speed))) if save else 1
    k0 = min(int(round(start / DT)), K - 1)
    k1 = K if duration is None else min(K, k0 + int(round(duration / DT)))
    if k0 > 0:
        frame(k0 - 1)
    anim = FuncAnimation(fig, frame, frames=range(k0, k1, step), interval=1000 * DT * step / speed, blit=False,
                         repeat=False)
    if save:
        fps = 1.0 / (DT * step / speed)
        anim.save(save, fps=fps, dpi=80, writer="pillow" if str(save).endswith(".gif") else None)
        print(f"saved {save}", flush=True)
    else:
        plt.show()
