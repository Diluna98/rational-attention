# rational-attention

Budgeted camera attention for a mobile robot with three cameras (front, left, right) and one person detector.
Every 0.1 s the scheduler decides which camera images the detector processes, in one batched call or in separate
calls, so that the pedestrians the robot may run into stay well tracked while the average compute spent stays at a
budget.

## Method

- **Tracking.** A constant-velocity Kalman filter per detected pedestrian.
- **Value of a look at a tracked pedestrian.** For each tracked pedestrian within range,

  V = h(δ) · r(d) · Σ<sub>s</sub> exp(−f r s) · [q(Σ, s) − q(Σ⁺, s)],  s = 0.5, 1.5, …, 5.5 s

  where q is the probability that the predicted position is more than 0.5 m off s seconds ahead, Σ the predicted
  covariance, Σ⁺ the covariance after a detection now, r(d) the detection probability at distance d, f the camera's
  current processing rate and h(δ) the conflict rate of pedestrians at distance δ from the route.
- **Value of a look for pedestrians not yet tracked.** Each camera keeps the expected number of undetected
  pedestrians that will come close to the robot. It grows with their arrival rate, which depends on how many
  pedestrians are tracked, and shrinks after every look.
- **Choice.** Each tick the scheduler takes, among the 15 ways to cover disjoint cameras with up to three calls (1, 2
  or 3 cameras per call), the one with the largest total value minus price × cost, or no call if none is positive.
- **Price.** The log price moves with the spending: up when it is above the budget, down when it is below, with a
  5-minute time constant. It is carried from route to route.
- **Costs.** One detector call on one camera costs 1 unit; calls on two and three cameras cost the measured ratios of
  each device (`config/devices.json`).

## Setup

Python 3.10 or newer and [7-Zip](https://www.7-zip.org/) (`7z` on the PATH, or the `SEVEN_ZIP` environment variable).

```bash
pip install -r requirements.txt
```

## Data

The replay uses the [ATC pedestrian tracking dataset](https://dil.atr.jp/crest2010_HRI/ATC_dataset/) (free for
research use; please cite D. Brščić, T. Kanda, T. Ikeda, T. Miyashita, "Person position and body direction tracking in
large public spaces using 3D range sensors", IEEE Transactions on Human-Machine Systems, 43(6), 2013).
`config/days.json` lists the days: 2 training days and 20 test days, from tracking archives 1, 2, 4, 6, 7 and 9
(about 26 GB to download).

## Usage

```bash
python -m rational_attention download --atc data/atc
python -m rational_attention prepare --atc data/atc --days data/days
python -m rational_attention run --days data/days --device xavier-nx-10w --cost energy --budget 5
```

`prepare` builds the map and samples 400 robot routes per day (shortest paths between random walkable cells, driven
at 1 m/s from a random start time). For every pedestrian sample along a route it stores which camera sees it,
whether the detector finds it and a noisy detected position. Detection follows the measured detection probability by
apparent person height in `config/detector.json`, with misses that persist over time.

`run` drives the robot through the test days with the scheduler and writes `results/<device>_<cost>.json`:

- `late_per_km`: late conflicts per km. A conflict is a pedestrian who comes within 0.6 m of the robot. It is late if,
  2 s before, the pedestrian's tracked position is more than 0.5 m off (or the pedestrian was never detected).
- `units_per_s`, `calls_per_s`, `call_share`: compute spent and how it was used.
- `detector_power_w` and `detector_energy_j_per_km` (energy costs) or `detector_busy_share` (time costs).

Devices: `xavier-nx-10w`, `xavier-nx-20w`, `rtx-4050`, `i3-1315u`; costs: `time` or `energy`; budgets 3.5 and 5
units/s for every cost table, and also 2.5, 3, 4, 6 and 7 for the energy tables.

The value model and the start prices in `config/` were fitted on the training days. To fit them again:

```bash
python -m rational_attention calibrate --days data/days
```

## Code

| File | Contents |
|------|----------|
| `rational_attention/scheduler.py` | the scheduler: values, choice of calls, price |
| `rational_attention/value.py` | value of a look at a tracked pedestrian |
| `rational_attention/tracker.py` | Kalman filter and detection updates |
| `rational_attention/calibration.py` | value model and start prices from the training days |
| `rational_attention/replay.py` | runs the scheduler over prepared days |
| `rational_attention/routes.py` | route sampling, camera visibility, detections, conflicts |
| `rational_attention/mapping.py` | map, line of sight, route planning graph |
| `rational_attention/detector.py` | detector model |
| `rational_attention/atc.py` | dataset download and conversion |
| `rational_attention/constants.py` | all fixed settings |
