import math

import numpy as np

# time and robot
DT = 0.1                         # s per tick
ROBOT_SPEED = 1.0                # m/s
STEP_M = DT * ROBOT_SPEED        # path sample spacing (m)
MIN_ROUTE_M = 15.0

# cameras: front within +-60 deg of the heading, left and right cover the rest
RANGE_M = 13.1
HALF_VIEW = math.radians(60.0)
N_VIEWS = 3                      # 0 front, 1 left, 2 right

# detector calls: one, two or three cameras per call (bit v = camera v)
ACTIONS = ("F", "L", "R", "FL", "FR", "LR", "FLR")
MASK = np.array([1, 2, 4, 3, 5, 6, 7], np.int64)
N_CAMS = np.array([1, 1, 1, 2, 2, 2, 3], np.int64)
# every way to cover disjoint cameras with up to three calls (-1 = none)
OPTION_SETS = np.array([[-1, -1, -1], [0, -1, -1], [1, -1, -1], [2, -1, -1], [3, -1, -1], [4, -1, -1], [5, -1, -1],
                        [6, -1, -1], [0, 1, -1], [0, 2, -1], [1, 2, -1], [0, 5, -1], [1, 4, -1], [2, 3, -1],
                        [0, 1, 2]], np.int64)

# pedestrian motion and detection noise
SIGMA_A = 0.5                    # acceleration noise (m/s^2)
Q_ACC = SIGMA_A ** 2
SIGMA_V0 = 1.0                   # velocity std of a new track (m/s)
SIG_LAT = 0.035                  # position noise across the line of sight (m)

# apparent person height in the image decides detection probability
FOCAL_PX = 771.0
PERSON_M = 1.7
HEIGHT_EDGES_PX = (100.0, 200.0, 400.0)

# conflicts: a pedestrian within CONFLICT_M of the robot; it must be tracked within ERR_TOL, WARN_TICKS before
CONFLICT_M = 0.6
WARN_TICKS = 20
ERR_TOL = 0.5
MIN_K = 20                       # ignore conflicts in the first 2 s of a route
MIN_AGE_S = 2.0                  # ignore pedestrians that appeared less than 2 s before the conflict

# value model
H_EDGES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 1e9])     # distance to the route (m)
L_EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 1e9])                    # time from detection to deadline (s)
N_HORIZON = 6                    # deadlines 0.5, 1.5, ..., 5.5 s ahead
PATH_LOOKAHEAD = 10              # route points checked for the distance to the route, every 0.5 s
HORIZON_TICKS = 60               # conflict horizon used to fit h (6 s)

# price
PRICE_STEP = 1.0 / 3000.0        # step of the log price per unit of overspending
RATE_TAU = 2.0                   # time constant of the camera processing rates (s)

# map (ATC occupancy grid)
RES, OX, OY = 0.05, -60.0, -40.0
CELL_PX, BLOCK_PX = 5, 3
CELL = RES * CELL_PX
MIN_IDS = 20                     # people per cell for a cell to count as walkable
OCC_MAX_VALUE = 89               # grid values below this are obstacles
LOS_CLOSE_M = 0.15               # closing radius of the line-of-sight map
ROUTE_CLEAR_M = 0.5              # route clearance from obstacles
ALIGN_MAX = 0.02                 # max share of track samples on obstacles for a day to be used
MAP_DAY = "20121114"

# days
ROUTES_PER_DAY = 400
SEED = 20261027
