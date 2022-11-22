import copy
import csv
from dataclasses import dataclass
import json
import math
import os
from os.path import abspath, isdir
import platform
import random
import shutil
from enum import Enum

import gym
import matplotlib
import matplotlib.pyplot as plt
import mmgdynamics as mmg
import rivergen as rg
import numpy as np
from multiprocessing import Pool
from gym import spaces
from gym.utils import seeding
from matplotlib import cm, transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from mmgdynamics.structs import RudderAngle, Surge, Sway, YawRate

try:
    from mmgdynamics.calibrated_vessels import SPTRR1
    from mmgdynamics.structs import InitialValues, InlandVessel
except ImportError:
    from mmgdynamics.structs import Vessel, InitialValues
    from mmgdynamics.calibrated_vessels import kvlcc2_fully_loaded

from collections import deque
from getpass import getuser
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

# --------- Global config -------------------

PI = math.pi
TWOPI = 2*PI
TRAIN_ON_TAURUS: bool = False


# Options for river generation
rg.options.VERBOSE = False
rg.options.GP = 101

if TRAIN_ON_TAURUS:
    RIVERDATA = abspath("/home/s2075466/riverdata")
    matplotlib.use("Agg")
else:
    PREFIX = "Users" if platform.system() == "Windows" else "home"
    matplotlib.use("TKAgg")

@dataclass(frozen=True)
class DATAPATHS:
    lr = abspath(f"/{PREFIX}/{getuser()}/Dropbox/TU Dresden/riverdata/lr")
    mr = abspath(f"/{PREFIX}/{getuser()}/Dropbox/TU Dresden/riverdata/mr")
    straight = abspath(f"/{PREFIX}/{getuser()}/Dropbox/TU Dresden/segGen/scenes/straight")
    straight_bow = abspath(f"/{PREFIX}/{getuser()}/Dropbox/TU Dresden/segGen/scenes/current_to_bow")
    straight_stern = abspath(f"/{PREFIX}/{getuser()}/Dropbox/TU Dresden/segGen/scenes/current_to_stern")

# Train on generated river.
# Validate on lower rhine
TRAIN = False

WITHFRAMES = True

# Number of evenly spaced gridpoints for which stream
# velocity and water depth data is available
N_GRIDPOINTS = 26 if not TRAIN else rg.options.GP

# Types
State = TypeVar("State")
@dataclass
class Point:
    x: np.float64
    y: np.float64

class SCENARIO(Enum):
    DUSSELDORF = 4200
    LR_STRAIGHT = 7300
    LORELEI = 2500
    DOWNSTREAM = 2
    LR_UPSTREAM = 9800
    MR_UPSTREAM = 2860

# --------- Gym Env ----------------------------------------------

class PathFollower(gym.Env):
    def __init__(self, direction: int, epi_steps: int = np.inf, mode = "step") -> None:
        super().__init__()

        assert direction in [-1,1], "Invalid direction"
        assert mode in ["step","cont"]
        self.DIR = direction
        self.mode = mode

        # All Gridpoints are BASEPOINT_DIST meters apart
        # to span a maximum rhine width of 500 meters
        self.BASEPOINT_DIST: int = 20

        self.K: float = 0.005 # Convergence rate of vector field for vector field guidance
        self.C: float = -1 # Derivative pentalty constant
        self.T = 0.01 # Constant for tanh of cte

        # Rudder increment/decrement per action [deg]
        RUDDER_INCR: int = 2
        self.RUDDER_INCR = to_rad(RUDDER_INCR)

        # Maximum possible rudder angle [deg]
        MAX_RUDDER: int = 20
        self.MAX_RUDDER = to_rad(MAX_RUDDER)

        # Minimum water under keel [m]
        self.MIN_UNDER_KEEL: float = 1.5

        # Current timestep of episode
        self.timestep: int = 1

        # Prepare the river data by importing it from file
        if TRAIN:
            self.RIVERPATH = None
        else:
            self.set_arrs(*import_river(DATAPATHS.mr))
            self.waypoint_idx = SCENARIO.LORELEI.value

        # Path roughness (Use only every nth data point of the path)
        self.SMOOTHNESS: int = 5

        # Render distance for faster rendering
        self.lookahead = 80
        self.lookbehind = 60

        # Vessel set-up ------------------------------------------------
        self.delta = 0.0 # Rudder angle [rad]
        self.nps = 4.0 # Propeller revolutions [s⁻¹]
        self.speed = 0# Overall vessel speed [m/s]

        # Vessel to be simulated
        #self.vessel: InlandVessel = InlandVessel(**SPTRR1)
        self.vessel: Vessel = Vessel(**kvlcc2_fully_loaded)

        # Movement heading can include drift angles
        self._movement_heading = 0.0

        # Heading, the longitudinal vessel axis points to
        self._aghead = 0.0

        # History object for value logging
        self.history: History = History()

        # PID specific constants ----------------------------------------------
        self.K_p = 2.8 # Proportional gain
        self.K_d = 64.46 # Derivative gain
        self.K_i = 0.0 # Integrator gain

        # self.K_p = 170/180*PI # Proportional gain
        # self.K_d = 16 # Derivative gain
        # self.K_i = 0.042/180*PI # Integrator gain

        self.integator_sum = 0.0 # Sum of the integrator over time
        self.pid_error = 0.0 # Error between
        self.commanded_rudder_angle = 0.0 # PID commanded rudder angle in [deg]
        self.r_lt = 0.0

        # Gym Action and observation space -------------------------------------
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=float
        )

        # Set action space according to mode
        self.n_actions = 3 if self.mode == "step" else 1
        if self.mode == "step":
            self.action_space = spaces.Discrete(self.n_actions)
        else:
            self.action_space = spaces.Box(
                low=np.array([-1.]),
                high=np.array([1.])
            )

        self.max_episode_steps = epi_steps

    # Main properties of the environment ----------------------------
    @property
    def rel_current_attack_angle(self) -> float:
        """Angle of attack for current relative to bow [rad]"""
        return self.rel_ang_diff(self.aghead, self.strdir)

    @property
    def state(self) -> State:
        """Current environment state"""
        return self._state

    @property
    def ivs(self) -> InitialValues:
        """Initial values for numerical integration"""
        return self._ivs

    @ivs.setter
    def ivs(self,value):
        """Set initial values for numerical integration"""
        self._ivs = value

    @property
    def strvel(self) -> float:
        """Stream velocity in [m/s]"""
        return self._strvel

    @property
    def strdir(self) -> float:
        """Attack angle relative to bow in [rad]"""
        return self._strdir

    @property
    def wd(self) -> float:
        """Water depth under vessel in [m]"""
        return self._wd

    @property
    def agpos(self) -> Point:
        """Agent position as Point"""
        return self._agpos

    @agpos.setter
    def agpos(self, val) -> Point:
        """Set the agent position"""
        self._agpos = val

    @property
    def aghead(self) -> float:
        """Agent heading in [rad]"""
        return self._aghead

    @property
    def cte(self) -> float:
        """Cross-track error in [m]"""
        return self._cte

    @property
    def desired_course(self) -> float:
        """Desired course in [rad]"""
        return self._dc

    @property
    def heading_error(self) -> float:
        """Error from current to desired heading in [rad]"""
        return self.rel_ang_diff(self.desired_course, self.movement_heading)

    @property
    def movement_heading(self) -> float:
        """Heading of vessel considering drift angle"""
        return self._movement_heading

    @property
    def cra(self) -> float:
        """"Commanded rudder angle in [rad]"""
        return self._cra

    # ------------------------------------------------------------------

    def reset(
        self,*,
        seed: Optional[int] = None,
        return_info: Optional[bool] = False,
        options: Optional[dict] = {}
        ) -> Union[State, Tuple[State,dict]]:

        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        self.history = History()

        # Generate new random river if training
        if TRAIN:
            if self.RIVERPATH is not None:
                delete_river(self.RIVERPATH)
            self.RIVERPATH = random_river(20,1,1.5)
            self.set_arrs(*import_river(abspath(f"./{self.RIVERPATH}")))

        def start_x(x): return self.path["x"][x]
        def start_y(y): return self.path["y"][y]

        # Index list of the path to follow
        self.path_index: np.ndarray = np.empty(self._wdarray.shape[0], dtype=int)

        # The path_index variable gets filled here.
        self.path = self.construct_path()
        self.red_path = self.smooth(
            copy.deepcopy(self.path),self.SMOOTHNESS, alpha=0.1
        )
        self.history.save_path(self)

        # Get the border to port and starboard where
        # the water depth equals the ship draft
        self.build_fairway_border()


        # Get the index of waypoint behind agent
        self.lwp_idx = self.get_lwp_idx(self.waypoint_idx)

        # Last waypoint and next waypoint
        self.lwp = self.get_red_wp(self.lwp_idx)
        self.nwp = self.get_red_wp(self.lwp_idx, 1)

        # Agent position plus some random noise
        self._agpos = Point(
            (start_x(self.waypoint_idx) + random.uniform(-20, 20)),
            (start_y(self.waypoint_idx) + random.uniform(-10, 10)))

        self._agpos = Point(
            (start_x(self.waypoint_idx)),(start_y(self.waypoint_idx))
        )

        # Set agent heading to path heading plus some small noise (5° = 1/36*pi rad)
        random_angle = 2/36*PI
        self._aghead = self.path_angle(p1=self.lwp, p2=self.nwp)# + random_angle

        # Drift angle == 0 -> movement heading is same as agent heading
        self._movement_heading = self.aghead

        # Construct frames of the input space
        # in order to build image vectors out of them
        self.construct_frames()

        # Create the vessel obj and place its center at the agent position
        if not TRAIN_ON_TAURUS:
            self.init_vessel()

        # Get water depth, current velocity
        # and current direction for the agent's position
        # Water depth in [m]
        # Velocity in [m/s]
        # Direction in [rad]
        self.find_river_metrics()

        self.update_cross_track_error() # Cross track error [m]
        self.update_desired_course()# Desired course [rad]
        self.reset_ivs() # Reset the vessel dynamics to their initial state
        self.build_state() # Construct state

        if return_info:
            return self.state, options
        return self.state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:

        # Reset the done flag on every step
        self.done: bool = False
        if self.timestep >= self.max_episode_steps:
            self.done = True

        # Map action to change in rudder
        self.delta = self.map_action(action, self.mode)

        # Unpack starting values for ode solver
        uvr = np.array([self.ivs.u,self.ivs.v,self.ivs.r])

        # Simulate the vessel with the MMG model for one second (default)
        try:
            sol = mmg.step(
                X=uvr,
                vessel=self.vessel,
                psi=self.aghead,
                nps=self.nps,
                delta=self.delta,
                fl_psi=(self.strdir + PI) % (TWOPI),
                water_depth=self.wd,
                fl_vel=self.strvel,
                atol=1e-6,
                rtol=1e-3,
                dT=1
            )

            # Unpack values of solver
            u, v, r = sol

        # Catch any errors in the dynamics
        except (ValueError,mmg.LogicError):
            sol = uvr
            self.done = True
            u, v, r = 0, 0, 0

        # Update Speed
        self.speed = math.sqrt(u**2+v**2)

        # Update state
        self.update_vessel_heading(r)
        self.update_movement_heading(u, v)
        self.update_position(u, v)

        # Check if agent crossed a waypoint
        if self.crossed_waypoint():
            self.update_waypoints()
            self.construct_frames()

        # Get new river metrics
        self.find_river_metrics()

        # Cross track error [m]
        self.update_cross_track_error()

        # Update desired course based on cross-track-error
        self.update_desired_course()

        # Log relevant values
        self.history.log(self, action)

        self.reward = self.calculate_reward()

        self.history.append(self.reward,"reward")

        # Set initial values to the result of the last iteration
        self.ivs.u, self.ivs.v, self.ivs.r = np.hstack(sol)

        # Rebuild state
        self.build_state()

        if self.done:
            self.timestep = 1
        else:
            self.timestep += 1

        #print(round(self.heading_error,2))
        return self.state, self.reward, self.done, {}

    def map_action(self, action: int, mode: str) -> None:
        """Maps an integer action to a change in rudder angle

        Args:
            action (int): Integer value representing an action
        """

        if mode == "step":
            if action ==1:
                return self.delta
            elif action == 0:
                return max(self.delta-self.RUDDER_INCR,-self.MAX_RUDDER)
            elif action == 2:
                return min(self.delta+self.RUDDER_INCR,self.MAX_RUDDER)

        else:
            # Offset actions so that median(n_actions) = noop
            action -= int(self.n_actions/2)

            # Commanded rudder angle
            self._cra = float(action*self.RUDDER_INCR)
            if action <=0:
                return max(self.delta-self.cra,-self.MAX_RUDDER)
            else:
                return min(self.delta+self.cra,self.MAX_RUDDER)

    def calculate_reward(self) -> float:

        draft = self.vessel.d

        if self.wd < 1.2*draft:
            self.done = True
            border_penalty = -20
        else:
            border_penalty = 0

        try:
            r_comf = 1 if self.history.action[-2] == self.history.action[-1] else 0
        except:
            r_comf = 0

        # Hyperparameter controlling the steepness
        # of the exponentiated cross track error
        k_cte = 0.1
        r_cte = math.exp(-k_cte * abs(self.cte))

        # Reward for heading angle
        k_ang = 10
        r_ang = math.exp(-k_ang * abs(self.heading_error))

        #weights = np.array([0.6,0.2,0.2])
        weights = np.array([0.6,0.4,0.0])
        rewards = np.array([r_cte,r_ang,r_comf])

        return weights@rewards + border_penalty

    def construct_path(self) -> Dict[str,List[float]]:
        """Generate the path for the vessel to follow.
        For now this is the deepest point of the fairway
        plus some noise for the entire river.

        To do this, the middle third of the rivers crossection
        is scanned to find the deepest point in that corridor.
        This point then gets mapped to its corresponding
        x and y coordinates, which then gets returned

        Returns:
            Dict: Dict with x and y coordinates of the path
        """

        arraysize = self._wdarray.shape[0]

        # Dictionary to hold the path
        path = {
            "x": np.empty(arraysize),
            "y": np.empty(arraysize),
            "idx": np.arange(arraysize)
        }

        # To use only the middle portion of the river
        offset = N_GRIDPOINTS // 3

        for col in range(arraysize):
            # Get middle thrid per column
            frame = self._wdarray[col][offset:2*offset+1]
            max_ind = np.argmax(frame)  # Find deepest index
            max_ind += offset  # Add back offset

            # Average to river midpoint to keep the path
            # more centered on the river
            max_ind = (max_ind + N_GRIDPOINTS//2)//2

            # Construct noise
            noise = random.randrange(-1,2)

            if col != 0:
                if abs(self.path_index[col -1]-max_ind)>1:
                    if self.path_index[col -1]-max_ind <0:
                        self.path_index[col] = self.path_index[col -1] + 1 + noise
                    else:
                        self.path_index[col] = self.path_index[col -1] - 1 + noise
                else:
                    # Add the index to the path index list
                    self.path_index[col] = int(max_ind) + noise
            else:
                self.path_index[col] = int(max_ind)

            # Get the coordinates of the path by returning the x and y coord
            # for the calculated max index per column
            path["x"][col] = self.rx[col][max_ind]
            path["y"][col] = self.ry[col][max_ind]

        return path

    def build_fairway_border(self) -> None:
        """Build a border to port and starboard of the vessel
        by checking if the water is deep enough to drive.

        This function runs once on every call to reset()
        and scans through the entire water depth array 'self._wdarray'
        to find all points for which the water depth is lower than
        the minimum allowed water under keel.
        """

        draft = self.vessel.d

        free_space: np.ndarray[bool] = self._wdarray < 1.2*draft

        lower_poly = np.full(free_space.shape[0], int(0))
        upper_poly = np.full(free_space.shape[0], N_GRIDPOINTS - 1)

        max_index = N_GRIDPOINTS - 1
        for i in range(free_space.shape[0]):
            mid_index: int = self.path_index[i]
            searching_upper: bool = True
            searching_lower: bool = True
            for lo in range(mid_index):
                if free_space[i][mid_index - lo] and searching_lower:
                    lower_poly[i] = mid_index - lo + 1
                    searching_lower = False
            for high in range(max_index - mid_index):
                if free_space[i][mid_index + high] and searching_upper:
                    upper_poly[i] = mid_index + high
                    searching_upper = False

        self.port_border: dict = {
            "x": np.array([self.rx[i][upper_poly[i]]
                           for i in range(self._wdarray.shape[0])]),
            "y": np.array([self.ry[i][upper_poly[i]]
                           for i in range(self._wdarray.shape[0])])
        }

        self.star_border: dict = {
            "x": np.array([self.rx[i][lower_poly[i]]
                           for i in range(self._wdarray.shape[0])]),
            "y": np.array([self.ry[i][lower_poly[i]]
                           for i in range(self._wdarray.shape[0])])
        }

    def smooth(self, path: Dict[str, np.ndarray],
               every_nth: int = 2, alpha: float = 0.05) -> Dict[str, np.ndarray]:
        """Smooth a given dictionary path

        Args:
            path (dict): dict with x and y coords in it
            every_nth (int): Use only every nth data point of the path
            in the output array

        Returns:
            dict: same dict but with smoothed coords
        """
        x, y, idx = path["x"], path["y"], path["idx"]

        alpha = alpha
        x = self.exponential_smoothing(x, alpha)
        y = self.exponential_smoothing(y, alpha)

        # Only use every nth data point of the path
        if every_nth > 1:
            x = x[::every_nth]
            y = y[::every_nth]
            idx = idx[::every_nth]

        path["x"], path["y"], path["idx"] = x, y, idx

        return path

    def build_state(self) -> None:
        """Build the state space"""

        if self.timestep == 1:
            state = np.hstack(
                [
                    self.ivs.u,
                    self.ivs.v,
                    self.ivs.r,
                    self.ivs.delta,
                    np.zeros(4),
                    math.tanh(self.T*self.cte),
                    np.zeros(1),
                    self.heading_error,
                    np.zeros(1),
                    (self.wd - self.vessel.d)/np.max(self._wdarray),
                    self.rel_current_attack_angle
                ]
            )
        else:
            state = np.hstack(
                [
                    self.ivs.u, #surge
                    self.ivs.v, #sway
                    self.ivs.r, #yaw
                    self.ivs.delta, #rudder angle
                    self.history.u[-2],
                    self.history.v[-2],
                    self.history.u[-2],
                    self.history.delta[-2],
                    math.tanh(self.T*self.cte), #cross-track-error
                    math.tanh(self.T*self.history.cte[-2]),
                    self.heading_error,
                    self.history.heading_error[-2],
                    (self.wd - self.vessel.d)/np.max(self._wdarray),
                    self.rel_current_attack_angle
                ]
            )
        self._state = state

    def reset_ivs(self) -> None:
        """Resets the dynamics of the vessel"""

        self._ivs = InitialValues(
            u     = 2.0,  # Longitudinal vessel speed [m/s]
            v     = 0.0,  # Lateral vessel speed [m/s]
            r     = 0.0,  # Yaw rate acceleration [rad/s]
            delta = 0.0,  # Rudder angle [rad]
            nps   = self.nps  # Propeller revs [s⁻¹]
        )

    @staticmethod
    def rel_ang_diff(a1: float, a2: float) -> float:
        """Relative angle difference for an angle range of [0,2*pi]

        Args:
            a1 (float): Angle in radians or degrees
            a2 (float): Angle in radians or degrees

        Returns:
            float: absolute diff in angles or degs
        """

        if abs(a1-a2) <= PI:
            if a1 <= a2:
                z = abs(a1-a2)
            else:
                z = a2-a1
        else:
            if a1 < a2:
                z = abs(a1-a2) - TWOPI
            else:
                z = TWOPI - abs(a1-a2)
        return float(z)

    def path_angle(self, p1: Point, p2: Point) -> float:
        """Get the angle in radians from the ordinate
        for two points forming a straight line.
        Angle increases clockwise.

        Args:
            p1 (Tuple): Coordinates of first point. Form: (x,y)
            p2 (Tuple): Coordinates of second point. Form: (x,y)

        Returns:
            float: Angle in radians
        """
        return np.arctan2(p2.x-p1.x,p2.y-p1.y) % TWOPI

    @staticmethod
    def dist(p1: Point, p2: Point) -> float:
        """Euler distance

        Args:
            p1 (Point):1st point
            p2 (Point):2nd point

        Returns:
            float: distance between points
        """
        return math.sqrt((p1.y-p2.y)**2 + (p1.x-p2.x)**2)

    def update_cross_track_error(self) -> None:
        """Calculate the cross track error for vector field guidance

        Returns:
            float: Cross track error
        """
        # Calculate distance to path as the height of
        # the triangle formed by three points (agpos,lwp,nwp)
        # two_a = np.linalg.det(
        #     np.array(
        #         [
        #             [self.lwp.x,self.agpos.x,self.nwp.x],
        #             [self.lwp.y,self.agpos.y,self.nwp.y],
        #             [1.,1.,1.]
        #         ],dtype=float
        #     )
        # )
        # cte = two_a/self.dist(self.lwp, self.nwp)
        ph = self.path_angle(self.lwp,self.nwp)
        cte = (self.agpos.x - self.lwp.x)*np.cos(ph)-(self.agpos.y - self.lwp.y)*np.sin(ph)

        self._cte = cte

    def update_desired_course(self) -> float:
        """Calculate the desired course of the vessel

        Returns:
            float: Desired vessel course in radians
        """
        # Distance between any two waypoints
        dbw = self.dist(self.lwp, self.get_red_wp(self.lwp_idx,1))

        path_heading = self.path_angle(self.lwp, self.nwp)
        next_heading = self.path_angle(
            self.nwp, self.get_red_wp(self.lwp_idx,2))

        # Distance to the last waypoint (along-track-error)
        dist_on_wp = (self.agpos.x-self.lwp.x)*math.sin(path_heading)+\
            (self.agpos.y-self.lwp.y)*math.cos(path_heading)

        if path_heading - next_heading <= -PI:
            path_heading += TWOPI
        elif path_heading - next_heading >= PI:
            next_heading += TWOPI

        # Percentage of distance traveled
        # from last to next waypoint
        frac = dist_on_wp/dbw

        # Angle pointing towards path (Vector field)
        tan_cte = -math.atan(self.K * self.cte)

        # Desired heading as a weighted sum
        # of current and next path segment
        head = (1-frac)*path_heading + frac*next_heading
        dc = tan_cte + head

        if dc < 0.0:
            dc += TWOPI

        self._dc = dc % TWOPI

    def distance_to_starboard(self) -> float:
        """
        Return distance from midship to the
        fairway border at starboard
        """

        sbx = self.star_border["x"][self.current_crosssection]
        sby = self.star_border["y"][self.current_crosssection]
        return self.dist(self.agpos,Point(sbx,sby))

    def distance_to_port(self) -> float:
        """
        Return distance from midship to the
        fairway border at port
        """

        px = self.port_border["x"][self.current_crosssection]
        py = self.port_border["y"][self.current_crosssection]
        return self.dist(self.agpos,Point(px,py))

    def crossed_waypoint(self) -> bool:
        """Determine if agent crossed a waypoint by checking
        if the distance to the waypoint after next is closer
        than to the last one.

        If the distance to the two-steps-ahead waypoint is closer,
        we know that the vessel has crossed the next waypoint in
        front of it. Therefore last and next waypoint switch, and the
        waypoint to be checked becomes the next waypoint.
        """
        to_check = self.get_red_wp(self.lwp_idx,plus_n=2)

        dist_to_last = self.dist(self.agpos, self.lwp)
        dist_to_check = self.dist(self.agpos, to_check)

        # Last waypoint is still the closest -> do nothing
        if dist_to_last < dist_to_check:
            return False

        return True

    def update_waypoints(self) -> None:
        """Switch last and next waypoint if agent crossed
        a waypoint"""

        new_nwp = self.get_red_wp(self.lwp_idx,plus_n=2)
        # Waypoint to check is closer than the old one
        # -> switch waypoints
        self.lwp = self.nwp
        self.nwp = new_nwp
        if self.DIR == 1:
            self.lwp_idx += 1
        elif self.DIR == -1:
            self.lwp_idx -= 1
        print(f"Currently at WP {self.lwp_idx}/{len(self.red_path['x'])}")

    def get_lwp_idx(self, index: int) -> int:
        """Get a waypoint based on its index

        Args:
            index (int): Index of the waypoint
            plus_n (int): waypoint n ahead of the current
                          dependent on vessel direction

        Returns:
            Tuple: x and y coordinates of the waypoint
        """
        red_indices = self.red_path["idx"]
        for idx, _ in enumerate(red_indices):
            if red_indices[idx] > index:
                return idx - 1 if self.DIR == 1 else idx
        raise RuntimeError(
            "End of path reached. No further indices found to process."
        )

    def get_red_wp(self,lwp_idx: int, plus_n: int = 0) -> Point:
        """Get the x and y coordinate of the waypoint
        behind the agent if plus_n==0.

        Args:
            index (int): _description_
            plus_n (int, optional): _description_. Defaults to 0.

        Returns:
            Tuple[float, float]: _description_
        """

        if self.DIR == 1:
            x = self.red_path["x"][lwp_idx + plus_n]
            y = self.red_path["y"][lwp_idx + plus_n]
        elif self.DIR == -1:
            x = self.red_path["x"][lwp_idx - plus_n]
            y = self.red_path["y"][lwp_idx - plus_n]
        return Point(x,y)


    def update_position(self, u: float, v: float) -> None:
        """Transform the numerical integration result from
        vessel fixed coordinate system to earth-fixed coordinate system
        and update the agent's x and y positions.

        Args:
            res (Tuple[float,float]): Tuple (u,v) of integration output:
                u: Longitudinal velocity
                v: Lateral velocity
        """

        # Update absolute positions
        vx = math.cos(self.aghead) * u - math.sin(self.aghead) * v
        vy = math.sin(self.aghead) * u + math.cos(self.aghead) * v

        # Since this env is defined as x-North|y-east we need to flip coords
        vx, vy = self.swap_xy(vx, vy)

        nx = float(self._agpos.x + vx)
        ny = float(self._agpos.y + vy)
        self.agpos = Point(nx,ny)

        # Update the exterior points of the vessel for plotting
        if not TRAIN_ON_TAURUS:
            self.set_exterior()

    def set_exterior(self) -> None:
        """Set corner points of the vessel according to
        movement
        """
        # Rectangle of the heading transformed vessel
        self.vessel_rect = Rectangle(
            self.ship_anchor(),
            width=self.vessel.B,
            height=self.vessel.Lpp,
            rotation_point="center",
            angle=360-self.aghead*180/PI,
            color="black"
        )
        # Corner points of the vessel rectangle
        self.vessel_exterior_xy = self.vessel_rect.get_corners()

    def init_vessel(self) -> None:
        """Wrapper to inititalize the vessel"""
        self.set_exterior()

    def update_vessel_heading(self, r: float) -> None:
        """Update the heading by adding the yaw rate to
        the heading.
        Additionally check correct the heading if
        it is larger or smaller than |360°|
        """
        # If agent heading > 360° reset to 0°
        self._aghead = (self._aghead + r) % TWOPI

    def update_movement_heading(self, u: float, v: float) -> None:
        """Set the movement heading, defined as the vessel heading plus
        its drift angle"""
        self._movement_heading = math.atan2(v, u) + self.aghead

    @staticmethod
    def swap_xy(x: float,y: float) -> Tuple[float,float]:
        """Swaps x and y coordinate in order to assign
        longitudinal motion to the ascending y axis

        Args:
            x (float): old x coord
            y (float): old y coord

        Returns:
            Tuple[float, float]: swapped x,y coords
        """
        return y,x

    def find_river_metrics(self) -> Tuple[float,float,float]:
        """Get the mean water depth and stream velocity under the vessel:
        Water depth:
            We first grid-search through the x-y coordinate vector
            to find the column and index of the closest known point
            to the current position of the vessel.

            Then the column number and index are used to find
            the corresponding water depth value.

            This is done for each of the four exterior points
            of the vessel

            This is an exhausive search algorithm for now. To
            map each point of the vessel to its corresponding
            water depth, around 200 distances are needed be calculated.

        Stream velocity:
            Since the stream velocity array is of the same shape as the
            water depth array, therefore resembling the same positions,
            we can just take the column and index value for the water depth
            and plug it into the velocity array.

        Returns:
            float: Mean water depth under vessel
        """
        width = 10

        start_idx = self.path["idx"][self.red_path["idx"][self.lwp_idx]]
        search_range = np.arange(start_idx-width, start_idx+width+1)

        # Shorten the distance function
        d = self.dist

        if TRAIN_ON_TAURUS:
            dist = [1000]  # [dist]
            col_idx = [(0, 0)]  # [(col, idx)]
            ptc = [list(self.agpos)]
        else:
            dist = [1000] * 4  # [dist]
            col_idx = [(0, 0)] * 4  # [(col, idx)]
            ptc = self.vessel_exterior_xy

        for idx, point in enumerate(ptc):
            for col in search_range:
                for colidx in range(N_GRIDPOINTS):
                    x = self.rx[col][colidx]
                    y = self.ry[col][colidx]
                    dtp = d(Point(x, y), Point(*point))
                    if dtp < dist[idx]:
                        dist[idx] = dtp
                        col_idx[idx] = col, colidx

        wd = [self._wdarray[col][idx] for col, idx in col_idx]
        str_vel = [self._strvelarray[col][idx] for col, idx in col_idx]

        str_dirx = [self.str_dirx[col][idx] for col, idx in col_idx]
        str_diry = [self.str_diry[col][idx] for col, idx in col_idx]
        str_dir = np.arctan2(str_diry, str_dirx)

        # Mean index of the current crosssection the vessel passes through
        self.current_crosssection = np.sum(col_idx[0])//len(ptc)

        self._wd = np.mean(wd)
        self._strvel = np.mean(str_vel)
        self._strdir = np.mean(str_dir)


    def exponential_smoothing(self, x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Simple exponential smoothing

        Args:
            x (np.ndarray): array of input values
            alpha (float, optional): Smoothing alpha. Defaults to 0.05.

        Returns:
            np.ndarray: smoothed array
        """
        s = np.zeros_like(x)
        for idx, x_val in enumerate(x):
            if idx == 0:
                s[idx] = x[idx]
            else:
                s[idx] = alpha * x_val + (1-alpha) * s[idx-1]
        return s

    def render(self, mode: str = "human") -> None:
        if mode == "human":

            if not plt.get_fignums():
                self.fig = plt.figure()
                self.fig.patch.set_facecolor("#212529")
                self.ax: plt.Axes = self.fig.add_subplot(1, 1, 1)
                plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05)

            self.ax.clear()
            if WITHFRAMES:
                self.ax.contourf(
                    self.rx_frame, self.ry_frame,
                    self._wdarray_frame, cmap=cm.ocean
                )
                self.ax.quiver(
                    self.rx_frame[::2, ::2], self.ry_frame[::2, ::2],
                    self.str_diry_frame[::2,::2], self.str_dirx_frame[::2, ::2],
                    scale=200,
                    headwidth=2
                )
            else:
                self.ax.contourf(
                    self.rx, self.ry,
                    self._wdarray, cmap=cm.ocean,
                    levels = np.linspace(np.min(self._wdarray),np.max(self._wdarray),30)
                )
                self.ax.quiver(
                    self.rx, self.ry,
                    self.str_diry, self.str_dirx,
                    scale=100, headwidth=2
                )
            self.ax.plot(
                self.red_path["x"], self.red_path["y"],
                color="red",
                marker=None
            )
            self.ax.plot(
                self.port_border["x"],
                self.port_border["y"], color="white"
            )
            self.ax.plot(
                self.star_border["x"],
                self.star_border["y"], color="white"
            )
            self.ax.arrow(
                *self.draw_heading(self.desired_course),
                color="green", width=5,
                label=f"Desired Heading: {round(float(self.desired_course)*180/PI,2)}°"
            )
            self.ax.arrow(
                *self.draw_heading(self.movement_heading),
                color="orange", width=5,
                label=f"Movement Heading: {round(float(self.movement_heading)*180/PI,2)}°"
            )

            self.ax.add_patch(self.vessel_rect)

            handles, _ = self.ax.get_legend_handles_labels()

            speed_count = Patch(
                color="white",
                label=f"Vessel Speed: {round(self.speed,2)} m/s"
            )
            wuk = round(self.wd - self.vessel.d, 2)
            wd_below_keel = Patch(
                color="white",
                label=f"Water under Keel: {wuk} m"
            )
            cta = np.round(((self.strdir + PI)%(TWOPI))*180/PI, 2)
            cta_patch = Patch(
                color="white", label=f"Current Attack Angle: {cta}°")
            delta = round(float(self.delta*180/PI), 2)
            delta_patch = Patch(
                color="white", label=f"Rudder Angle: {delta}°")

            fl_vel = round(float(self.strvel), 2)
            fl_vel_patch = Patch(
                color="white", label=f"Current velocity: {fl_vel} m/s")

            handles.append(speed_count)
            handles.append(wd_below_keel)
            handles.append(cta_patch)
            handles.append(delta_patch)
            handles.append(fl_vel_patch)
            self.ax.legend(handles=handles)
            self.ax.set_facecolor("#363a47")

            #zoom = 800
            zoom = 600
            self.ax.set_xlim(
                self.agpos.x - zoom * 1.6,
                self.agpos.x + zoom * 1.6
            )
            self.ax.set_ylim(
                self.agpos.y - zoom,
                self.agpos.y + zoom
            )
            plt.pause(0.001)

        else:
            raise NotImplementedError(
                "Currently no other mode than 'human' is available.")

    def construct_frames(self) -> None:
        """Construct frames of every metric
        that are used to speed up rendering
        """

        def make(*, obj: Any, la: int, lb: int) -> Any:
            """Frame constructor basis function

            Args:
                obj (Any): metric to frame
                la (int): lookahead distance
                lb (int): lookbehind distance

            Returns:
                Any: Object of same type as `obj`
                     but cut according to la and lb
            """
            if not isinstance(obj, dict):
                wp = self.path["idx"][self.red_path["idx"][self.lwp_idx]]
                if self.DIR == 1:
                    return obj[wp-lb:wp+la]
                else:
                    return obj[wp-la:wp+lb]
            else:
                wp = self.path["idx"][self.red_path["idx"][self.lwp_idx]]
                if self.DIR == 1:
                    return (
                        obj["x"][wp-lb:wp+la],
                        obj["y"][wp-lb:wp+la]
                    )
                else:
                    return (
                        obj["x"][wp-la:wp+lb],
                        obj["y"][wp-la:wp+lb])

        self.path_frame = make(
            obj=self.path,
            la=self.lookahead,
            lb=self.lookbehind
        )
        self._wdarray_frame = make(
            obj=self._wdarray - self.vessel.d,
            la=self.lookahead,
            lb=self.lookbehind
        )

        self.str_vel_frame = make(
            obj=self._strvelarray,
            la=self.lookahead,
            lb=self.lookbehind
        )

        self.rx_frame = make(
            obj=self.rx,
            la=self.lookahead,
            lb=self.lookbehind
        )

        self.ry_frame = make(
            obj=self.ry,
            la=self.lookahead,
            lb=self.lookbehind
        )

        self.port_border_frame = make(
            obj=self.port_border,
            la=self.lookahead,
            lb=self.lookbehind
        )

        self.star_border_frame = make(
            obj=self.star_border,
            la=self.lookahead,
            lb=self.lookbehind
        )

        self.str_dirx_frame = make(
            obj=self.str_dirx,
            la=self.lookahead,
            lb=self.lookbehind
        )

        self.str_diry_frame = make(
            obj=self.str_diry,
            la=self.lookahead,
            lb=self.lookbehind
        )

    def draw_heading(self, angle: float, len: int = 200) -> Tuple[float, float, float, float]:
        """Draws an arrow of length `len` in direction of `angle`.

        Args:
            angle (float): _description_
            len (int, optional): _description_. Defaults to 200.

        Returns:
            Tuple[float, float, float, float]: x and y coordinates of base and tip
            of arrow respectively
        """
        a = self.agpos
        endx = len * math.sin(angle)
        endy = len * math.cos(angle)
        return float(a.x), float(a.y), endx, endy

    def ship_anchor(self) -> Tuple:
        """Build a coordinate tuple resembling the
        anchor point of this rectangle (used to plot the vessel)
        :                + - width - +
        :   y            |           |
        :   |            |           |
        :   |            |           |
        :   |_____x      |           |
        :              height        |
        :                |           |
        :             (anchor)------ +

        Returns:
            Tuple: coordinates of the anchor
        """

        ap = self.agpos
        # Length and Breadth of the simulated vessel
        L, B = self.vessel.Lpp, self.vessel.B
        anchor = ap.x - B/2, ap.y - L/2
        return anchor

    # PID controller functions ------------------------------------------

    def get_pid_command(self) -> float:

        e = -self.heading_error
        #e = -np.tanh(self.cte)
        self.integator_sum += e # Integrate error

        # PID command
        comm =  self.K_p*e + self.K_d*self.r_lt + self.K_i*self.integator_sum
        if abs(comm) > self.RUDDER_INCR:
            comm = self.RUDDER_INCR * np.sign(comm)
        self._cra = comm

        # Calculate commanded rudder angle
        if self._cra < 0.0:
            self._cra = max(self.delta+comm,-self.MAX_RUDDER)
        else:
            self._cra = min(self.delta+comm,self.MAX_RUDDER)
        return self.cra

    def pid_step(self):

        # Calculate commanded rudder angle
        self.delta = self.get_pid_command()

        uvr = np.array([self.ivs.u,self.ivs.v,self.ivs.r])

        # Simulate the vessel with the MMG model for one second (default)
        try:
            sol = mmg.step(
                X=uvr,
                vessel=self.vessel,
                psi=self.aghead,
                nps=self.nps,
                delta=self.delta,
                fl_psi=(self.strdir + PI) % (TWOPI),
                water_depth=self.wd,
                fl_vel=self.strvel,
                atol=1e-6,
                rtol=1e-3,
                dT=1
            )

            # Unpack values of solver
            u, v, r, *_ = sol

        # Catch any errors in the dynamics
        except ValueError:
            sol = self.ivs
            u, v, r = 0, 0, 0

        self.history.log(self, None)

        # Update Speed
        self.speed = math.sqrt(u**2+v**2)

        # Store old heading_error in [deg]
        self.r_lt = -r

        # Update state
        self.update_vessel_heading(r)
        self.update_movement_heading(u, v)
        self.update_position(u, v)

        # Check if agent crossed a waypoint
        if self.crossed_waypoint():
            self.update_waypoints()
            self.construct_frames()

        # Get new river metrics
        self.find_river_metrics()

        # Cross track error [m]
        self.update_cross_track_error()

        # Update desired course based on corss-track-error
        self.update_desired_course()

        self.reward = self.calculate_reward()

        self.history.append(self.reward,"reward")

        # Set initial values to the result of the last iteration
        self.ivs.u, self.ivs.v, self.ivs.r = np.hstack(sol)

        #print(np.round(self.cra*180/PI,2),np.round(self.delta*180/PI,2))
        return

    # Dynamic setter for extracted river data ----------------------------

    def set_arrs(self, coords: np.ndarray, metrics: np.ndarray) -> None:
        """Set the arrays of coordinates and metrics.

        Args:
            coords (ArrayLike): array of coordinates
            metrics (ArrayLike): array of metrics
        """
        # Since our generated river can have variable width
        # I implement a switch here.

        # River X coordinate
        self.rx = np.array([row[0] for row in coords])
        self.rx = self.rx.reshape((-1, N_GRIDPOINTS))

        # River Y coordinate
        self.ry = np.array([row[1] for row in coords])
        self.ry = self.ry.reshape((-1, N_GRIDPOINTS))

        # Extract water depth and reshape
        if TRAIN:
            self._wdarray = np.array([row[3] for row  in metrics])
        else:
            self._wdarray = np.array([math.sqrt(row[3]) for row in metrics])
            self._wdarray = self._wdarray + 4
        self._wdarray = self._wdarray.reshape((-1, N_GRIDPOINTS))

        # Stream direction
        self.str_diry = np.array([row[1] for row in metrics])

        self.str_dirx = np.array([row[2] for row in metrics])

        # Extract stream velocity and reshape
        try:
            self._strvelarray = np.array([row[6]for row  in metrics])
        except:
            self._strvelarray = np.sqrt(self.str_dirx**2+self.str_diry**2)
        self._strvelarray = self._strvelarray.reshape((-1, N_GRIDPOINTS))

        self.str_dirx = self.str_dirx.reshape((-1, N_GRIDPOINTS))
        self.str_diry = self.str_diry.reshape((-1, N_GRIDPOINTS))


# Additional helper functions and classes -------------------------------

def import_river(path: str) -> Tuple[Dict[str, list], Dict[str, list]]:
    """Import river metrics from two separate files

    Args:
        path (str): destination of the files

    Returns:
        Tuple[list, list]: list of coordinates and list of metrics
        (water depth, stream velocity)
    """
    data = {
        "coords": [],
        "metrics": []
    }

    if not path.endswith("/"):
        path += "/"

    for file in ["coords", "metrics"]:
        with open(f"{path}{file}.txt", "r") as f:
            reader = csv.reader(f, delimiter=" ")
            data[file] = list(reader)

    return np.array(data["coords"],dtype=float), np.array(data["metrics"],dtype=float)

def random_river(nsegments: int, var: float, vel: float) -> os.PathLike:
    """Generate a random river with nsegments from the rivergen package.

    Args:
        nsegments (int): number of segments of the river
    """
    return rg.build(nsegments,var,vel)


def delete_river(path: str) -> None:
    """Delete a river file.

    Args:
        path (str): path to the river file
    """
    abspath  = abspath(path)
    shutil.rmtree(abspath) if isdir(abspath) else None

def to_rad(ang: float) -> float:
    """Convert from degrees to radians"""
    return ang/180*PI

def to_deg(ang: float) -> float:
    """Convert from radians to degrees"""
    return ang*180/PI

@dataclass
class Particle:
    pos: np.ndarray
    velocity: np.ndarray
    best_pos: np.ndarray
    score: float

def _eval(pf: PathFollower,p:Particle):
    pf.K_p = p.pos[0]
    pf.K_d = p.pos[1]
    pf.K_i = p.pos[2]
    err = 0
    pf.reset()
    for _ in range(500):
        pf.pid_step()
        err += abs(pf.heading_error)
        err += pf.heading_error if pf.heading_error > 0 else 0
    return float(err)

def _clip_velocity(actual: float, target: float):
    if abs(actual) > target:
        return target * np.sign(actual)
    else: return actual

def PSO(pf: PathFollower) -> List[float]:

    n_particles = 20
    maxvel = [0.5,1,0.05]

    particles = []
    for _ in range(n_particles):
        _pos = np.array([
            np.random.uniform(0.25,3.75), # K_p
            np.random.uniform(10,30), # K_d
            np.random.uniform(0.025,0.075) # K_i
        ])
        _vel = np.array([0.05,1,0.005])
        _best = _pos
        particles.append(Particle(pos=_pos,velocity=_vel,best_pos=_best,score=np.inf))

    ITERS = 1000
    p_best = None
    score_best = np.inf
    for iter in range(ITERS):

        omega = np.random.uniform(0.5,1)
        with Pool() as p:
            res = p.starmap(_eval,[(pf,particle) for particle in particles])

        for idx, particle in enumerate(particles):
            particle: Particle
            if res[idx] < particle.score:
                particle.score = res[idx]
                particle.best_pos = particle.pos

        best_idx = np.argmin(res)
        if res[best_idx] < score_best:
            score_best = res[best_idx]
            p_best = particles[best_idx].pos
        print(f"{p_best} at Iteration {iter}. Score_best: {score_best:.2f}")

        u = np.random.uniform(0,2)

        for pa in particles:
            pa: Particle
            pa.velocity = omega*pa.velocity + u*(pa.best_pos-pa.pos)+u*(p_best-pa.pos)
            pa.velocity = np.array([_clip_velocity(vel,tgt) for vel,tgt in zip(pa.velocity,maxvel)])
            pa.pos += pa.velocity
            pa.pos = np.clip(pa.pos,0,None)

    return p_best

class History:
    """Logging object for storing
    relevant environment information"""

    pos: Sequence[Point]
    path: Dict[str,np.ndarray]

    u: Sequence[Surge]
    v: Sequence[Sway]
    r: Sequence[YawRate]
    delta: Sequence[RudderAngle]
    ivs: Sequence[np.ndarray]
    action: Sequence[int]
    cte: Sequence[float]
    heading_error: Sequence[float]
    cross_curr_angle: Sequence[float]
    timestep: Sequence[int]
    reward: Sequence[float]

    def __repr__(self) -> str:
        print("\n".join(f"{key}: {type(val)}" for key,
              val in vars(self).items()))

    def append(self, val: Any, attr: str, mode: str = "train") -> None:

        assert mode in ["train", "test"], "Unknown mode."

        if not hasattr(self, attr):
            setattr(self, attr, [])

        item = getattr(self, attr)
        if isinstance(item, (list, deque)):
            item.append(val)
        else:
            raise RuntimeError(
                "Can only append to 'list' and 'deque', "
                f"but {type(item)} was found"
                )

    def save_path(self,pf: PathFollower) -> None:
        setattr(self,"path",[(x,y) for x,y in zip(pf.red_path["x"],pf.red_path["y"])])

    def log(self, pf: PathFollower, action: int) -> None:

        self.append(pf.agpos,"pos")
        self.append(pf.ivs.u,"u")
        self.append(pf.ivs.v,"v")
        self.append(pf.ivs.r,"r")
        self.append(float(pf.delta),"delta")
        self.append(pf.cte,"cte")
        self.append(pf.heading_error,"heading_error")
        self.append(pf.timestep,"timestep")
        self.append(action, "action")
        self.append([pf.ivs.u,pf.ivs.v,pf.ivs.r,pf.ivs.delta],"ivs")
        self.append(pf.rel_ang_diff(pf.movement_heading,(pf.strdir)%TWOPI),"cross_curr_angle")
        #self.append(pf.cra,"cra")


if __name__ == "__main__":
    COLOR = 'white'
    matplotlib.rcParams['text.color'] = COLOR
    matplotlib.rcParams['axes.labelcolor'] = COLOR
    matplotlib.rcParams['xtick.color'] = COLOR
    matplotlib.rcParams['ytick.color'] = COLOR

    #pf = PathFollower(direction=1)
    #PSO(pf)


    pf = PathFollower(direction=1)
    s = pf.reset()
    for i in range(500000):
        try:
            print(f"Iteration {i}",end="\r")
            pf.pid_step()
            if pf.lwp_idx == len(pf.red_path['x']):
                break
        except:
            break
        #pf.render()


    # with open("trajectory_plots/star_border", "w") as file:
    #         file.write(json.dumps([list(pf.star_border["x"]),list(pf.star_border["y"])]))

    # with open("trajectory_plots/port_border", "w") as file:
    #         file.write(json.dumps([list(pf.port_border["x"]),list(pf.port_border["y"])]))

    SAVE = True
    if SAVE:
        path = "trajectory_plots/"
        path = "/home/niklaspaulig/Dropbox/TU Dresden/writing/Autonomous Navigation on Rivers using Deep Reinforcement Learning/data/entire_river/5rps/downstream/pid/lr/"
        with open(path + "cte", "w") as file:
            file.write(json.dumps(list(pf.history.cte)))

        with open(path + "heading_error", "w") as file:
            file.write(json.dumps(list(pf.history.heading_error)))

        with open(path + "rudder_movement", "w") as file:
            file.write(json.dumps(list(pf.history.delta)))

        with open(path + "cross_curr", "w") as file:
            file.write(json.dumps(list(pf.history.cross_curr_angle)))

        with open(path + "yaw_accel", "w") as file:
            file.write(json.dumps(list(pf.history.r)))

        with open(path + "reward", "w") as file:
            file.write(json.dumps(list(pf.history.reward)))

        with open(path + "pos", "w") as file:
            file.write(json.dumps([[pos.x,pos.y] for pos in pf.history.pos]))

        with open(path + "path", "w") as file:
            file.write(json.dumps(list(pf.history.path)))
