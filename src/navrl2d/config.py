"""All tunable constants for the 2D navigation task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    # world 
    arena_size: float = 10.0          # side of the square arena (units)
    dt: float = 0.1                   # integration timestep (seconds)
    max_steps: int = 500              # episode truncation horizon

    # robot 
    robot_radius: float = 0.2         # collision radius of the robot
    v_max: float = 1.0                # max forward speed (units/s)
    omega_max: float = 2.0            # max yaw rate (rad/s)

    # obstacles 
    n_obstacles: int = 3
    obs_radius_min: float = 0.5
    obs_radius_max: float = 1.0
    obs_wall_margin: float = 0.2      # keep obstacle surfaces off the walls
    obs_min_gap: float = 0.6          # free gap between two obstacle surfaces
                                      # (> 2 * robot_radius, so a path exists)

    # episode sampling 
    goal_tolerance: float = 0.3       # distance at which the goal counts as reached
    spawn_clearance: float = 0.3      # extra free space around start/goal
    min_start_goal_dist: float = 5.0  # reject trivially short episodes
    max_sample_attempts: int = 1000   # rejection-sampling budget per reset

    # reward
    # Progress is zero-sum by construction: moving away costs exactly what
    # moving back earns, so oscillating nets zero and only pays the step cost.
    # |r_goal| == |r_collision| balances risk-taking against freezing, and
    # r_step is ~100x smaller than the terminal rewards -> efficiency without
    # recklessness. Undiscounted episode totals land at roughly
    #   success ~ +16,  collision ~ -7.5,  timeout ~ -5.
    w_progress: float = 1.0           # weight on distance decrease per step
    r_step: float = -0.01             # small per-step cost -> prefer short paths
    r_goal: float = 10.0              # bonus for reaching the goal
    r_collision: float = -10.0        # penalty for hitting an obstacle or wall
    r_timeout: float = -5.0           # penalty for running out of steps

    # observation scaling
    # Lengths are divided by this instead of by the arena diagonal (14.14).
    # Dividing everything by the diagonal looks tidy and is geometrically
    # consistent, but it squashes the features that matter most: a threatening
    # obstacle 1 unit away arrives as 0.07 while the goal bearing arrives as 1.0,
    # so the obstacle channel is ~20x quieter than the goal channel. Measured
    # cost of that choice: a hard ceiling near 65% success, for RL *and* for
    # supervised imitation of a 96% controller. 5.0 is roughly the range over
    # which an obstacle is worth reacting to, and it puts every feature's
    # standard deviation within ~2x of the others.
    obs_dist_scale: float = 5.0

    # rendering
    render_fps: int = 30
    window_size: int = 640            # window is square, in pixels

    @property
    def diagonal(self) -> float:
        """Longest possible distance in the arena -- used to normalise lengths."""
        return math.sqrt(2.0) * self.arena_size

    @property
    def pixels_per_unit(self) -> float:
        return self.window_size / self.arena_size


DEFAULT_CONFIG = EnvConfig()
