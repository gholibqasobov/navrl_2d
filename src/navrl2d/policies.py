"""Scripted reference policies.

These are the yardsticks the learned agents are measured against, so they live in
the package rather than in a script: `run_baseline.py` and `evaluate.py` both use
the exact same controllers.

Every policy has the signature ``policy(obs, rng) -> action`` and returns an
action in the environment's symmetric ``[-1, 1]^2`` space.
"""

from __future__ import annotations

import math

import numpy as np

from .config import DEFAULT_CONFIG

# Observations are normalised per feature; controllers that reason about real
# distances (units, not fractions) have to undo that.
_SCALE = DEFAULT_CONFIG.obs_dist_scale
_R_MID = 0.5 * (DEFAULT_CONFIG.obs_radius_max + DEFAULT_CONFIG.obs_radius_min)
_R_HALF = 0.5 * (DEFAULT_CONFIG.obs_radius_max - DEFAULT_CONFIG.obs_radius_min)


def random_policy(obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform noise. Lower bound on performance."""
    return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


def greedy_policy(obs: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Full speed, steering straight at the goal. Completely obstacle-blind.

    obs[1], obs[2] are cos/sin of the goal bearing, so atan2 recovers the angle
    without any wrap-around special cases. This is the score a learned policy
    must beat: the gap to 100% is exactly the obstacle-avoidance part of the task.
    """
    bearing = math.atan2(obs[2], obs[1])
    return np.array([1.0, float(np.clip(2.0 * bearing, -1.0, 1.0))], dtype=np.float32)


def still_policy(obs: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Do nothing. Shows what pure timeout looks like in the reward table."""
    return np.array([-1.0, 0.0], dtype=np.float32)


def potential_field_policy(
    obs: np.ndarray,
    rng: np.random.Generator | None = None,
    k_rep: float = 2.5,
    influence: float = 1.5,
) -> np.ndarray:
    """Attractive pull toward the goal plus a repulsive push off nearby obstacles.

    Purely reactive, ~10 lines, and it reads *only* the 12D observation the agent
    gets. That makes it the honest ceiling for this task: any gap between a
    learned policy and this one is a failure of learning, not of the observation
    space or of the environment.
    """
    bearing = math.atan2(obs[2], obs[1])
    turn = 2.0 * bearing
    speed = 1.0

    for slot in range(3):
        dx, dy, r_norm = obs[3 + 3 * slot : 6 + 3 * slot]
        dx, dy = dx * _SCALE, dy * _SCALE
        radius = r_norm * _R_HALF + _R_MID
        surface_dist = math.hypot(dx, dy) - radius
        # Only obstacles ahead and inside the influence radius matter.
        if surface_dist < influence and dx > -0.2:
            strength = k_rep * (1.0 / max(surface_dist, 0.05) - 1.0 / influence)
            turn -= math.copysign(strength, dy)   # steer away from the side it is on
            speed = min(speed, 0.3 + 0.7 * surface_dist / influence)  # slow down when close

    return np.array(
        [2.0 * speed - 1.0, float(np.clip(turn, -1.0, 1.0))], dtype=np.float32
    )


SCRIPTED_POLICIES = {
    "random": random_policy,
    "greedy": greedy_policy,
    "still": still_policy,
    "potential_field": potential_field_policy,
}
