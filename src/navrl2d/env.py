"""2D navigation for a unicycle robot with static circular obstacles.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import DEFAULT_CONFIG, EnvConfig


class NavEnv2D(gym.Env):
    """Unicycle navigation in a square arena with 3 static circular obstacles.

    Action space (Box, 2D, symmetric ``[-1, 1]``):
        a[0] -> forward speed, remapped to ``[0, v_max]``  (no reversing)
        a[1] -> yaw rate,      remapped to ``[-omega_max, omega_max]``

    The symmetric range is deliberate: SAC's squashed-Gaussian policy is built
    around ``[-1, 1]`` and warns on asymmetric spaces, so the ``[0, 1]`` speed
    range from the spec is produced by an internal affine map instead.

    Observation space (Box, 12D, ego-centric):
        0     distance to goal
        1, 2  cos / sin of the goal bearing in the robot frame
        3-5   nearest obstacle:       dx, dy (robot frame), radius
        6-8   second-nearest obstacle: dx, dy, radius
        9-11  farthest obstacle:       dx, dy, radius

    Obstacles are sorted by surface distance, so a given slot always carries the
    same *meaning* rather than an arbitrary spawn label.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
        normalize_obs: bool = True,
        render_fps: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config if config is not None else DEFAULT_CONFIG
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.normalize_obs = normalize_obs
        self.render_fps = render_fps if render_fps is not None else self.cfg.render_fps

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Radii are rescaled onto [-1, 1] using their own sampling range.
        self._r_mid = 0.5 * (self.cfg.obs_radius_max + self.cfg.obs_radius_min)
        self._r_half = 0.5 * (self.cfg.obs_radius_max - self.cfg.obs_radius_min)

        # Normalised lengths are bounded by (arena diagonal / obs_dist_scale);
        # unnormalised ones by the diagonal itself.
        obs_limit = (
            max(self.cfg.diagonal / self.cfg.obs_dist_scale, 1.0)
            if normalize_obs
            else self.cfg.diagonal
        )
        self.observation_space = spaces.Box(
            low=-obs_limit, high=obs_limit, shape=(12,), dtype=np.float32
        )

        # episode state (filled in by reset) 
        self.robot = np.zeros(3, dtype=np.float64)          # x, y, theta
        self.goal = np.zeros(2, dtype=np.float64)
        self.obstacles = np.zeros((self.cfg.n_obstacles, 3))  # cx, cy, r
        self.step_count = 0
        self.prev_dist = 0.0
        self.episode_return = 0.0
        self.last_reward = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)     # (v, omega), world units

        self._renderer = None

    # reset

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)  # seeds self.np_random

        self.obstacles = self._sample_obstacles()
        start, goal = self._sample_start_and_goal(self.obstacles)
        theta = self.np_random.uniform(-math.pi, math.pi)

        self.robot = np.array([start[0], start[1], theta], dtype=np.float64)
        self.goal = goal
        self.step_count = 0
        self.prev_dist = float(np.linalg.norm(self.goal - self.robot[:2]))
        self.episode_return = 0.0
        self.last_reward = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    # step

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # Affine remap of the symmetric action into physical commands.
        v = (action[0] + 1.0) / 2.0 * self.cfg.v_max      # [0, v_max]
        omega = action[1] * self.cfg.omega_max            # [-w_max, w_max]
        self.last_action = np.array([v, omega], dtype=np.float32)

        
        # heading is updated first, then the body moves along the heading.
        x, y, theta = self.robot
        theta = self._wrap_angle(theta + omega * self.cfg.dt)
        x += v * math.cos(theta) * self.cfg.dt
        y += v * math.sin(theta) * self.cfg.dt
        self.robot = np.array([x, y, theta], dtype=np.float64)

        self.step_count += 1

        dist = float(np.linalg.norm(self.goal - self.robot[:2]))
        collided = self._check_collision(self.robot[:2])
        reached = dist < self.cfg.goal_tolerance

        terminated = bool(collided or reached)
        # Running out of time is NOT a terminal state of the MDP: the task could
        # still be completed. Keeping it as `truncated` lets PPO/SAC bootstrap
        # the value of the final state instead of treating it as a dead end.
        truncated = bool(not terminated and self.step_count >= self.cfg.max_steps)

        #  reward
        # Dense shaping term for each step; it telescopes over an episode, so the
        # total progress reward of a successful run is just (d_start - d_end).
        reward = self.cfg.w_progress * (self.prev_dist - dist) + self.cfg.r_step
        if collided:
            reward += self.cfg.r_collision
        elif reached:
            reward += self.cfg.r_goal
        elif truncated:
            reward += self.cfg.r_timeout

        self.prev_dist = dist
        self.episode_return += reward
        self.last_reward = reward

        if self.render_mode == "human":
            self._render_frame()

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            self._get_info(collided=collided, reached=reached, timed_out=truncated),
        )

    #  observation

    def _to_robot_frame(self, point: np.ndarray) -> tuple[float, float]:
        """World point -> robot frame (x forward, y left). Rotation by -theta."""
        theta = self.robot[2]
        rel_x = point[0] - self.robot[0]
        rel_y = point[1] - self.robot[1]
        c, s = math.cos(theta), math.sin(theta)
        dx = rel_x * c + rel_y * s      # forward / backward
        dy = -rel_x * s + rel_y * c     # left / right
        return dx, dy

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(12, dtype=np.float64)

        # goal: distance + bearing as cos/sin 
        gx, gy = self._to_robot_frame(self.goal)
        dist = math.hypot(gx, gy)
        obs[0] = dist
        if dist > 1e-9:
            # cos/sin of the bearing angle; avoids the +-pi discontinuity that a
            # raw angle would have.
            obs[1] = gx / dist
            obs[2] = gy / dist
        else:
            obs[1], obs[2] = 1.0, 0.0

        # obstacles, nearest surface first
        centers = self.obstacles[:, :2]
        radii = self.obstacles[:, 2]
        surface_dist = np.linalg.norm(centers - self.robot[:2], axis=1) - radii
        order = np.argsort(surface_dist)

        for slot, idx in enumerate(order):
            dx, dy = self._to_robot_frame(centers[idx])
            base = 3 + 3 * slot
            obs[base + 0] = dx
            obs[base + 1] = dy
            obs[base + 2] = radii[idx]

        if self.normalize_obs:
            # Per-feature scaling: every channel should reach the network with a
            # comparable magnitude. Distances share one scale (so they stay
            # mutually comparable), while radii are mapped onto their own
            # sampling range -- dividing a 0.5..1.0 radius by the arena diagonal
            # leaves a feature with std 0.01, i.e. numerically invisible.
            scale = self.cfg.obs_dist_scale
            obs[0] /= scale
            for slot in range(self.cfg.n_obstacles):
                base = 3 + 3 * slot
                obs[base] /= scale
                obs[base + 1] /= scale
                obs[base + 2] = (obs[base + 2] - self._r_mid) / self._r_half

        return np.clip(
            obs.astype(np.float32),
            self.observation_space.low,
            self.observation_space.high,
        )

    def _get_info(
        self, collided: bool = False, reached: bool = False, timed_out: bool = False
    ) -> dict[str, Any]:
        # The three flags are mutually exclusive and are what Monitor/callbacks
        # aggregate into success / collision / timeout rates during training.
        return {
            "is_success": bool(reached),
            "collision": bool(collided),
            "timeout": bool(timed_out),
            "distance": float(np.linalg.norm(self.goal - self.robot[:2])),
            "steps": self.step_count,
            "episode_return": float(self.episode_return),
        }

    # collisions

    def _check_collision(self, position: np.ndarray) -> bool:
        r = self.cfg.robot_radius
        # Walls: the robot's disc must stay fully inside the arena.
        if (
            position[0] < r
            or position[0] > self.cfg.arena_size - r
            or position[1] < r
            or position[1] > self.cfg.arena_size - r
        ):
            return True
        # Obstacles: disc-disc overlap.
        d = np.linalg.norm(self.obstacles[:, :2] - position, axis=1)
        return bool(np.any(d < self.obstacles[:, 2] + r))

    def _is_free(self, position: np.ndarray, clearance: float) -> bool:
        r = self.cfg.robot_radius + clearance
        if (
            position[0] < r
            or position[0] > self.cfg.arena_size - r
            or position[1] < r
            or position[1] > self.cfg.arena_size - r
        ):
            return False
        d = np.linalg.norm(self.obstacles[:, :2] - position, axis=1)
        return bool(np.all(d > self.obstacles[:, 2] + r))

    # sampling
    def _sample_obstacles(self) -> np.ndarray:
        cfg = self.cfg
        obstacles: list[np.ndarray] = []
        attempts = 0
        while len(obstacles) < cfg.n_obstacles:
            attempts += 1
            if attempts > cfg.max_sample_attempts:
                raise RuntimeError(
                    "Could not place obstacles; loosen obs_min_gap / radii in EnvConfig."
                )
            radius = self.np_random.uniform(cfg.obs_radius_min, cfg.obs_radius_max)
            lo = radius + cfg.obs_wall_margin
            hi = cfg.arena_size - radius - cfg.obs_wall_margin
            center = self.np_random.uniform(lo, hi, size=2)

            # Keep a corridor between obstacles wide enough for the robot.
            ok = all(
                np.linalg.norm(center - o[:2]) > radius + o[2] + cfg.obs_min_gap
                for o in obstacles
            )
            if ok:
                obstacles.append(np.array([center[0], center[1], radius]))
        return np.stack(obstacles)

    def _sample_free_point(self) -> np.ndarray:
        cfg = self.cfg
        for _ in range(cfg.max_sample_attempts):
            p = self.np_random.uniform(0.0, cfg.arena_size, size=2)
            if self._is_free(p, cfg.spawn_clearance):
                return p
        raise RuntimeError("Could not find a free point; arena is too crowded.")

    def _sample_start_and_goal(self, obstacles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.obstacles = obstacles  # _is_free reads from self
        for _ in range(self.cfg.max_sample_attempts):
            start = self._sample_free_point()
            goal = self._sample_free_point()
            if np.linalg.norm(goal - start) >= self.cfg.min_start_goal_dist:
                return start, goal
        raise RuntimeError(
            "Could not sample a start/goal pair; lower min_start_goal_dist."
        )

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap to (-pi, pi]."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    # -------------------------------------------------------------- rendering

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(self) -> np.ndarray | None:
        if self._renderer is None:
            from .render import PygameRenderer  # imported lazily: training never needs pygame

            self._renderer = PygameRenderer(
                self.cfg, mode=self.render_mode, fps=self.render_fps
            )
        return self._renderer.draw(self)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
