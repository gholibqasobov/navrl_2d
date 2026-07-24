"""Pygame visualisation for NavEnv2D.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from .config import EnvConfig

# palette 
C_BG = (247, 247, 250)
C_WALL = (60, 63, 70)
C_GRID = (226, 228, 234)
C_OBSTACLE = (108, 112, 122)
C_OBSTACLE_EDGE = (72, 76, 86)
C_GOAL = (34, 160, 92)
C_ROBOT = (36, 96, 200)
C_HEADING = (250, 250, 252)
C_TRAIL = (150, 180, 230)
C_TEXT = (40, 42, 48)
C_ALERT = (200, 60, 60)


class PygameRenderer:
    def __init__(self, cfg: EnvConfig, mode: str = "human", fps: int = 30) -> None:
        self.cfg = cfg
        self.mode = mode
        self.fps = fps
        self.size = cfg.window_size
        self.hud_height = 84
        self.trail: list[tuple[float, float]] = []
        self._last_step = -1

        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("dejavusansmono,monospace", 15)
        self.font_small = pygame.font.SysFont("dejavusansmono,monospace", 13)

        surface_size = (self.size, self.size + self.hud_height)
        if mode == "human":
            pygame.display.init()
            pygame.display.set_caption("NavRL2D")
            self.screen = pygame.display.set_mode(surface_size)
            self.clock = pygame.time.Clock()
        else:
            self.screen = pygame.Surface(surface_size)
            self.clock = None

    # utils

    def _to_px(self, x: float, y: float) -> tuple[int, int]:
        """World coordinates -> screen pixels (y axis points up in the world)."""
        ppu = self.cfg.pixels_per_unit
        return int(round(x * ppu)), int(round(self.size - y * ppu))

    def _scale(self, length: float) -> int:
        return max(1, int(round(length * self.cfg.pixels_per_unit)))

    # draw

    def draw(self, env) -> np.ndarray | None:
        # A step counter that went backwards means a new episode started.
        if env.step_count <= self._last_step:
            self.trail.clear()
        self._last_step = env.step_count
        self.trail.append((float(env.robot[0]), float(env.robot[1])))

        self.screen.fill(C_BG)
        self._draw_grid()
        self._draw_trail()
        self._draw_obstacles(env)
        self._draw_goal(env)
        self._draw_robot(env)
        pygame.draw.rect(self.screen, C_WALL, (0, 0, self.size, self.size), width=3)
        self._draw_hud(env)

        if self.mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(self.fps)
            return None

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def _draw_grid(self) -> None:
        for i in range(1, int(self.cfg.arena_size)):
            px, _ = self._to_px(i, 0)
            pygame.draw.line(self.screen, C_GRID, (px, 0), (px, self.size))
            _, py = self._to_px(0, i)
            pygame.draw.line(self.screen, C_GRID, (0, py), (self.size, py))

    def _draw_trail(self) -> None:
        if len(self.trail) < 2:
            return
        points = [self._to_px(x, y) for x, y in self.trail]
        pygame.draw.lines(self.screen, C_TRAIL, False, points, 2)

    def _draw_obstacles(self, env) -> None:
        for cx, cy, r in env.obstacles:
            center = self._to_px(cx, cy)
            pygame.draw.circle(self.screen, C_OBSTACLE, center, self._scale(r))
            pygame.draw.circle(self.screen, C_OBSTACLE_EDGE, center, self._scale(r), 2)

    def _draw_goal(self, env) -> None:
        center = self._to_px(env.goal[0], env.goal[1])
        pygame.draw.circle(
            self.screen, C_GOAL, center, self._scale(self.cfg.goal_tolerance), 2
        )
        pygame.draw.circle(self.screen, C_GOAL, center, 5)

    def _draw_robot(self, env) -> None:
        x, y, theta = env.robot
        center = self._to_px(x, y)
        radius = self._scale(self.cfg.robot_radius)
        pygame.draw.circle(self.screen, C_ROBOT, center, radius)
        nose = self._to_px(
            x + 1.6 * self.cfg.robot_radius * math.cos(theta),
            y + 1.6 * self.cfg.robot_radius * math.sin(theta),
        )
        pygame.draw.line(self.screen, C_ROBOT, center, nose, 3)
        pygame.draw.circle(self.screen, C_HEADING, center, max(2, radius // 3))

    def _draw_hud(self, env) -> None:
        top = self.size + 8
        v, omega = env.last_action
        dist = float(np.linalg.norm(env.goal - env.robot[:2]))

        lines = [
            f"step {env.step_count:3d}/{self.cfg.max_steps}"
            f"   dist {dist:5.2f}"
            f"   v {v:+.2f}  w {omega:+.2f}",
            f"reward {env.last_reward:+7.3f}   return {env.episode_return:+8.2f}",
        ]
        for i, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, C_TEXT), (10, top + 20 * i))

        status, color = "", C_TEXT
        if dist < self.cfg.goal_tolerance:
            status, color = "GOAL REACHED", C_GOAL
        elif env._check_collision(env.robot[:2]):
            status, color = "COLLISION", C_ALERT
        elif env.step_count >= self.cfg.max_steps:
            status, color = "TIMEOUT", C_ALERT
        if status:
            self.screen.blit(
                self.font.render(status, True, color), (10, top + 44)
            )

    def close(self) -> None:
        if self.mode == "human":
            pygame.display.quit()
        pygame.quit()
