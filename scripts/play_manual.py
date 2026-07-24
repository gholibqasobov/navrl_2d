"""Drive the robot with the keyboard.

    python scripts/play_manual.py

Controls
    up / W       forward (hold)          left  / A   turn left  (CCW)
    down / S     brake                   right / D   turn right (CW)
    R            new episode             ESC / Q     quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from navrl2d import NavEnv2D  # noqa: E402


def read_action(keys, pygame) -> np.ndarray:
    """Keyboard state -> action in the agent's own [-1, 1] space."""
    forward = keys[pygame.K_UP] or keys[pygame.K_w]
    brake = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]

    # a[0] = -1 maps to v = 0, a[0] = +1 maps to v = v_max.
    throttle = 1.0 if (forward and not brake) else -1.0
    turn = (1.0 if left else 0.0) - (1.0 if right else 0.0)
    return np.array([throttle, turn], dtype=np.float32)


def describe(info: dict) -> str:
    if info["is_success"]:
        return "GOAL"
    if info["collision"]:
        return "COLLISION"
    return "TIMEOUT"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20, help="10 = real time (dt=0.1s)")
    args = parser.parse_args()

    env = NavEnv2D(render_mode="human", render_fps=args.fps)
    episode = 0
    obs, info = env.reset(seed=args.seed)  # creates the window

    import pygame  # import after the renderer initialised pygame

    running, episode_over = True, False
    print(__doc__)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    episode += 1
                    obs, info = env.reset(seed=args.seed + episode)
                    episode_over = False

        if not running:
            break

        if episode_over:
            env._render_frame()
            continue

        action = read_action(pygame.key.get_pressed(), pygame)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            episode_over = True
            print(
                f"episode {episode}: {describe(info):<9} "
                f"steps {info['steps']:3d}  return {info['episode_return']:+8.2f}"
                "   -- press R for a new episode"
            )

    env.close()


if __name__ == "__main__":
    main()
