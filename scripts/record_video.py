"""Record a policy driving, headless, into an mp4 (for the report video).

    python scripts/record_video.py --model runs/ppo_3m/best_model.zip --episodes 4

Renders the environment's rgb_array frames with no display (SDL dummy driver) and
writes them with OpenCV, so it works over SSH / in CI with no window and no system
ffmpeg. Falls back to an animated GIF if the mp4 codec is unavailable.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # render with no window

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navrl2d.env import NavEnv2D  # noqa: E402
from navrl2d.policies import SCRIPTED_POLICIES  # noqa: E402


def load_policy(spec: str):
    if spec in SCRIPTED_POLICIES:
        return SCRIPTED_POLICIES[spec]
    from stable_baselines3 import PPO, SAC

    algo = SAC if "sac" in spec.lower() else PPO
    model = algo.load(spec, device="cpu")
    return lambda obs, rng: model.predict(obs, deterministic=True)[0]


def collect_frames(policy, episodes: int, seed: int, hold: int) -> list[np.ndarray]:
    env = NavEnv2D(render_mode="rgb_array")
    frames: list[np.ndarray] = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        frames.append(env.render())
        term = trunc = False
        while not (term or trunc):
            obs, _, term, trunc, _ = env.step(policy(obs, None))
            frames.append(env.render())
        frames.extend([frames[-1]] * hold)  # freeze on the outcome for a moment
    env.close()
    return frames


def write_mp4(frames: list[np.ndarray], out: Path, fps: int) -> bool:
    try:
        import cv2
    except ImportError:
        return False
    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        return False
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    return out.exists() and out.stat().st_size > 0


def write_gif(frames: list[np.ndarray], out: Path, fps: int) -> None:
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames[::2]]  # halve for size
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps * 2), loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="runs/ppo_3m/best_model.zip",
                        help=".zip path or a scripted policy name")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--hold", type=int, default=12, help="frames to freeze at episode end")
    parser.add_argument("--out", type=Path, default=ROOT / "report" / "rollout.mp4")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    policy = load_policy(args.model)
    frames = collect_frames(policy, args.episodes, args.seed, args.hold)
    print(f"collected {len(frames)} frames ({args.episodes} episodes)")

    if write_mp4(frames, args.out, args.fps):
        print(f"wrote {args.out}")
    else:
        gif = args.out.with_suffix(".gif")
        write_gif(frames, gif, args.fps)
        print(f"mp4 codec unavailable; wrote {gif} instead")


if __name__ == "__main__":
    main()
