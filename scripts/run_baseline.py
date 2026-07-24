"""Run scripted policies in the environment and report outcome statistics.

    python scripts/run_baseline.py                       # all baselines
    python scripts/run_baseline.py --policy greedy --episodes 300
    python scripts/run_baseline.py --policy greedy --episodes 3 --render

These are the references that calibrate the task before any learning happens:

random  uniform actions. Should almost never reach the goal -- if it does, the
        task is trivial and `min_start_goal_dist` should go up.
greedy  full speed, always steering straight at the goal, ignoring obstacles.
        This is the score the RL agent has to beat: the gap to 100% is exactly
        the part of the task that *is* obstacle avoidance.
still   never moves; shows what a pure timeout episode is worth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from navrl2d.env import NavEnv2D  # noqa: E402
from navrl2d.evaluation import evaluate, print_table, summarize  # noqa: E402
from navrl2d.policies import SCRIPTED_POLICIES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy", choices=sorted(SCRIPTED_POLICIES), default=None,
                        help="default: run all of them")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="open a pygame window")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    names = [args.policy] if args.policy else sorted(SCRIPTED_POLICIES)
    env = NavEnv2D(render_mode="human" if args.render else None, render_fps=args.fps)

    rows = []
    for name in names:
        results = evaluate(
            env, SCRIPTED_POLICIES[name], episodes=args.episodes, seed=args.seed
        )
        rows.append((name, summarize(results)))
    env.close()

    print_table(rows, title=f"scripted baselines, {args.episodes} episodes each")


if __name__ == "__main__":
    main()
