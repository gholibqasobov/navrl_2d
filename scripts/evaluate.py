"""Score a saved policy, or watch it drive.

    # numbers over 200 fixed evaluation seeds
    python scripts/evaluate.py --model runs/ppo/best_model.zip

    python scripts/evaluate.py --model runs/ppo/best_model.zip --render --episodes 5

    # the same table for a scripted baseline
    python scripts/evaluate.py --model greedy

    # several policies on identical episodes, side by side
    python scripts/evaluate.py --model runs/ppo/best_model.zip runs/sac/best_model.zip greedy

--model accepts a path to a .zip or the name of a scripted policy
(random / greedy / still). The algorithm is inferred from the path; use --algo
to override.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stable_baselines3 import PPO, SAC  # noqa: E402

from navrl2d.env import NavEnv2D  # noqa: E402
from navrl2d.evaluation import EpisodeResult, evaluate, print_table, summarize  # noqa: E402
from navrl2d.policies import SCRIPTED_POLICIES  # noqa: E402

ALGOS = {"ppo": PPO, "sac": SAC}
EVAL_SEED = 100_000  # same block train.py uses, so numbers are comparable


def infer_algo(path: Path) -> str:
    text = str(path).lower()
    for name in ALGOS:
        if name in text:
            return name
    raise SystemExit(
        f"cannot tell which algorithm produced {path}; pass --algo ppo|sac"
    )


def load_policy(spec: str, algo: str | None, deterministic: bool):
    """Return (label, policy_fn) for either a scripted name or a saved model."""
    if spec in SCRIPTED_POLICIES:
        return spec, SCRIPTED_POLICIES[spec]

    path = Path(spec)
    if not path.exists():
        raise SystemExit(f"no such model: {path} (or unknown scripted policy)")

    algo = algo or infer_algo(path)
    model = ALGOS[algo].load(path, device="cpu")

    def policy(obs, rng):
        return model.predict(obs, deterministic=deterministic)[0]

    label = f"{algo} {path.parent.name}/{path.stem}"
    return label, policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", nargs="+", required=True,
                        help="model .zip path(s) and/or scripted policy name(s)")
    parser.add_argument("--algo", choices=sorted(ALGOS), default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=EVAL_SEED,
                        help="first seed of the fixed evaluation block")
    parser.add_argument("--render", action="store_true", help="open a pygame window")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stochastic", action="store_true",
                        help="sample from the policy instead of taking its mean action")
    args = parser.parse_args()

    if args.render and args.episodes > 20:
        print(f"note: rendering {args.episodes} episodes will take a while "
              f"(try --episodes 5)")

    env = NavEnv2D(render_mode="human" if args.render else None, render_fps=args.fps)
    rows = []

    for spec in args.model:
        label, policy = load_policy(spec, args.algo, deterministic=not args.stochastic)

        def report(index: int, result: EpisodeResult) -> None:
            # Path efficiency only means anything if the goal was actually
            # reached: a crash 3 steps in has a short path and a flattering ratio.
            efficiency = (
                f"  efficiency {result.efficiency:.2f}" if result.outcome == "goal" else ""
            )
            print(f"  ep {index:3d}  {result.outcome:<9} "
                  f"steps {result.steps:3d}  return {result.ret:+7.2f}{efficiency}")

        print(f"\nevaluating {label} on seeds {args.seed}..{args.seed + args.episodes - 1}")
        results = evaluate(
            env,
            policy,
            episodes=args.episodes,
            seed=args.seed,
            on_episode=report if args.render else None,
        )
        rows.append((label, summarize(results)))

    env.close()
    mode = "stochastic" if args.stochastic else "deterministic"
    print_table(rows, title=f"{args.episodes} episodes, identical seeds, {mode} actions")


if __name__ == "__main__":
    main()
