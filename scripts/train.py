"""Train a navigation policy with PPO or SAC (Stable-Baselines3)

    python scripts/train.py --algo ppo                     # 1M steps, 8 envs
    python scripts/train.py --algo sac                     # 300k steps, 1 env
    python scripts/train.py --algo ppo --timesteps 20000 --run-name smoke

runs/<name>/:
    best_model.zip    best mean eval reward seen during training
    final_model.zip   the policy at the last timestep
    worker_*.monitor.csv   one row per training episode (return, length, outcome)
    meta.json         hyperparameters + EnvConfig + timings, for the report
    tb/               TensorBoard logs:  tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.callbacks import EvalCallback  # noqa: E402

from navrl2d import DEFAULT_CONFIG  # noqa: E402
from navrl2d.evaluation import evaluate, print_table, summarize  # noqa: E402
from navrl2d.env import NavEnv2D  # noqa: E402
from navrl2d.policies import greedy_policy  # noqa: E402
from navrl2d.sb3_utils import (  # noqa: E402
    OffPolicyStatsCallback,
    RolloutStatsCallback,
    make_vec_env,
)

# Hyperparameters
DEFAULTS = {
    "ppo": dict(
        total_timesteps=1_000_000,
        n_envs=8,
        # 8 envs x 512 steps = 4096 transitions per policy update.
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        net_arch=[64, 64],
    ),
    "sac": dict(
        total_timesteps=300_000,
        n_envs=1,
        buffer_size=300_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_rate=3e-4,
        ent_coef="auto",
        net_arch=[256, 256],
    ),
}

EVAL_SEED = 100_000  # eval episodes come from a seed block


def build_model(algo: str, venv, hp: dict, seed: int, tb_dir: Path):
    policy_kwargs = dict(net_arch=hp["net_arch"])
    common = dict(
        policy="MlpPolicy",
        env=venv,
        seed=seed,
        verbose=1,
        device="cpu",
        tensorboard_log=str(tb_dir),
        policy_kwargs=policy_kwargs,
        learning_rate=hp["learning_rate"],
        gamma=hp["gamma"],
        batch_size=hp["batch_size"],
    )
    if algo == "ppo":
        return PPO(
            **common,
            n_steps=hp["n_steps"],
            n_epochs=hp["n_epochs"],
            gae_lambda=hp["gae_lambda"],
            clip_range=hp["clip_range"],
            ent_coef=hp["ent_coef"],
            vf_coef=hp["vf_coef"],
            max_grad_norm=hp["max_grad_norm"],
        )
    return SAC(
        **common,
        buffer_size=hp["buffer_size"],
        learning_starts=hp["learning_starts"],
        tau=hp["tau"],
        train_freq=hp["train_freq"],
        gradient_steps=hp["gradient_steps"],
        ent_coef=hp["ent_coef"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--algo", choices=["ppo", "sac"], required=True)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--eval-freq", type=int, default=25_000,
                        help="environment steps between evaluations")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--final-eval-episodes", type=int, default=200)
    args = parser.parse_args()


    torch.set_num_threads(1)

    hp = dict(DEFAULTS[args.algo])
    if args.timesteps is not None:
        hp["total_timesteps"] = args.timesteps
    if args.n_envs is not None:
        hp["n_envs"] = args.n_envs
    if args.lr is not None:
        hp["learning_rate"] = args.lr
    if args.gamma is not None:
        hp["gamma"] = args.gamma

    train_config = DEFAULT_CONFIG

    name = args.run_name or f"{args.algo}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(__file__).resolve().parents[1] / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"run directory: {run_dir}")
    print(f"hyperparameters: {json.dumps(hp, default=str)}")

    train_env = make_vec_env(
        hp["n_envs"], seed=args.seed, config=train_config,
        monitor_path=str(run_dir / "worker")
    )
    # Separate env, separate seed block: never evaluate on training episodes.
    eval_env = make_vec_env(1, seed=EVAL_SEED)

    model = build_model(args.algo, train_env, hp, args.seed, run_dir / "tb")

    stats_callback = (
        RolloutStatsCallback() if args.algo == "ppo" else OffPolicyStatsCallback(2_000)
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=max(args.eval_freq // hp["n_envs"], 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    start = time.time()
    model.learn(
        total_timesteps=hp["total_timesteps"],
        callback=[stats_callback, eval_callback],
        tb_log_name="tb",
        progress_bar=False,
    )
    duration = time.time() - start

    model.save(run_dir / "final_model")
    train_env.close()
    eval_env.close()

    print(f"\ntrained {hp['total_timesteps']} steps in {duration / 60:.1f} min "
          f"({hp['total_timesteps'] / duration:.0f} steps/s)")

    best_path = run_dir / "best_model.zip"
    scored = "best" if best_path.exists() else "final"
    scored_model = model.__class__.load(run_dir / f"{scored}_model", device="cpu")
    env = NavEnv2D()

    def model_policy(obs, rng):
        return scored_model.predict(obs, deterministic=True)[0]

    learned = summarize(evaluate(env, model_policy, args.final_eval_episodes, seed=EVAL_SEED))
    greedy = summarize(evaluate(env, greedy_policy, args.final_eval_episodes, seed=EVAL_SEED))
    env.close()

    print_table(
        [(f"{args.algo} ({scored})", learned), ("greedy baseline", greedy)],
        title=f"final evaluation, {args.final_eval_episodes} fixed seeds",
    )

    meta = {
        "algo": args.algo,
        "run_name": name,
        "seed": args.seed,
        "hyperparameters": hp,
        "env_config": asdict(train_config),
        "wall_clock_minutes": round(duration / 60, 2),
        "steps_per_second": round(hp["total_timesteps"] / duration, 1),
        "final_eval": {"learned": learned, "greedy_baseline": greedy},
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved: {run_dir}/final_model.zip, meta.json"
          + (f", best_model.zip" if best_path.exists() else ""))
    print(f"watch it drive:\n  python scripts/evaluate.py "
          f"--model {run_dir}/{scored}_model.zip --render")


if __name__ == "__main__":
    main()
