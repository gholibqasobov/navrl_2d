"""Shared evaluation logic: roll a policy out and summarise what happened.

Used by `scripts/evaluate.py`, by `scripts/run_baseline.py` and at the end of a
training run, so every number in the report comes from the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

import numpy as np

from .env import NavEnv2D

Policy = Callable[[np.ndarray, np.random.Generator], np.ndarray]


class EpisodeHook(Protocol):
    def __call__(self, index: int, result: "EpisodeResult") -> None: ...


@dataclass
class EpisodeResult:
    outcome: str          # "goal" | "collision" | "timeout"
    steps: int
    ret: float
    path_length: float    # distance actually driven
    straight_line: float  # start -> goal distance at reset

    @property
    def efficiency(self) -> float:
        """1.0 = drove the straight line; obstacles necessarily push this down."""
        return self.straight_line / self.path_length if self.path_length > 0 else 0.0


def run_episode(
    env: NavEnv2D, policy: Policy, seed: int | None, rng: np.random.Generator
) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    core = env.unwrapped
    start = core.robot[:2].copy()
    straight_line = float(np.linalg.norm(core.goal - start))

    path_length = 0.0
    previous = start
    terminated = truncated = False

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(policy(obs, rng))
        position = core.robot[:2]
        path_length += float(np.linalg.norm(position - previous))
        previous = position.copy()

    outcome = (
        "goal" if info["is_success"]
        else "collision" if info["collision"]
        else "timeout"
    )
    return EpisodeResult(
        outcome=outcome,
        steps=info["steps"],
        ret=info["episode_return"],
        path_length=path_length,
        straight_line=straight_line,
    )


def evaluate(
    env: NavEnv2D,
    policy: Policy,
    episodes: int,
    seed: int = 0,
    rng_seed: int = 0,
    on_episode: EpisodeHook | None = None,
) -> list[EpisodeResult]:
    """Run `episodes` episodes on the *fixed* seed block seed .. seed+episodes-1.

    Fixing the seeds is what makes two policies comparable: they face exactly the
    same maps, so a difference in success rate is a difference in skill and not
    in luck.
    """
    rng = np.random.default_rng(rng_seed)
    results: list[EpisodeResult] = []
    for i in range(episodes):
        result = run_episode(env, policy, seed=seed + i, rng=rng)
        results.append(result)
        if on_episode is not None:
            on_episode(i, result)
    return results


def summarize(results: Iterable[EpisodeResult]) -> dict[str, float]:
    results = list(results)
    n = len(results)
    successes = [r for r in results if r.outcome == "goal"]
    counts = {k: sum(r.outcome == k for r in results) for k in ("goal", "collision", "timeout")}

    return {
        "episodes": n,
        "success_rate": counts["goal"] / n,
        "collision_rate": counts["collision"] / n,
        "timeout_rate": counts["timeout"] / n,
        "mean_return": float(np.mean([r.ret for r in results])),
        "mean_steps_success": float(np.mean([r.steps for r in successes])) if successes else float("nan"),
        "std_steps_success": float(np.std([r.steps for r in successes])) if successes else float("nan"),
        "mean_efficiency": float(np.mean([r.efficiency for r in successes])) if successes else float("nan"),
    }


HEADER = (
    f"{'policy':<22}{'success':>9}{'collide':>9}{'timeout':>9}"
    f"{'return':>10}{'steps':>16}{'efficiency':>12}"
)


def format_row(label: str, s: dict[str, float]) -> str:
    steps = (
        f"{s['mean_steps_success']:6.1f}+-{s['std_steps_success']:<5.1f}"
        if s["mean_steps_success"] == s["mean_steps_success"]  # NaN check
        else f"{'--':>13}"
    )
    efficiency = (
        f"{s['mean_efficiency']:11.3f}"
        if s["mean_efficiency"] == s["mean_efficiency"]
        else f"{'--':>11}"
    )
    return (
        f"{label:<22}"
        f"{100 * s['success_rate']:8.1f}%"
        f"{100 * s['collision_rate']:8.1f}%"
        f"{100 * s['timeout_rate']:8.1f}%"
        f"{s['mean_return']:+10.2f}"
        f"{steps:>16}"
        f"{efficiency:>12}"
    )


def print_table(rows: list[tuple[str, dict[str, float]]], title: str = "") -> None:
    if title:
        print(f"\n{title}")
    print(HEADER)
    print("-" * len(HEADER))
    for label, summary in rows:
        print(format_row(label, summary))
    print("\nsteps = mean +- std over successful episodes only")
    print("efficiency = straight-line distance / distance driven (1.0 is optimal)")
