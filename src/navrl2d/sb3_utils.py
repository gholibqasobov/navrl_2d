"""Stable-Baselines3 glue: env factories and a training-statistics callback.

Kept separate from `env.py` so the environment itself has no dependency on SB3 —
the env stays a plain Gymnasium env that any framework could consume.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from .config import EnvConfig
from .env import NavEnv2D

# Monitor copies these keys from the final `info` dict of each episode into
# monitor.csv and into model.ep_info_buffer, which is what makes success and
# collision rates visible during training.
INFO_KEYWORDS = ("is_success", "collision", "timeout")


def make_env(
    seed: int | None = None,
    config: EnvConfig | None = None,
    render_mode: str | None = None,
    monitor_path: str | None = None,
) -> Callable[[], Monitor]:
    """Return a thunk that builds one monitored environment (SB3 wants a factory)."""

    def _init() -> Monitor:
        env = NavEnv2D(config=config, render_mode=render_mode)
        env = Monitor(env, filename=monitor_path, info_keywords=INFO_KEYWORDS)
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        return env

    return _init


def make_vec_env(
    n_envs: int,
    seed: int = 0,
    config: EnvConfig | None = None,
    monitor_path: str | None = None,
) -> VecEnv:
    """Vectorised env. Each worker gets its own seed offset.

    This environment steps in a few microseconds, so process-based parallelism
    only pays for itself once the IPC cost is amortised over several workers --
    below ~4 envs DummyVecEnv (a plain loop, no IPC) is faster.
    """
    # Every worker needs its OWN monitor file: with SubprocVecEnv they are
    # separate processes, and pointing them at one path makes their writes
    # interleave into a corrupt CSV.
    fns = [
        make_env(
            seed=seed + i,
            config=config,
            monitor_path=f"{monitor_path}_{i}" if monitor_path else None,
        )
        for i in range(n_envs)
    ]
    if n_envs == 1:
        return DummyVecEnv(fns)
    return SubprocVecEnv(fns, start_method="fork")


class RolloutStatsCallback(BaseCallback):
    """Log success / collision / timeout rates of recent training episodes.

    SB3 prints `ep_rew_mean` out of the box, but a rising return cannot tell you
    whether the agent is reaching goals or merely learning not to crash. These
    three rates always sum to 1 and say exactly what the agent is doing.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)

    def _on_step(self) -> bool:  # required by the ABC; per-step work is not needed
        return True

    def _on_rollout_end(self) -> None:
        self._log_rates()

    def _log_rates(self) -> None:
        buffer = self.model.ep_info_buffer
        if not buffer:
            return
        for key, name in (
            ("is_success", "rollout/success_rate"),
            ("collision", "rollout/collision_rate"),
            ("timeout", "rollout/timeout_rate"),
        ):
            values = [ep[key] for ep in buffer if key in ep]
            if values:
                self.logger.record(name, float(np.mean(values)))


class OffPolicyStatsCallback(RolloutStatsCallback):
    """Same rates for algorithms that never fire `_on_rollout_end` per episode.

    SAC's rollout boundaries are per `train_freq`, so logging there would be very
    noisy; instead the rates are recomputed every `log_interval_steps`.
    """

    def __init__(self, log_interval_steps: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_interval_steps = log_interval_steps

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_interval_steps == 0:
            self._log_rates()
        return True

    def _on_rollout_end(self) -> None:  # would fire every train_freq steps
        pass
