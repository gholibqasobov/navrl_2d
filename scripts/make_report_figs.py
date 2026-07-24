"""Generate every figure and the metrics table for the report, from runs/ artifacts.

    python scripts/make_report_figs.py

Writes into report/:
    metrics_table.md               master comparison + compute/time tables
    figs/learning_curves.png       return + eval success vs env steps
    figs/baseline_ladder.png       success-rate ladder, random .. potential_field
    figs/obs_scaling_ablation.png  the ~65% ceiling + the per-feature std bug
    figs/trajectories.png          top-down rollouts, PPO vs an obstacle-blind fail

No training happens here; everything is recomputed from saved runs and models.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stable_baselines3 import PPO, SAC  # noqa: E402

from navrl2d.config import DEFAULT_CONFIG  # noqa: E402
from navrl2d.env import NavEnv2D  # noqa: E402
from navrl2d.evaluation import evaluate, summarize  # noqa: E402
from navrl2d.policies import SCRIPTED_POLICIES, greedy_policy  # noqa: E402
from plot_results import load_monitor, rolling  # noqa: E402

warnings.filterwarnings("ignore")

RUNS = ROOT / "runs"
OUT = ROOT / "report"
FIGS = OUT / "figs"
EVAL_SEED = 100_000
GREEDY_SUCCESS = 0.56  # obstacle-blind reference

# One consistent style for every learned/scripted series.
COLORS = {
    "ppo_3m": "#2460c8",
    "ppo_1m": "#4a90d9",
    "sac": "#e07b39",
    "greedy": "#c83c3c",
    "potential_field": "#22a05c",
    "old": "#9aa0aa",
}


def load_model(run: str):
    algo = PPO if "ppo" in run else SAC
    return algo.load(RUNS / run / "best_model", device="cpu")


def model_policy(run: str):
    model = load_model(run)
    return lambda obs, rng: model.predict(obs, deterministic=True)[0]


def eval_curve(run: str):
    """(timesteps, mean eval success) from a run's evaluations.npz."""
    d = np.load(RUNS / run / "evaluations.npz")
    return d["timesteps"], d["successes"].mean(axis=1)


def _smooth(y: np.ndarray, w: int = 5) -> np.ndarray:
    """Centered moving average that keeps the array length (for noisy 20-ep evals)."""
    if len(y) < w:
        return y
    kernel = np.ones(w) / w
    pad = w // 2
    padded = np.pad(y, pad, mode="edge")
    return np.convolve(padded, kernel, mode="same")[pad:-pad]


# --------------------------------------------------------------------------- #
# Master metric table (also feeds the baseline ladder)
# --------------------------------------------------------------------------- #
def compute_metrics(episodes: int = 200) -> dict[str, dict]:
    env = NavEnv2D()
    rows: dict[str, dict] = {}
    learned = {
        "PPO (3M)": model_policy("ppo_3m"),
        "PPO (1M)": model_policy("ppo_1m"),
        "SAC (300k)": model_policy("sac_verify"),
    }
    scripted = {
        "potential_field": SCRIPTED_POLICIES["potential_field"],
        "greedy": SCRIPTED_POLICIES["greedy"],
        "still": SCRIPTED_POLICIES["still"],
        "random": SCRIPTED_POLICIES["random"],
    }
    for label, policy in {**learned, **scripted}.items():
        rows[label] = summarize(evaluate(env, policy, episodes, seed=EVAL_SEED))
    env.close()
    return rows


def write_metrics_table(rows: dict[str, dict]) -> None:
    order = [
        "potential_field", "PPO (3M)", "PPO (1M)", "greedy",
        "SAC (300k)", "still", "random",
    ]
    note = {
        "potential_field": "obs-only ceiling (reactive controller)",
        "greedy": "obstacle-blind bar",
        "still": "never moves",
        "random": "uniform actions",
    }
    lines = [
        "# Metrics",
        "",
        f"Fixed seed block {EVAL_SEED}..{EVAL_SEED + 199}, deterministic actions "
        "(200 episodes each, identical maps).",
        "",
        "| policy | success | collide | timeout | mean return | steps (success) | path eff. | |",
        "|---|--:|--:|--:|--:|--:|--:|---|",
    ]
    for label in order:
        s = rows[label]
        steps = (
            f"{s['mean_steps_success']:.0f}±{s['std_steps_success']:.0f}"
            if s["mean_steps_success"] == s["mean_steps_success"] else "—"
        )
        eff = (
            f"{s['mean_efficiency']:.2f}"
            if s["mean_efficiency"] == s["mean_efficiency"] else "—"
        )
        lines.append(
            f"| {label} | {s['success_rate']:.0%} | {s['collision_rate']:.0%} | "
            f"{s['timeout_rate']:.0%} | {s['mean_return']:+.1f} | {steps} | {eff} | "
            f"{note.get(label, '')} |"
        )

    lines += ["", "## Compute / time (CPU only)", "",
              "| run | algo | steps | wall-clock | throughput |",
              "|---|---|--:|--:|--:|"]
    for run in ["ppo_1m", "ppo_3m", "sac_verify"]:
        m = json.loads((RUNS / run / "meta.json").read_text())
        hp = m["hyperparameters"]
        lines.append(
            f"| {run} | {m['algo'].upper()} | {hp['total_timesteps']:,} | "
            f"{m['wall_clock_minutes']:.1f} min | {m['steps_per_second']:,.0f} steps/s |"
        )
    lines.append("")
    (OUT / "metrics_table.md").write_text("\n".join(lines))
    print("wrote report/metrics_table.md")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_learning_curves() -> None:
    runs = [("ppo_3m", "PPO (3M)"), ("ppo_1m", "PPO (1M)"), ("sac_verify", "SAC (300k)")]
    fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(11, 4.2))

    for run, label in runs:
        key = "sac" if "sac" in run else run
        df = load_monitor(RUNS / run)
        ax_r.plot(df["timesteps"], rolling(df["r"], 200), color=COLORS[key], label=label)
        ts, succ = eval_curve(run)
        ax_s.plot(ts, succ, color=COLORS[key], label=label, marker="o", ms=3)

    ax_r.axhline(0, color="gray", lw=0.8, ls=":")
    ax_r.set(xlabel="environment steps", ylabel="episode return",
             title="Training return")
    ax_r.grid(alpha=0.3)
    ax_r.legend(fontsize=9)

    ax_s.axhline(GREEDY_SUCCESS, color=COLORS["greedy"], ls="--", lw=1,
                 label=f"greedy baseline ({GREEDY_SUCCESS:.0%})")
    ax_s.set(xlabel="environment steps", ylabel="success rate", ylim=(0, 1.02),
             title="Success rate")
    ax_s.grid(alpha=0.3)
    ax_s.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGS / "learning_curves.png", dpi=140)
    plt.close(fig)
    print("wrote report/figs/learning_curves.png")


def fig_baseline_ladder(rows: dict[str, dict]) -> None:
    order = ["random", "SAC (300k)", "greedy", "PPO (1M)", "PPO (3M)"]
    succ = [rows[k]["success_rate"] for k in order]
    colors = []
    for k in order:
        if "PPO" in k:
            colors.append(COLORS["ppo_1m"])
        elif "SAC" in k:
            colors.append(COLORS["sac"])
        elif k == "greedy":
            colors.append(COLORS["greedy"])
        elif k == "potential_field":
            colors.append(COLORS["potential_field"])
        else:
            colors.append(COLORS["old"])

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(order, succ, color=colors)
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in succ], padding=4, fontsize=9)
    ax.axvline(GREEDY_SUCCESS, color=COLORS["greedy"], ls="--", lw=1)
    ax.set(xlim=(0, 1.0), xlabel="success rate", title="Success ladder")
    ax.text(0.575, -0.65, "obstacle-blind bar", color=COLORS["greedy"], fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "baseline_ladder.png", dpi=140)
    plt.close(fig)
    print("wrote report/figs/baseline_ladder.png")


def _feature_stds(old: bool) -> np.ndarray:
    """Per-feature std of the 12-D observation under new vs old length scaling.

    Raw (unnormalised) obs give distances in world units and radii in [0.5,1.0];
    we then apply either the shipped per-feature scaling (obs_dist_scale) or the
    naive 'divide every length by the arena diagonal' scaling.
    """
    cfg = DEFAULT_CONFIG
    diag = cfg.diagonal
    r_mid = 0.5 * (cfg.obs_radius_max + cfg.obs_radius_min)
    r_half = 0.5 * (cfg.obs_radius_max - cfg.obs_radius_min)

    env = NavEnv2D(normalize_obs=False)
    rng = np.random.default_rng(0)
    raw = []
    obs, _ = env.reset(seed=0)
    for _ in range(4000):
        raw.append(obs.copy())
        obs, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 2))
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    raw = np.array(raw)

    scaled = raw.copy()
    length_idx = [0, 3, 4, 6, 7, 9, 10]   # goal dist + obstacle dx,dy
    radius_idx = [5, 8, 11]
    if old:
        scaled[:, length_idx] = raw[:, length_idx] / diag
        scaled[:, radius_idx] = raw[:, radius_idx] / diag
    else:
        scaled[:, length_idx] = raw[:, length_idx] / cfg.obs_dist_scale
        scaled[:, radius_idx] = (raw[:, radius_idx] - r_mid) / r_half
    # cos/sin (idx 1,2) are already unit-scaled in both schemes
    return scaled.std(axis=0)


def fig_obs_scaling_ablation() -> None:
    fig, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(11, 4.2))

    for run, label, col in [("ppo_1m", "per-feature scaling (shipped)", COLORS["ppo_1m"]),
                            ("ppo_1m_oldobs", "÷ arena diagonal (naive)", COLORS["old"])]:
        ts, succ = eval_curve(run)
        ax_c.plot(ts, succ, color=col, alpha=0.22, lw=1)
        ax_c.plot(ts, _smooth(succ), color=col, label=label, lw=2)
    ax_c.axhline(GREEDY_SUCCESS, color=COLORS["greedy"], ls="--", lw=1, label="greedy (56%)")
    ax_c.axhspan(0.60, 0.68, color="red", alpha=0.06)
    ax_c.text(ax_c.get_xlim()[1] * 0.98, 0.645, "~65% ceiling", color="red",
              fontsize=8, ha="right", va="center")
    ax_c.set(xlabel="environment steps", ylabel="success rate", ylim=(0, 1.02),
             title="Observation scaling")
    ax_c.grid(alpha=0.3)
    ax_c.legend(fontsize=8, loc="lower right")

    labels = ["goal\ndist", "cos", "sin", "o1 dx", "o1 dy", "o1 r",
              "o2 dx", "o2 dy", "o2 r", "o3 dx", "o3 dy", "o3 r"]
    x = np.arange(12)
    ax_s.bar(x - 0.2, _feature_stds(old=False), 0.4, color=COLORS["ppo_1m"],
             label="per-feature (shipped)")
    ax_s.bar(x + 0.2, _feature_stds(old=True), 0.4, color=COLORS["old"],
             label="÷ diagonal (naive)")
    ax_s.axhline(0.1, color="red", ls=":", lw=1)
    ax_s.text(11.4, 0.11, "std 0.1", color="red", fontsize=7, ha="right")
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax_s.set(ylabel="feature std over a rollout", title="Feature std")
    ax_s.legend(fontsize=8)
    ax_s.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGS / "obs_scaling_ablation.png", dpi=140)
    plt.close(fig)
    print("wrote report/figs/obs_scaling_ablation.png")


def _rollout(policy, seed: int):
    env = NavEnv2D()
    obs, _ = env.reset(seed=seed)
    core = env.unwrapped
    start = core.robot[:2].copy()
    obstacles = core.obstacles.copy()
    goal = core.goal.copy()
    path = [start.copy()]
    term = trunc = False
    info = {}
    while not (term or trunc):
        obs, _, term, trunc, info = env.step(policy(obs, None))
        path.append(core.robot[:2].copy())
    env.close()
    outcome = ("goal" if info["is_success"] else "collision" if info["collision"] else "timeout")
    return np.array(path), obstacles, start, goal, outcome


def _draw(ax, path, obstacles, start, goal, outcome, title):
    size = DEFAULT_CONFIG.arena_size
    ax.add_patch(plt.Rectangle((0, 0), size, size, fill=False, ec="#3f424a", lw=1.5))
    for cx, cy, r in obstacles:
        ax.add_patch(plt.Circle((cx, cy), r, color="#6c707a", alpha=0.85))
    color = {"goal": COLORS["potential_field"], "collision": COLORS["greedy"],
             "timeout": COLORS["sac"]}[outcome]
    ax.plot(path[:, 0], path[:, 1], color=color, lw=1.8)
    ax.plot(*start, "o", color="#2460c8", ms=7)
    ax.plot(*goal, "*", color="#22a05c", ms=15)
    if outcome == "collision":
        ax.plot(*path[-1], "x", color=COLORS["greedy"], ms=10, mew=2.5)
    ax.set(xlim=(-0.3, size + 0.3), ylim=(-0.3, size + 0.3), title=title)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def fig_trajectories() -> None:
    ppo = model_policy("ppo_3m")
    # First three: PPO reaches the goal on maps where the obstacle-blind greedy
    # controller crashes (obstacles sit on the direct line). Last: same map with
    # both paths overlaid, to make the avoidance explicit.
    seeds = [100005, 100006, 100008]
    contrast_seed = 100010

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, s in zip(axes[:3], seeds):
        path, obs, start, goal, outcome = _rollout(ppo, s)
        _draw(ax, path, obs, start, goal, outcome, f"seed {s} — {outcome}")

    gp, obs, start, goal, g_out = _rollout(greedy_policy, contrast_seed)
    _draw(axes[3], gp, obs, start, goal, g_out, "")
    pp, obs, start, goal, p_out = _rollout(ppo, contrast_seed)
    axes[3].plot(pp[:, 0], pp[:, 1], color=COLORS["potential_field"], lw=1.8, ls="--")
    axes[3].set_title("greedy ✖ vs PPO ★ (same map)", fontsize=10)

    fig.suptitle("● start   ★ goal   ✖ collision", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "trajectories.png", dpi=140)
    plt.close(fig)
    print("wrote report/figs/trajectories.png")


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    rows = compute_metrics()
    write_metrics_table(rows)
    fig_learning_curves()
    fig_baseline_ladder(rows)
    fig_obs_scaling_ablation()
    fig_trajectories()
    print("\nall report figures + metrics_table.md written under report/")


if __name__ == "__main__":
    main()
