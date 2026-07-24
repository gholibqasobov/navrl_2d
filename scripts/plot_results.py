"""Plot learning curves from the monitor logs of one or more runs.

    python scripts/plot_results.py runs/ppo_1m runs/sac_300k
    python scripts/plot_results.py runs/*            --out runs/learning_curves.png

Left panel:  episode return vs environment steps.
Right panel: success rate vs environment steps, with the greedy baseline drawn
             as a horizontal line -- the only reference that matters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write a file, never try to open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GREEDY_SUCCESS = 0.567  # scripts/run_baseline.py --policy greedy, 300 episodes


def load_monitor(run_dir: Path) -> pd.DataFrame:
    """Concatenate the per-worker monitor CSVs of a run, ordered by wall time."""
    files = sorted(run_dir.glob("*.monitor.csv"))
    if not files:
        raise SystemExit(f"no monitor*.csv in {run_dir}")

    frames = [pd.read_csv(f, skiprows=1) for f in files]
    df = pd.concat(frames).sort_values("t").reset_index(drop=True)
    # Episodes finish out of order across workers; cumulative length is the
    # honest x-axis (total environment steps consumed so far).
    df["timesteps"] = df["l"].cumsum()
    return df


def rolling(series: pd.Series, window: int) -> np.ndarray:
    return series.rolling(window, min_periods=max(window // 10, 1)).mean().to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--window", type=int, default=200, help="rolling-mean window, episodes")
    parser.add_argument("--out", type=Path, default=Path("runs/learning_curves.png"))
    args = parser.parse_args()

    fig, (ax_return, ax_success) = plt.subplots(1, 2, figsize=(12, 4.5))

    for run_dir in args.runs:
        if not run_dir.is_dir():
            continue
        df = load_monitor(run_dir)
        label = f"{run_dir.name} ({len(df)} episodes)"
        ax_return.plot(df["timesteps"], rolling(df["r"], args.window), label=label)
        if "is_success" in df:
            ax_success.plot(
                df["timesteps"], rolling(df["is_success"].astype(float), args.window),
                label=label,
            )

    ax_return.set(xlabel="environment steps", ylabel="episode return",
                  title=f"return (rolling mean, {args.window} episodes)")
    ax_return.axhline(0, color="gray", lw=0.8, ls=":")
    ax_return.grid(alpha=0.3)
    ax_return.legend(fontsize=8)

    ax_success.axhline(GREEDY_SUCCESS, color="crimson", ls="--", lw=1,
                       label=f"greedy baseline ({GREEDY_SUCCESS:.0%})")
    ax_success.set(xlabel="environment steps", ylabel="success rate", ylim=(0, 1.02),
                   title="success rate during training")
    ax_success.grid(alpha=0.3)
    ax_success.legend(fontsize=8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
