#!/usr/bin/env python
"""Plot the training curves from a checkpoint's trainer_state.json.

    source scripts/env.sh
    python $GROOT_PROJECT/scripts/plot_training.py

Reads log_history (loss / grad_norm / learning_rate per logging step) and
writes one combined figure plus one PNG per metric into results/.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
METRICS = [
    ("loss", "Training loss", "#1f77b4", True),
    ("grad_norm", "Gradient norm", "#d62728", False),
    ("learning_rate", "Learning rate", "#2ca02c", False),
]


def load_history(state_path):
    state = json.loads(state_path.read_text())
    return state["log_history"], state


def series(history, key):
    pts = [(e["step"], e[key]) for e in history if key in e]
    return [p[0] for p in pts], [p[1] for p in pts]


def smooth(values, window=20):
    """Trailing mean, so the smoothed curve never leads the raw one."""
    out, acc = [], []
    for v in values:
        acc.append(v)
        if len(acc) > window:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state",
        type=Path,
        default=PROJECT / "checkpoints/so101_ft/checkpoint-3000/trainer_state.json",
    )
    ap.add_argument("--outdir", type=Path, default=PROJECT / "results")
    ap.add_argument("--title", default="GR00T-N1.7-3B — so101 table cleanup")
    args = ap.parse_args()

    history, state = load_history(args.state)
    args.outdir.mkdir(parents=True, exist_ok=True)

    subtitle = (
        f"{state['global_step']} steps · {state['epoch']:.2f} epochs · "
        f"final loss {series(history, 'loss')[1][-1]:.4f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (key, label, color, do_smooth) in zip(axes, METRICS):
        steps, vals = series(history, key)
        if not steps:
            ax.set_visible(False)
            continue
        ax.plot(steps, vals, color=color, lw=0.9, alpha=0.35 if do_smooth else 0.9)
        if do_smooth:
            ax.plot(steps, smooth(vals), color=color, lw=2.0, label="smoothed (20)")
            ax.legend(frameon=False, fontsize=9)
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25, lw=0.5)

        # One metric per file as well, for dropping into slides.
        f1, a1 = plt.subplots(figsize=(7, 4.5))
        a1.plot(steps, vals, color=color, lw=0.9, alpha=0.35 if do_smooth else 0.9)
        if do_smooth:
            a1.plot(steps, smooth(vals), color=color, lw=2.0, label="smoothed (20)")
            a1.legend(frameon=False, fontsize=9)
        a1.set_title(f"{label} — {subtitle}", fontsize=10)
        a1.set_xlabel("step")
        a1.grid(alpha=0.25, lw=0.5)
        f1.tight_layout()
        f1.savefig(args.outdir / f"{key}.png", dpi=150)
        plt.close(f1)

    fig.suptitle(f"{args.title}\n{subtitle}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    combined = args.outdir / "training_curves.png"
    fig.savefig(combined, dpi=150)
    plt.close(fig)

    print(f"wrote {combined}")
    for key, *_ in METRICS:
        print(f"wrote {args.outdir / f'{key}.png'}")


if __name__ == "__main__":
    main()
