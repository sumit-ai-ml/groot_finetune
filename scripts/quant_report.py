#!/usr/bin/env python
"""Turn results/quant_sweep.jsonl into a comparison table and figures.

    source scripts/env.sh
    python $GROOT_PROJECT/scripts/quant_report.py

The FP rows carry more than one seed. Their spread is the noise floor: the
action head samples noise, so any quantization delta smaller than the seed-to-
seed spread is not evidence of degradation. Every plot draws that band.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
COLORS = {"apexquant": "#1f77b4", "quarot": "#d62728", "rtn_absmax": "#7f7f7f"}
MARKERS = {"apexquant": "o", "quarot": "s", "rtn_absmax": "^"}
GB = 1e9


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    fp = [r for r in rows if r["method"] == "fp"]
    quant = [r for r in rows if r["method"] != "fp"]
    return fp, quant


def band(fp, key):
    """(reference value, min, max) across FP seeds."""
    vals = [r[key] for r in fp]
    return fp[0][key], min(vals), max(vals)


def fmt(x, nd=4):
    if x != x or abs(x) > 1e4:
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def table(fp, quant, key="mse"):
    ref, lo, hi = band(fp, key)
    lines = [
        f"FP reference {key} = {fmt(ref)}   (seed spread {fmt(lo)} .. {fmt(hi)}, "
        f"n={len(fp)} seeds)",
        "",
        f"{'point':28s} {'mse':>10s} {'vs FP':>9s} {'term_l2':>9s} {'drift':>7s} "
        f"{'w_mse':>10s} {'GB':>6s} {'quant_s':>8s} {'ms/act':>7s}",
        "-" * 100,
    ]
    for r in quant:
        delta = r["mse"] / ref if ref else float("nan")
        lines.append(
            f"{r['label']:28s} {fmt(r['mse']):>10s} {delta:>8.2f}x "
            f"{fmt(r['terminal_l2'], 3):>9s} {r['drift_ratio']:>7.2f} "
            f"{r['weight_mse']:>10.2e} {r['packed_bytes']/GB:>6.2f} "
            f"{r['quant_seconds']:>8.0f} {1000*r['latency_s']:>7.0f}"
        )
    return "\n".join(lines)


def draw_band(ax, fp, key):
    ref, lo, hi = band(fp, key)
    ax.axhspan(lo, hi, color="0.85", zorder=0,
               label=f"FP seed spread (n={len(fp)})")
    ax.axhline(ref, color="0.35", ls="--", lw=1.2, zorder=1, label="FP reference")


def plot_vs_bits(fp, quant, key, ylabel, outpath, logy=True):
    both = [r for r in quant if r["tower"] == "both"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    draw_band(ax, fp, key)
    for method in sorted({r["method"] for r in both}):
        pts = sorted([r for r in both if r["method"] == method], key=lambda r: r["bits"])
        ax.plot([p["bits"] for p in pts], [p[key] for p in pts],
                marker=MARKERS.get(method, "o"), color=COLORS.get(method), lw=1.8,
                ms=7, label=method)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("weight bits")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({r["bits"] for r in both}))
    ax.invert_xaxis()
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"{ylabel} vs weight bit width (whole model)", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_vs_size(fp, quant, outpath):
    both = [r for r in quant if r["tower"] == "both"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    draw_band(ax, fp, "mse")
    fp_gb = fp[0]["packed_bytes"] / GB
    ax.axvline(fp_gb, color="0.35", ls=":", lw=1.2)
    ax.annotate(f"FP {fp_gb:.1f} GB", (fp_gb, ax.get_ylim()[1]), fontsize=8,
                ha="right", va="top", rotation=90, color="0.35")
    for method in sorted({r["method"] for r in both}):
        pts = sorted([r for r in both if r["method"] == method],
                     key=lambda r: r["packed_bytes"])
        ax.plot([p["packed_bytes"] / GB for p in pts], [p["mse"] for p in pts],
                marker=MARKERS.get(method, "o"), color=COLORS.get(method), lw=1.8,
                ms=7, label=method)
        for p in pts:
            ax.annotate(f"{p['bits']}b", (p["packed_bytes"] / GB, p["mse"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8,
                        color=COLORS.get(method))
    ax.set_yscale("log")
    ax.set_xlabel("packed weight size (GB)")
    ax.set_ylabel("normalized action MSE")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Accuracy vs model size — the tradeoff the sweep is about", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_towers(fp, quant, outpath):
    ablation = [r for r in quant if r["tower"] != "both"]
    if not ablation:
        return
    bits = ablation[0]["bits"]
    method = ablation[0]["method"]
    both = [r for r in quant
            if r["tower"] == "both" and r["bits"] == bits and r["method"] == method]
    rows = ablation + both
    ref, lo, hi = band(fp, "mse")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [r["tower"] for r in rows]
    ax.bar(labels, [r["mse"] for r in rows], color="#1f77b4", width=0.55)
    ax.axhspan(lo, hi, color="0.85", zorder=0)
    ax.axhline(ref, color="0.35", ls="--", lw=1.2, label="FP reference")
    for i, r in enumerate(rows):
        ax.annotate(f"{r['mse']:.3f}", (i, r["mse"]), ha="center",
                    textcoords="offset points", xytext=(0, 3), fontsize=9)
    ax.set_ylabel("normalized action MSE")
    ax.set_title(f"Which tower carries the loss — {method} @ {bits}-bit", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(PROJECT / "results/quant_sweep.jsonl"))
    ap.add_argument("--outdir", type=Path, default=PROJECT / "results")
    args = ap.parse_args()

    fp, quant = load(args.sweep)
    args.outdir.mkdir(parents=True, exist_ok=True)

    report = table(fp, quant)
    print(report)
    (args.outdir / "quant_comparison.txt").write_text(report + "\n")

    plot_vs_bits(fp, quant, "mse", "normalized action MSE",
                 args.outdir / "quant_mse_vs_bits.png")
    plot_vs_bits(fp, quant, "terminal_l2", "terminal L2 error",
                 args.outdir / "quant_terminal_vs_bits.png")
    plot_vs_size(fp, quant, args.outdir / "quant_accuracy_vs_size.png")
    plot_towers(fp, quant, args.outdir / "quant_tower_ablation.png")
    print(f"\nwrote figures + table to {args.outdir}")


if __name__ == "__main__":
    main()
