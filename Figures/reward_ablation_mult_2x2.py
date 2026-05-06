"""Subset figure for the main text: 2x2 reward ablation on multiplication.

  Rows:    CIA (top)        |  Accuracy (bottom)
  Cols:    Gemma-2-9B (L)   |  Qwen3-8B (R)

Lines per panel: acc_faith / acc_only / faith_only (full / no-faith / no-acc),
plus a horizontal dashed baseline for the un-fine-tuned base model.

Re-uses CSVs at Figures/runs/multiplication__<model>__<variant>.csv.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D


mpl.rcParams.update({
    "figure.dpi":         100,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     11,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.6,
    "lines.linewidth":    2.0,
})

RUNS_DIR = Path("${REPO_ROOT}/Figures/runs")
FIG_OUT  = Path("${REPO_ROOT}/Figures/out")
FIG_OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["gemma_9b", "qwen3_8b"]
MODEL_LABELS = {"gemma_9b": "Gemma-2-9B", "qwen3_8b": "Qwen3-8B"}

VARIANTS = {
    "acc_faith":  dict(label="Acc + Faith (full)", color="#E67E22", ls="-",  lw=2.5, z=3),
    "acc_only":   dict(label="Acc only",           color="#7F8C8D", ls="--", lw=2.0, z=2),
    "faith_only": dict(label="Faith only",         color="#3498DB", ls=":",  lw=2.0, z=2),
}

# Base-model values for the dashed horizontal baseline. Pulled from the
# 36-cell ablation table (build_ablation_table.py) on 2026-05-05 — Mult
# CIA uses B_CoT = self_consistent (force_b regime).
BASE = {
    ("gemma_9b", "cia"):      0.772,
    ("gemma_9b", "accuracy"): 0.646,
    ("qwen3_8b", "cia"):      0.601,
    ("qwen3_8b", "accuracy"): 0.742,
}


def rolling(y: np.ndarray, win: int = 3) -> np.ndarray:
    if len(y) <= win:
        return y
    return pd.Series(y).rolling(win, min_periods=1, center=True).mean().values


def load_run(model: str, variant: str) -> pd.DataFrame | None:
    p = RUNS_DIR / f"multiplication__{model}__{variant}.csv"
    return pd.read_csv(p) if p.exists() else None


def plot_panel(ax, model: str, metric: str, base_val: float | None,
               smoothing_win: int = 3) -> int:
    """Render one (model, metric) subplot across the 3 reward variants."""
    n_plotted = 0
    for vname, vcfg in VARIANTS.items():
        df = load_run(model, vname)
        if df is None or df.empty or metric not in df.columns:
            continue
        ys = df[metric].values.astype(float)
        if np.all(np.isnan(ys)):
            continue
        xs = df["step"].values if "step" in df.columns else np.arange(len(df))
        ys_s = rolling(ys, smoothing_win)
        ax.plot(xs, ys_s,
                color=vcfg["color"], linestyle=vcfg["ls"],
                linewidth=vcfg["lw"], zorder=vcfg["z"], label=vcfg["label"])

        if metric == "cia" and "cia_std" in df.columns:
            std = df["cia_std"].values.astype(float)
            ax.fill_between(xs, ys_s - std, ys_s + std,
                            color=vcfg["color"], alpha=0.12, zorder=1)

        i_peak = int(np.nanargmax(ys_s))
        ax.scatter([xs[i_peak]], [ys_s[i_peak]],
                   color=vcfg["color"], s=55, marker="*",
                   edgecolor="black", linewidth=0.6, zorder=5)
        n_plotted += 1

    if base_val is not None:
        ax.axhline(base_val, color="#2C3E50", linestyle="--",
                   linewidth=1.0, alpha=0.75, zorder=0)

    if n_plotted == 0:
        ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes,
                ha="center", va="center", color="#95A5A6")
    return n_plotted


def auto_ylim(metric: str, base_val: float | None) -> tuple[float, float]:
    if metric == "cia":
        return (0.2, 0.95)
    base_val = base_val if base_val is not None else 0.5
    return (max(0.0, base_val - 0.20), 1.0)


def draw(save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.20})

    for col, model in enumerate(MODELS):
        # CIA (top row)
        ax = axes[0, col]
        plot_panel(ax, model, metric="cia", base_val=BASE[(model, "cia")])
        ax.set_title(MODEL_LABELS[model], pad=6, fontweight="bold")
        ax.set_xlabel("")
        ax.set_xticks([0, 40, 80, 120])
        ax.set_ylim(*auto_ylim("cia", BASE[(model, "cia")]))
        if col == 0:
            ax.set_ylabel("CIA", fontweight="bold")

        # Accuracy (bottom row)
        ax = axes[1, col]
        plot_panel(ax, model, metric="accuracy",
                   base_val=BASE[(model, "accuracy")])
        ax.set_xlabel("Training step")
        ax.set_xticks([0, 40, 80, 120])
        ax.set_ylim(*auto_ylim("accuracy", BASE[(model, "accuracy")]))
        if col == 0:
            ax.set_ylabel("Accuracy", fontweight="bold")

    var_handles = [
        Line2D([0], [0], color=c["color"], linestyle=c["ls"],
               linewidth=c["lw"], label=c["label"])
        for c in VARIANTS.values()
    ]
    aux_handles = [
        Line2D([0], [0], color="#2C3E50", linestyle="--",
               linewidth=1.0, alpha=0.75, label="Base model"),
        Line2D([0], [0], color="black", marker="*", linestyle="None",
               markersize=8, markeredgecolor="black", label="Peak checkpoint"),
    ]
    fig.legend(handles=var_handles + aux_handles, loc="upper center",
               ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.02), columnspacing=1.5,
               handlelength=2.2)

    fig.suptitle("Reward-function ablation — 2-Digit Multiplication",
                 y=1.08, fontsize=11.5, fontweight="bold")
    fig.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    png = str(save_path.with_suffix(".png"))
    fig.savefig(png, dpi=250, bbox_inches="tight")
    print(f"Saved: {save_path}  +  {png}")
    return fig


if __name__ == "__main__":
    draw(FIG_OUT / "reward_ablation_mult_gemma_qwen3_2x2.pdf")
