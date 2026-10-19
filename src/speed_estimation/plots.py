"""Figures: Fig. 5 (metric bar chart) and the synthetic-validation comparison."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#8a5cc7", "#1baf7a"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
METRICS = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 score")]


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_metrics(metrics: dict, path: str | Path, title: str) -> None:
    """Fig. 5: one bar per metric, value printed on each bar."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    labels = [name for _, name in METRICS]
    values = [float(metrics[key]) for key, _ in METRICS]
    bars = ax.bar(labels, values, width=0.55, color=SERIES[0])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", color=MUTED, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_simulation(summary: pd.DataFrame, details: pd.DataFrame, path: str | Path) -> None:
    """Left: the four metrics per variant. Right: distribution of speed errors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150,
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    variants = list(summary["variant"])
    x = np.arange(len(METRICS))
    width = 0.8 / len(variants)
    for i, (variant, color) in enumerate(zip(variants, SERIES)):
        row = summary[summary["variant"] == variant].iloc[0]
        ax1.bar(x + (i - (len(variants) - 1) / 2) * width,
                [row[k] for k, _ in METRICS], width=width * 0.92, color=color, label=variant)
    ax1.set_xticks(x, [n for _, n in METRICS])
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score", color=MUTED, fontsize=9)
    ax1.set_title("Metrics on synthetic traffic with known speeds", color=INK, fontsize=11, loc="left")
    ax1.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
    _style(ax1)

    data = []
    for variant in variants:
        d = details[(details["variant"] == variant) & details["outcome"].isin(["correct", "wrong"])]
        data.append((d["estimated_speed_kmh"] - d["true_speed_kmh"]).to_numpy())
    parts = ax2.boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "linewidth": 1.2})
    for patch, color in zip(parts["boxes"], SERIES):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.85)
    for w in parts["whiskers"] + parts["caps"]:
        w.set_color(MUTED)
    ax2.axvline(0, color=MUTED, linewidth=1)
    ax2.set_yticks(range(1, len(variants) + 1), [v.replace(" (", "\n(") for v in variants], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Estimated − true speed (km/h)", color=MUTED, fontsize=9)
    ax2.set_title("Speed error per vehicle", color=INK, fontsize=11, loc="left")
    _style(ax2)
    ax2.yaxis.grid(False)
    ax2.xaxis.grid(True, color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
