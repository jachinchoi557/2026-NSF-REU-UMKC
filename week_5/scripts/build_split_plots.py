"""
Generate four individual panel PNGs from week5_results_table.csv.

Output files (all to tairo_results/outputs/):
    panel_a_success_rate.png
    panel_b_avg_reward.png
    panel_c_recovery_rate.png
    panel_d_trustworthiness.png

Run:
    python scripts/build_split_plots.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from config import OUTPUTS_DIR

TABLE_PATH = os.path.join(OUTPUTS_DIR, "week5_results_table.csv")

# ---------------------------------------------------------------------------
# Condition ordering — source names match the "Condition" column in the CSV
# ---------------------------------------------------------------------------
COND_SOURCE_ORDER = [
    "Clean",
    "Action Delay",
    "Action Reverse",
    "Action Clipping",
    "Sensor Dropout",
    "Sensor Bias",
    "Goal Spoof (t=0)",
    "Goal Spoof (t=20)",
]

COND_DISPLAY = [
    "Clean",
    "Action\nDelay",
    "Action\nReverse",
    "Action\nClip 0.30",
    "Sensor\nDropout",
    "Sensor\nBias 0.10",
    "Goal Spoof\n(t=0)",
    "Goal Spoof\n(t=20)",
]

N_CONDS = len(COND_SOURCE_ORDER)

# ---------------------------------------------------------------------------
# Method order and styling
# ---------------------------------------------------------------------------
METHOD_ORDER = [
    "Random",
    "Rule-Based",
    "SAC+HER",
    "Random + Recovery",
    "Rule-Based + Recovery",
    "SAC+HER + Recovery",
    "SAC (no HER)",
    "SAC (no HER) + Recovery",
]

COLORS = {
    "Random":                  "#4472C4",
    "Rule-Based":              "#70AD47",
    "SAC+HER":                 "#ED7D31",
    "Random + Recovery":       "#4472C4",
    "Rule-Based + Recovery":   "#70AD47",
    "SAC+HER + Recovery":      "#ED7D31",
    "SAC (no HER)":            "#7F7F7F",
    "SAC (no HER) + Recovery": "#7F7F7F",
}

HATCH = {m: "//" if "Recovery" in m else "" for m in METHOD_ORDER}

RECOVERY_METHODS = [m for m in METHOD_ORDER if "Recovery" in m]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_val(df: pd.DataFrame, method: str, cond_source: str, metric: str) -> float:
    sub = df[(df["Method"] == method) & (df["Condition"] == cond_source)]
    if len(sub) == 0 or sub[metric].isna().all():
        return np.nan
    return float(sub[metric].iloc[0])


def bar_style(ax):
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", which="both", length=0)


def make_legend_handles(methods):
    return [
        mpatches.Patch(
            facecolor=COLORS[m],
            hatch=HATCH[m],
            edgecolor="#555",
            linewidth=0.5,
            label=m,
        )
        for m in methods
    ]


def set_xticks(ax, x_centers):
    ax.set_xticks(x_centers)
    ax.set_xticklabels(COND_DISPLAY, ha="center", fontsize=9)
    ax.tick_params(axis="x", which="both", length=0)


def add_ablation_marker(ax):
    """Dashed vertical line and annotation after the last condition group."""
    ax.axvline(x=7.62, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(7.67, -0.12,
            "SAC (no HER)\nablation",
            transform=ax.get_xaxis_transform(),
            ha="left", va="top",
            fontsize=7.5, color="#555555", fontstyle="italic")


# ---------------------------------------------------------------------------
# Panel A — Success Rate
# ---------------------------------------------------------------------------
def panel_a(df: pd.DataFrame, out_dir: str) -> None:
    bar_width = 0.09
    x = np.arange(N_CONDS, dtype=float)
    offsets = np.linspace(-3.5 * bar_width, 3.5 * bar_width, len(METHOD_ORDER))

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    for j, method in enumerate(METHOD_ORDER):
        vals = [get_val(df, method, c, "Success Rate") for c in COND_SOURCE_ORDER]
        ax.bar(x + offsets[j], vals,
               width=bar_width * 0.90,
               color=COLORS[method],
               hatch=HATCH[method],
               edgecolor="white",
               linewidth=0.5,
               zorder=3)

    add_ablation_marker(ax)

    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.set_ylabel("Success Rate", fontsize=10, labelpad=8)
    ax.set_title("Panel A — Success Rate by Method & Condition",
                 fontsize=11, fontweight="bold", pad=12, loc="left")

    set_xticks(ax, x)
    ax.set_xlim(-0.55, 8.05)
    bar_style(ax)

    handles = make_legend_handles(METHOD_ORDER)
    ax.legend(handles=handles,
              bbox_to_anchor=(0.5, 1.02), loc="lower center",
              ncol=4, fontsize=9, frameon=False,
              handlelength=1.4, handleheight=1.0, columnspacing=0.8)

    plt.tight_layout(pad=1.5)
    out = os.path.join(out_dir, "panel_a_success_rate.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Panel B — Avg Reward
# ---------------------------------------------------------------------------
def panel_b(df: pd.DataFrame, out_dir: str) -> None:
    bar_width = 0.09
    x = np.arange(N_CONDS, dtype=float)
    offsets = np.linspace(-3.5 * bar_width, 3.5 * bar_width, len(METHOD_ORDER))

    all_vals = [
        get_val(df, m, c, "Avg Reward")
        for m in METHOD_ORDER for c in COND_SOURCE_ORDER
    ]
    finite = [v for v in all_vals if not np.isnan(v)]
    floor_val = np.floor(min(finite) / 5) * 5

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    for j, method in enumerate(METHOD_ORDER):
        vals = [get_val(df, method, c, "Avg Reward") for c in COND_SOURCE_ORDER]
        ax.bar(x + offsets[j], vals,
               width=bar_width * 0.90,
               color=COLORS[method],
               hatch=HATCH[method],
               edgecolor="white",
               linewidth=0.5,
               zorder=3)

    add_ablation_marker(ax)

    ax.set_ylim(floor_val - 2, 2)
    ax.set_ylabel("Avg Reward", fontsize=10, labelpad=8)
    ax.set_title("Panel B — Avg Reward by Method & Condition",
                 fontsize=11, fontweight="bold", pad=12, loc="left")

    set_xticks(ax, x)
    ax.set_xlim(-0.55, 8.05)
    bar_style(ax)

    handles = make_legend_handles(METHOD_ORDER)
    ax.legend(handles=handles,
              bbox_to_anchor=(0.5, 1.02), loc="lower center",
              ncol=4, fontsize=9, frameon=False,
              handlelength=1.4, handleheight=1.0, columnspacing=0.8)

    plt.tight_layout(pad=1.5)
    out = os.path.join(out_dir, "panel_b_avg_reward.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Panel C — Recovery Triggered Rate
# ---------------------------------------------------------------------------
def panel_c(df: pd.DataFrame, out_dir: str) -> None:
    bar_width = 0.18
    x = np.arange(N_CONDS, dtype=float)
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, len(RECOVERY_METHODS))

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    for j, method in enumerate(RECOVERY_METHODS):
        vals = []
        for c in COND_SOURCE_ORDER:
            v = get_val(df, method, c, "Recovery Triggered Rate")
            vals.append(0.0 if np.isnan(v) else v)
        ax.bar(x + offsets[j], vals,
               width=bar_width * 0.88,
               color=COLORS[method],
               hatch=HATCH[method],
               edgecolor="white",
               linewidth=0.5,
               zorder=3)

    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.set_ylabel("Recovery Triggered\nRate (% of steps)", fontsize=9, labelpad=8)
    ax.set_title("Panel C — Recovery Triggered Rate (recovery variants only)",
                 fontsize=11, fontweight="bold", pad=12, loc="left")

    set_xticks(ax, x)
    ax.set_xlim(-0.55, N_CONDS - 0.45)
    bar_style(ax)

    handles = make_legend_handles(RECOVERY_METHODS)
    ax.legend(handles=handles,
              bbox_to_anchor=(0.5, 1.02), loc="lower center",
              ncol=4, fontsize=9, frameon=False,
              handlelength=1.4, handleheight=1.0, columnspacing=0.8)

    plt.tight_layout(pad=1.5)
    out = os.path.join(out_dir, "panel_c_recovery_rate.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Panel D — Trustworthiness Score (Equal vs Weighted)
# ---------------------------------------------------------------------------
def panel_d(df: pd.DataFrame, out_dir: str) -> None:
    d_methods = [
        "Rule-Based",
        "SAC+HER",
        "Rule-Based + Recovery",
        "SAC+HER + Recovery",
    ]
    n_methods = len(d_methods)
    x = np.arange(N_CONDS, dtype=float)

    bw      = 0.06    # width of each individual score bar
    within  = 0.02    # gap between Equal and Weighted bars within a method
    between = 0.10    # gap between methods within a condition group

    pair_w = 2 * bw + within           # total width of one method's bar pair
    stride = pair_w + between           # center-to-center distance between methods

    # Center offsets for each method relative to condition center
    method_offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * stride

    score_cols   = ["Trustworthiness Score (Equal)", "Trustworthiness Score (Weighted)"]
    score_colors = ["#5B9BD5", "#ED7D31"]
    score_labels = ["Equal (0.20×5)", "Weighted (TAIRO)"]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    for j, method in enumerate(d_methods):
        m_off = method_offsets[j]
        alpha = 0.62 + 0.28 * (j % 2)   # alternate shade per method pair
        for k, (col, color) in enumerate(zip(score_cols, score_colors)):
            vals = [get_val(df, method, c, col) for c in COND_SOURCE_ORDER]
            # bar k is offset from method center by ±(bw/2 + within/2)
            bar_off = (k - 0.5) * (bw + within)
            ax.bar(x + m_off + bar_off, vals,
                   width=bw * 0.92,
                   color=color,
                   alpha=alpha,
                   edgecolor="white",
                   linewidth=0.4,
                   zorder=3)

    # Method name labels below x-axis tick area
    short_name = {
        "Rule-Based":              "RB",
        "SAC+HER":                 "SAC+HER",
        "Rule-Based + Recovery":   "RB\n+Rec",
        "SAC+HER + Recovery":      "SAC+HER\n+Rec",
    }
    for j, method in enumerate(d_methods):
        m_off = method_offsets[j]
        for ci in range(N_CONDS):
            ax.text(x[ci] + m_off, -0.06, short_name[method],
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top",
                    fontsize=5.2, color="#444444")

    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.set_ylabel("Trustworthiness Score", fontsize=9, labelpad=8)
    ax.set_title("Panel D — Equal vs. Weighted Trustworthiness Score by Condition",
                 fontsize=11, fontweight="bold", pad=12, loc="left")

    set_xticks(ax, x)
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    bar_style(ax)

    score_handles = [
        mpatches.Patch(facecolor=c, alpha=0.8, label=l)
        for c, l in zip(score_colors, score_labels)
    ]
    ax.legend(handles=score_handles,
              bbox_to_anchor=(0.5, 1.02), loc="lower center",
              ncol=2, fontsize=9, frameon=False)

    plt.tight_layout(pad=1.5)
    out = os.path.join(out_dir, "panel_d_trustworthiness.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df = load_data(TABLE_PATH)
    panel_a(df, OUTPUTS_DIR)
    panel_b(df, OUTPUTS_DIR)
    panel_c(df, OUTPUTS_DIR)
    panel_d(df, OUTPUTS_DIR)


if __name__ == "__main__":
    main()
