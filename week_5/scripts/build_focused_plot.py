"""
Generate week5_focused_plot.png — SAC variants only, with Rule-Based reference.

Three panels:
  A — Success Rate by method × condition (SAC+HER, SAC+HER+Recovery, SAC no HER, SAC no HER+Recovery)
  B — Avg Reward by method × condition
  C — Recovery Triggered Rate (% steps) for _recovery variants only

A narrow reference panel at the right of A and B shows Rule-Based success rate
and avg reward per condition without giving it equal visual weight.

Run:
    python scripts/build_focused_plot.py
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
OUT_PATH   = os.path.join(OUTPUTS_DIR, "week5_focused_plot.png")

# ── Condition display order ────────────────────────────────────────────────────
COND_LABEL = {
    ("Clean",             0.00): "Clean",
    ("Action Delay",      0.00): "Act\nDelay",
    ("Action Reverse",    0.00): "Act\nReverse",
    ("Action Clipping",   0.30): "Act\nClip",
    ("Sensor Dropout",    0.00): "Sensor\nDropout",
    ("Sensor Bias",       0.10): "Sensor\nBias",
    ("Goal Spoof (t=0)",  0.10): "Goal\nSpoof t=0",
    ("Goal Spoof (t=20)", 0.10): "Goal\nSpoof t=20",
}
COND_ORDER = list(COND_LABEL.values())

# ── Method groups ──────────────────────────────────────────────────────────────
FOCUS_METHODS       = ["SAC+HER", "SAC+HER + Recovery", "SAC (no HER)", "SAC (no HER) + Recovery"]
FOCUS_RECOV_METHODS = ["SAC+HER + Recovery", "SAC (no HER) + Recovery"]
REFERENCE_METHOD    = "Rule-Based"

# ── Colour palette ─────────────────────────────────────────────────────────────
COLORS = {
    "SAC+HER":                  "#D65F5F",
    "SAC+HER + Recovery":       "#F0AEAE",
    "SAC (no HER)":             "#888888",
    "SAC (no HER) + Recovery":  "#BBBBBB",
    REFERENCE_METHOD:           "#5CB85C",
}
HATCHES = {m: "//" if "Recovery" in m else "" for m in COLORS}


def load_table(path):
    df = pd.read_csv(path)
    df["cond_label"] = [
        COND_LABEL.get((row["Condition"], row["Attack Level"]), row["Condition"])
        for _, row in df.iterrows()
    ]
    return df


def bar_style(ax):
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)


def draw_focus_bars(ax, df, metric, methods, cond_order,
                    ref_method=None, width=0.13, gap=0.28):
    """Draw grouped bars for focus methods, with an optional slim reference column."""
    n_m = len(methods)
    x_centers = np.arange(len(cond_order)) * (n_m * width + gap)

    for j, method in enumerate(methods):
        sub = df[df["Method"] == method]
        vals = [
            float(sub.loc[sub["cond_label"] == c, metric].iloc[0])
            if (sub["cond_label"] == c).any() else np.nan
            for c in cond_order
        ]
        offsets = (np.arange(n_m) - (n_m - 1) / 2) * width
        ax.bar(x_centers + offsets[j], vals,
               width=width * 0.88,
               color=COLORS[method],
               hatch=HATCHES[method],
               edgecolor="white",
               linewidth=0.5,
               zorder=3)

    if ref_method:
        ref_sub = df[df["Method"] == ref_method]
        right_edge = x_centers[-1] + (n_m - 1) / 2 * width
        ref_start  = right_edge + gap * 3.2
        ref_xs = []
        ref_w  = width * 0.7
        for k, cond in enumerate(cond_order):
            if not (ref_sub["cond_label"] == cond).any():
                continue
            val = float(ref_sub.loc[ref_sub["cond_label"] == cond, metric].iloc[0])
            xp  = ref_start + k * (ref_w * 1.6)
            ax.bar(xp, val,
                   width=ref_w,
                   color=COLORS[ref_method],
                   alpha=0.65,
                   edgecolor="white",
                   linewidth=0.5,
                   zorder=3)
            ref_xs.append(xp)

        sep_x = right_edge + gap * 1.6
        ax.axvline(sep_x, color="#AAAAAA", linestyle=":", linewidth=1.0, zorder=2)
        if ref_xs:
            ax.text(np.mean(ref_xs), -0.13, "Rule-Based\nreference",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.5,
                    color="#444444", fontstyle="italic")

        all_xs = list(x_centers) + ref_xs
        ax.set_xlim(all_xs[0] - gap * 0.8, all_xs[-1] + gap * 0.8)
    else:
        ax.set_xlim(x_centers[0] - gap * 0.8, x_centers[-1] + gap * 0.8)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(cond_order, fontsize=8)
    bar_style(ax)
    return x_centers


def main():
    df = load_table(TABLE_PATH)

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#FAFAFA")
    gs = fig.add_gridspec(3, 1, hspace=0.70, top=0.92, bottom=0.05,
                          left=0.055, right=0.97,
                          height_ratios=[1.0, 1.0, 0.72])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ── Panel A: Success Rate ──────────────────────────────────────────────────
    draw_focus_bars(ax1, df, "Success Rate", FOCUS_METHODS, COND_ORDER,
                    ref_method=REFERENCE_METHOD)
    ax1.set_ylim(0, 1.18)
    ax1.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax1.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax1.set_ylabel("Success Rate", fontsize=9, labelpad=7)
    ax1.set_title("Panel A — Success Rate: SAC Variants (Rule-Based reference at right)",
                  fontsize=10.5, fontweight="bold", pad=9, loc="left")

    # ── Panel B: Avg Reward ────────────────────────────────────────────────────
    draw_focus_bars(ax2, df, "Avg Reward", FOCUS_METHODS, COND_ORDER,
                    ref_method=REFERENCE_METHOD)
    ax2.set_ylim(-56, 4)
    ax2.set_yticks([-50, -40, -30, -20, -10, 0])
    ax2.set_yticklabels(["-50", "-40", "-30", "-20", "-10", "0"], fontsize=8)
    ax2.set_ylabel("Avg Reward", fontsize=9, labelpad=7)
    ax2.set_title("Panel B — Avg Reward: SAC Variants (Rule-Based reference at right)",
                  fontsize=10.5, fontweight="bold", pad=9, loc="left")

    # ── Panel C: Recovery Triggered Rate ──────────────────────────────────────
    n_rm = len(FOCUS_RECOV_METHODS)
    w3   = 0.22
    gap3 = 0.50
    x3c  = np.arange(len(COND_ORDER)) * (n_rm * w3 + gap3)

    for j, method in enumerate(FOCUS_RECOV_METHODS):
        sub = df[df["Method"] == method]
        vals = []
        for cond in COND_ORDER:
            row = sub[sub["cond_label"] == cond]
            if len(row) and not pd.isna(row["Recovery Triggered Rate"].iloc[0]):
                vals.append(float(row["Recovery Triggered Rate"].iloc[0]) * 100)
            else:
                vals.append(0.0)
        offsets = (np.arange(n_rm) - (n_rm - 1) / 2) * w3
        ax3.bar(x3c + offsets[j], vals,
                width=w3 * 0.88,
                color=COLORS[method],
                hatch=HATCHES[method],
                edgecolor="white",
                linewidth=0.5,
                zorder=3)

    ax3.set_xticks(x3c)
    ax3.set_xticklabels(COND_ORDER, fontsize=8)
    ax3.set_ylim(0, 115)
    ax3.set_yticks([0, 25, 50, 75, 100])
    ax3.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax3.set_ylabel("Recovery Triggered\nRate (% of steps)", fontsize=8.5, labelpad=7)
    ax3.set_title("Panel C — Recovery Triggered Rate  (SAC recovery variants only)",
                  fontsize=10.5, fontweight="bold", pad=9, loc="left")
    ax3.set_xlim(x3c[0] - gap3 * 0.8, x3c[-1] + gap3 * 0.8)
    bar_style(ax3)

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_order = FOCUS_METHODS + [REFERENCE_METHOD]
    handles = [
        mpatches.Patch(facecolor=COLORS[m], hatch=HATCHES.get(m, ""),
                       edgecolor="#666", linewidth=0.4, label=m)
        for m in legend_order
    ]
    fig.legend(handles=handles,
               loc="upper center",
               ncol=len(legend_order),
               fontsize=8,
               framealpha=0.92,
               edgecolor="#CCCCCC",
               bbox_to_anchor=(0.5, 0.998),
               handlelength=1.5,
               handleheight=1.0,
               columnspacing=0.75)

    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
