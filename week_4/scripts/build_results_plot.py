"""
Generate week4_results_plot.png from week4_results_table.csv.

Three panels:
  A — Success Rate by method × condition
  B — Avg Reward by method × condition
  C — Recovery Triggered Rate (% steps) for _recovery variants only

action_delay is excluded throughout (consistent with the table).
SAC (no HER) is shown as a labelled ablation block, visually separated from
the main comparison group (Random / Rule-Based / SAC+HER) so that its 0%
success does not dominate the scale in Panels A and B.

Run:
    python scripts/build_results_plot.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from config import RESULT_DIR

TABLE_PATH = os.path.join(RESULT_DIR, "week4_results_table.csv")
OUT_PATH   = os.path.join(RESULT_DIR, "week4_results_plot.png")

# ── Condition ordering ─────────────────────────────────────────────────────────
COND_LABEL = {
    ("Clean",        0.00): "Clean",
    ("Sensor Noise", 0.01): "Sensor\nNoise 0.01",
    ("Sensor Noise", 0.05): "Sensor\nNoise 0.05",
    ("Action Noise", 0.05): "Action\nNoise 0.05",
    ("Action Scale", 0.50): "Action\nScale 0.5×",
    ("Target Shift", 0.03): "Target\nShift 0.03",
}
COND_ORDER = list(COND_LABEL.values())

# ── Method groups ──────────────────────────────────────────────────────────────
MAIN_METHODS  = ["Random", "Rule-Based", "SAC+HER"]
RECOV_METHODS = ["Random + Recovery", "Rule-Based + Recovery", "SAC+HER + Recovery"]
ABLATION      = "SAC (no HER)"

RECOV_PANEL_METHODS = ["Random + Recovery", "Rule-Based + Recovery", "SAC+HER + Recovery"]

# ── Colour palette ─────────────────────────────────────────────────────────────
COLORS = {
    "Random":                  "#4878CF",
    "Random + Recovery":       "#A8C4E8",
    "Rule-Based":              "#5CB85C",
    "Rule-Based + Recovery":   "#B0DDB0",
    "SAC+HER":                 "#D65F5F",
    "SAC+HER + Recovery":      "#F0AEAE",
    "SAC (no HER)":            "#888888",
    "SAC (no HER) + Recovery": "#BBBBBB",
}
HATCHES = {m: "//" if "Recovery" in m else "" for m in COLORS}
LEGEND_ORDER = MAIN_METHODS + RECOV_METHODS + [ABLATION]


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


def draw_grouped_bars(ax, df, metric, methods, cond_order,
                      ablation=None, width=0.11, gap=0.22):
    """
    Draw grouped bars for `methods` over `cond_order`.
    Optionally append a visually separated ablation column group.
    Returns the x-centre array for the main groups.
    """
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

    # ── ablation block ──────────────────────────────────────────────────────
    if ablation:
        abl_sub = df[df["Method"] == ablation]
        right_edge = x_centers[-1] + (n_m - 1) / 2 * width
        abl_start  = right_edge + gap * 3.5
        abl_xs = []
        for k, cond in enumerate(cond_order):
            if not (abl_sub["cond_label"] == cond).any():
                continue
            val = float(abl_sub.loc[abl_sub["cond_label"] == cond, metric].iloc[0])
            xp  = abl_start + k * (width * 1.5)
            ax.bar(xp, val,
                   width=width * 0.88,
                   color=COLORS[ablation],
                   edgecolor="white",
                   linewidth=0.5,
                   zorder=3)
            abl_xs.append(xp)

        # separator
        sep_x = right_edge + gap * 1.8
        ax.axvline(sep_x, color="#AAAAAA", linestyle=":", linewidth=1.0, zorder=2)

        # label below x-axis
        if abl_xs:
            ax.text(np.mean(abl_xs), -0.13, "SAC (no HER)\nablation",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.5,
                    color="#666666", fontstyle="italic")

        all_xs = list(x_centers) + abl_xs
        ax.set_xlim(all_xs[0] - gap * 0.8, all_xs[-1] + gap * 0.8)
    else:
        ax.set_xlim(x_centers[0] - gap * 0.8, x_centers[-1] + gap * 0.8)

    # x-tick labels for main groups only
    ax.set_xticks(x_centers)
    ax.set_xticklabels(cond_order, fontsize=8.5)
    bar_style(ax)
    return x_centers


def main():
    df = load_table(TABLE_PATH)

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#FAFAFA")
    gs = fig.add_gridspec(3, 1, hspace=0.62, top=0.92, bottom=0.05,
                          left=0.055, right=0.97,
                          height_ratios=[1.0, 1.0, 0.72])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    all_methods = MAIN_METHODS + RECOV_METHODS

    # ── Panel A: Success Rate ────────────────────────────────────────────────
    draw_grouped_bars(ax1, df, "Success Rate", all_methods, COND_ORDER,
                      ablation=ABLATION)
    ax1.set_ylim(0, 1.18)
    ax1.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax1.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax1.set_ylabel("Success Rate", fontsize=9, labelpad=7)
    ax1.set_title("Panel A — Success Rate by Method & Condition",
                  fontsize=10.5, fontweight="bold", pad=9, loc="left")

    # ── Panel B: Avg Reward ─────────────────────────────────────────────────
    draw_grouped_bars(ax2, df, "Avg Reward", all_methods, COND_ORDER,
                      ablation=ABLATION)
    ax2.set_ylim(-56, 4)
    ax2.set_yticks([-50, -40, -30, -20, -10, 0])
    ax2.set_yticklabels(["-50", "-40", "-30", "-20", "-10", "0"], fontsize=8)
    ax2.set_ylabel("Avg Reward", fontsize=9, labelpad=7)
    ax2.set_title("Panel B — Avg Reward by Method & Condition",
                  fontsize=10.5, fontweight="bold", pad=9, loc="left")

    # ── Panel C: Recovery Triggered Rate ────────────────────────────────────
    rec_conds = [c for c in COND_ORDER if c != "Clean"]
    n_rm  = len(RECOV_PANEL_METHODS)
    w3    = 0.20
    gap3  = 0.40
    x3c   = np.arange(len(rec_conds)) * (n_rm * w3 + gap3)

    for j, method in enumerate(RECOV_PANEL_METHODS):
        sub = df[df["Method"] == method]
        vals = []
        for cond in rec_conds:
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
    ax3.set_xticklabels(rec_conds, fontsize=8.5)
    ax3.set_ylim(0, 115)
    ax3.set_yticks([0, 25, 50, 75, 100])
    ax3.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax3.set_ylabel("Recovery Triggered\nRate (% of steps)", fontsize=8.5, labelpad=7)
    ax3.set_title("Panel C — Recovery Triggered Rate  (recovery variants only, no clean baseline run)",
                  fontsize=10.5, fontweight="bold", pad=9, loc="left")
    ax3.set_xlim(x3c[0] - gap3 * 0.8, x3c[-1] + gap3 * 0.8)
    bar_style(ax3)

    # ── Legend ───────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=COLORS[m], hatch=HATCHES[m],
                       edgecolor="#666", linewidth=0.4, label=m)
        for m in LEGEND_ORDER
    ]
    fig.legend(handles=handles,
               loc="upper center",
               ncol=len(LEGEND_ORDER),
               fontsize=7.8,
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
