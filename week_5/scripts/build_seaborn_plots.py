"""
Generate four individual panel PNGs using seaborn, from week5_results_table.csv.

Output files (all to tairo_results/outputs/):
    panel_a_success_rate.png
    panel_b_avg_reward.png
    panel_c_recovery_rate.png
    panel_d_trustworthiness.png

Run:
    python scripts/build_seaborn_plots.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import OUTPUTS_DIR

TABLE_PATH = os.path.join(OUTPUTS_DIR, "week5_results_table.csv")

# ---------------------------------------------------------------------------
# Condition ordering and display names
# ---------------------------------------------------------------------------
# The CSV uses "Action Clipping" and "Sensor Bias"; rename to match desired labels.
COND_RENAME = {
    "Action Clipping": "Action Clip 0.30",
    "Sensor Bias":     "Sensor Bias 0.10",
}

COND_ORDER = [
    "Clean",
    "Action Delay",
    "Action Reverse",
    "Action Clip 0.30",
    "Sensor Dropout",
    "Sensor Bias 0.10",
    "Goal Spoof (t=0)",
    "Goal Spoof (t=20)",
]

LABEL_MAP = {
    "Clean":            "Clean",
    "Action Delay":     "Action\nDelay",
    "Action Reverse":   "Action\nReverse",
    "Action Clip 0.30": "Action\nClip 0.30",
    "Sensor Dropout":   "Sensor\nDropout",
    "Sensor Bias 0.10": "Sensor\nBias 0.10",
    "Goal Spoof (t=0)": "Goal Spoof\n(t=0)",
    "Goal Spoof (t=20)":"Goal Spoof\n(t=20)",
}

# ---------------------------------------------------------------------------
# Method order and palette
# ---------------------------------------------------------------------------
FOCUS_METHODS = [
    "SAC+HER",
    "SAC+HER + Recovery",
]

METHOD_ORDER = [
    "SAC+HER",
    "SAC+HER + Recovery",
]

BASE_PALETTE = {
    "SAC+HER": "#E05C2A",
}


def lighten(hex_color: str, factor: float = 0.5) -> tuple:
    rgb = mc.to_rgb(hex_color)
    return tuple(c + (1 - c) * factor for c in rgb)


RECOVERY_PALETTE = {k: lighten(v) for k, v in BASE_PALETTE.items()}

METHOD_PALETTE: dict = {}
for _m, _c in BASE_PALETTE.items():
    METHOD_PALETTE[_m] = _c
    METHOD_PALETTE[_m + " + Recovery"] = RECOVERY_PALETTE[_m]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Method"].isin(FOCUS_METHODS)].copy()
    df["Condition"] = df["Condition"].replace(COND_RENAME)
    df["is_recovery"] = df["Method"].str.contains("Recovery")
    df["base_method"] = df["Method"].str.replace(" + Recovery", "", regex=False).str.strip()
    df["Condition"] = pd.Categorical(df["Condition"], categories=COND_ORDER, ordered=True)
    df["Method"] = pd.Categorical(df["Method"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values("Condition")
    return df


# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

def set_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams.update({
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.grid.axis":    "y",
        "grid.alpha":        0.4,
        "font.family":       "sans-serif",
    })


def fix_xlabels(ax, rotation: int = 0) -> None:
    """Replace condition names with two-line versions; suppress x-axis label."""
    ax.figure.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels()]
    # FixedLocator required before set_xticklabels to avoid matplotlib warning
    ax.xaxis.set_major_locator(plt.FixedLocator(ax.get_xticks()))
    ax.set_xticklabels(
        [LABEL_MAP.get(t, t) for t in labels],
        rotation=rotation, ha="center", fontsize=9,
    )
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("")


def bar_style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_visible(False)


# ---------------------------------------------------------------------------
# Panel A — Success Rate
# ---------------------------------------------------------------------------

def panel_a(df: pd.DataFrame, out_dir: str) -> None:
    set_style()

    fig, ax = plt.subplots(figsize=(14, 6))

    sns.barplot(
        data=df,
        x="Condition", y="Success Rate",
        hue="Method", hue_order=METHOD_ORDER,
        palette=METHOD_PALETTE,
        order=COND_ORDER,
        errorbar=None,
        ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylabel("Success Rate", fontsize=10)
    fix_xlabels(ax)
    bar_style(ax)

    if ax.get_legend():
        ax.get_legend().remove()
    ax.legend(
        handles=[mpatches.Patch(color=METHOD_PALETTE[m], label=m) for m in METHOD_ORDER],
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=False, fontsize=9,
    )
    fig.suptitle("Panel A — Success Rate by Method & Condition", fontsize=12, y=1.08)

    plt.savefig(os.path.join(out_dir, "panel_a_success_rate.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: panel_a_success_rate.png")


# ---------------------------------------------------------------------------
# Panel B — Avg Reward
# ---------------------------------------------------------------------------

def panel_b(df: pd.DataFrame, out_dir: str) -> None:
    set_style()

    data_min = df["Avg Reward"].min()

    fig, ax = plt.subplots(figsize=(14, 6))

    sns.barplot(
        data=df,
        x="Condition", y="Avg Reward",
        hue="Method", hue_order=METHOD_ORDER,
        palette=METHOD_PALETTE,
        order=COND_ORDER,
        errorbar=None,
        ax=ax,
    )
    ax.set_ylim(bottom=data_min - 2)
    ax.set_ylabel("Avg Reward", fontsize=10)
    fix_xlabels(ax)
    bar_style(ax)

    if ax.get_legend():
        ax.get_legend().remove()
    ax.legend(
        handles=[mpatches.Patch(color=METHOD_PALETTE[m], label=m) for m in METHOD_ORDER],
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=False, fontsize=9,
    )
    fig.suptitle("Panel B — Avg Reward by Method & Condition", fontsize=12, y=1.08)

    plt.savefig(os.path.join(out_dir, "panel_b_avg_reward.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: panel_b_avg_reward.png")


# ---------------------------------------------------------------------------
# Panel C — Recovery Triggered Rate
# ---------------------------------------------------------------------------

def panel_c(df: pd.DataFrame, out_dir: str) -> None:
    set_style()

    rec_hue_order = ["SAC+HER + Recovery"]
    rec_df = df[
        df["is_recovery"] & df["Recovery Triggered Rate"].notna()
    ].copy()

    fig, ax = plt.subplots(figsize=(14, 5))

    sns.barplot(
        data=rec_df,
        x="Condition", y="Recovery Triggered Rate",
        hue="Method", hue_order=rec_hue_order,
        palette={m: METHOD_PALETTE[m] for m in rec_hue_order},
        order=COND_ORDER,
        errorbar=None,
        ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylabel("Recovery Triggered Rate\n(% of steps)", fontsize=9)
    ax.set_title("Panel C — Recovery Triggered Rate (recovery variants only)",
                 fontsize=12, pad=32, loc="left")
    fix_xlabels(ax)
    bar_style(ax)

    if ax.get_legend():
        ax.get_legend().remove()

    handles = [mpatches.Patch(color=METHOD_PALETTE[m], label=m) for m in rec_hue_order]
    ax.legend(
        handles=handles,
        loc="lower center", bbox_to_anchor=(0.5, 1.06),
        ncol=1, frameon=False, fontsize=8.5,
    )

    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(out_dir, "panel_c_recovery_rate.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: panel_c_recovery_rate.png")


# ---------------------------------------------------------------------------
# Panel D — Trustworthiness Scores (FacetGrid)
# ---------------------------------------------------------------------------

def panel_d(df: pd.DataFrame, out_dir: str) -> None:
    set_style()

    d_methods = ["SAC+HER", "SAC+HER + Recovery"]

    trust_df = df[df["Method"].isin(d_methods)].copy()
    trust_long = trust_df.melt(
        id_vars=["Method", "Condition"],
        value_vars=[
            "Trustworthiness Score (Equal)",
            "Trustworthiness Score (Weighted)",
        ],
        var_name="Score Type",
        value_name="Score",
    )
    trust_long["Score Type"] = trust_long["Score Type"].map({
        "Trustworthiness Score (Equal)":    "Equal (0.20×5)",
        "Trustworthiness Score (Weighted)": "Weighted (TAIRO)",
    })
    # Re-apply categorical ordering after melt
    trust_long["Condition"] = pd.Categorical(
        trust_long["Condition"], categories=COND_ORDER, ordered=True
    )
    trust_long["Method"] = pd.Categorical(
        trust_long["Method"].astype(str), categories=d_methods, ordered=True
    )

    score_palette = {"Equal (0.20×5)": "#4472C4", "Weighted (TAIRO)": "#ED7D31"}

    g = sns.FacetGrid(
        trust_long,
        col="Method",
        col_order=["SAC+HER", "SAC+HER + Recovery"],
        height=4.5, aspect=1.0,
        sharey=True,
    )
    g.map_dataframe(
        sns.barplot,
        x="Condition", y="Score",
        hue="Score Type",
        hue_order=["Equal (0.20×5)", "Weighted (TAIRO)"],
        palette=score_palette,
        order=COND_ORDER,
        errorbar=None,
    )
    g.set_titles(col_template="{col_name}", size=10)
    g.set_axis_labels("", "Trustworthiness Score")
    g.set(ylim=(0, 1.05))

    # Fix tick labels on each facet after drawing
    g.figure.canvas.draw()
    for ax in g.axes.flat:
        labels = [t.get_text() for t in ax.get_xticklabels()]
        ax.xaxis.set_major_locator(plt.FixedLocator(ax.get_xticks()))
        ax.set_xticklabels(
            [LABEL_MAP.get(t, t) for t in labels],
            rotation=45, ha="right", fontsize=7,
        )
        ax.tick_params(axis="x", length=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_visible(False)

    g.add_legend(
        title="Score Type",
        bbox_to_anchor=(1.0, 0.5),
        loc="center right",
        frameon=False,
        fontsize=9,
    )
    g.figure.suptitle(
        "Panel D — Equal vs. Weighted Trustworthiness Score",
        y=1.02, fontsize=12,
    )

    g.savefig(
        os.path.join(out_dir, "panel_d_trustworthiness.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print("Saved: panel_d_trustworthiness.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df = load_data(TABLE_PATH)
    panel_a(df, OUTPUTS_DIR)
    panel_b(df, OUTPUTS_DIR)
    panel_c(df, OUTPUTS_DIR)
    panel_d(df, OUTPUTS_DIR)


if __name__ == "__main__":
    main()
