"""
Build week5_results_table.csv from combined_summary.csv and combined_step_logs.csv.

Scope
-----
Methods    : random, rule_based, sac_her, sac_plain  (and _recovery counterparts)
Conditions : sensor_dropout, sensor_bias,
             goal_spoof_immediate, goal_spoof_midep  (Week 5 new attacks)

Columns in output
-----------------
Method, Condition, Attack Level, Success Rate, Avg Reward, Final Distance,
Recovery Triggered Rate, Trustworthiness Score (Equal), Trustworthiness Score (Weighted)

Recovery Triggered Rate : fraction of steps where recovery_triggered == 1
for _recovery method variants; NaN for base variants (they never call recovery).

Run
---
    python scripts/build_results_table.py
    # Optional overrides:
    python scripts/build_results_table.py \\
        --summary  tairo_results/canonical/combined_summary.csv \\
        --steplogs tairo_results/canonical/combined_step_logs.csv \\
        --output   tairo_results/outputs/week5_results_table.csv
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CANONICAL_DIR, OUTPUTS_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INCLUDE_METHODS = [
    "random",              "random_recovery",
    "rule_based",          "rule_based_recovery",
    "sac_plain",           "sac_plain_recovery",
    "sac_her",             "sac_her_recovery",
]

# Week 5 new attack conditions
INCLUDE_CONDITIONS = {
    "sensor_dropout",
    "sensor_bias",
    "goal_spoof_immediate",
    "goal_spoof_midep",
}

PRETTY_METHOD = {
    "random":               "Random",
    "random_recovery":      "Random + Recovery",
    "rule_based":           "Rule-Based",
    "rule_based_recovery":  "Rule-Based + Recovery",
    "sac_plain":            "SAC (no HER)",
    "sac_plain_recovery":   "SAC (no HER) + Recovery",
    "sac_her":              "SAC+HER",
    "sac_her_recovery":     "SAC+HER + Recovery",
}

PRETTY_CONDITION = {
    "sensor_dropout":      "Sensor Dropout",
    "sensor_bias":         "Sensor Bias",
    "goal_spoof_immediate": "Goal Spoof (t=0)",
    "goal_spoof_midep":    "Goal Spoof (t=20)",
}


def parse_args():
    p = argparse.ArgumentParser(description="Regenerate week5_results_table.csv")
    p.add_argument("--summary",  default=os.path.join(CANONICAL_DIR, "combined_summary.csv"))
    p.add_argument("--steplogs", default=os.path.join(CANONICAL_DIR, "combined_step_logs.csv"))
    p.add_argument("--output",   default=os.path.join(OUTPUTS_DIR,   "week5_results_table.csv"))
    return p.parse_args()


def compute_recovery_rates(step_df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of steps with recovery_triggered==1 per method/condition/attack_level."""
    rec_methods = [m for m in INCLUDE_METHODS if m.endswith("_recovery")]
    sub = step_df[
        step_df["method"].isin(rec_methods)
        & step_df["condition"].isin(INCLUDE_CONDITIONS)
    ]
    rate = (
        sub.groupby(["method", "condition", "attack_level"])["recovery_triggered"]
        .mean()
        .reset_index()
        .rename(columns={"recovery_triggered": "recovery_triggered_rate"})
    )
    return rate


def build_table(summary_path: str, steplogs_path: str, output_path: str) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    step_df = pd.read_csv(steplogs_path)

    # Filter to in-scope methods and conditions
    df = summary[
        summary["method"].isin(INCLUDE_METHODS)
        & summary["condition"].isin(INCLUDE_CONDITIONS)
    ].copy()

    # Merge per-step recovery rates for _recovery variants
    rec_rates = compute_recovery_rates(step_df)
    df = df.merge(rec_rates, on=["method", "condition", "attack_level"], how="left")

    # Base methods have no recovery path — NaN is correct; leave it
    base_methods = [m for m in INCLUDE_METHODS if not m.endswith("_recovery")]
    df.loc[df["method"].isin(base_methods), "recovery_triggered_rate"] = float("nan")

    # Build output columns
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = pd.DataFrame({
        "Method":                          df["method"].map(PRETTY_METHOD),
        "Condition":                       df["condition"].map(PRETTY_CONDITION),
        "Attack Level":                    df["attack_level"],
        "Success Rate":                    df["success_rate"].round(3),
        "Avg Reward":                      df["avg_reward"].round(2),
        "Final Distance":                  df["final_distance"].round(4),
        "Recovery Triggered Rate":         df["recovery_triggered_rate"].round(3),
        "Trustworthiness Score (Equal)":   df["trustworthiness_score_equal"].round(4),
        "Trustworthiness Score (Weighted)": df["trustworthiness_score_weighted"].round(4),
    })

    # Sort: method order preserved, then condition alphabetically, then attack level
    method_order = {m: i for i, m in enumerate(INCLUDE_METHODS)}
    out["_method_rank"] = df["method"].map(method_order).values
    out = out.sort_values(["_method_rank", "Condition", "Attack Level"]).drop(columns="_method_rank")
    out = out.reset_index(drop=True)

    out.to_csv(output_path, index=False)
    return out


def main():
    args = parse_args()
    table = build_table(args.summary, args.steplogs, args.output)

    print(f"Wrote {len(table)} rows to {args.output}\n")
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(table.to_string(index=True))


if __name__ == "__main__":
    main()
