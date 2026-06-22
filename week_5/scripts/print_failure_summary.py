"""
Print a summary table of the failure analysis sweep to stdout and write
tairo_results/outputs/failure_summary_table.csv.

Reads: tairo_results/outputs/failure_analysis_episodes.csv
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OUTPUTS_DIR

EPISODE_CSV  = os.path.join(OUTPUTS_DIR, "failure_analysis_episodes.csv")
SUMMARY_CSV  = os.path.join(OUTPUTS_DIR, "failure_summary_table.csv")

CONDITION_ORDER = [
    "clean",
    "sensor_dropout",
    "sensor_bias",
    "goal_spoof_immediate",
    "goal_spoof_midep",
    "action_delay",
    "action_reverse",
    "action_clipping",
]

SUMMARY_FIELDS = [
    "condition",
    "success_rate",
    "avg_reward",
    "avg_final_distance",
    "recovery_pct",
    "avg_first_recovery_step",
]


def load_episodes(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "condition":               row["condition"],
                "success":                 int(row["success"]),
                "total_reward":            float(row["total_reward"]),
                "final_distance":          float(row["final_distance"]),
                "recovery_triggered_count": int(row["recovery_triggered_count"]),
                "first_recovery_step":     int(row["first_recovery_step"]),
            })
    return rows


def compute_summary(rows):
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)

    summary = []
    for cond in CONDITION_ORDER:
        eps = by_cond.get(cond, [])
        if not eps:
            continue
        n = len(eps)

        success_rate   = 100.0 * sum(e["success"] for e in eps) / n
        avg_reward     = sum(e["total_reward"] for e in eps) / n
        avg_final_dist = sum(e["final_distance"] for e in eps) / n
        recovery_pct   = 100.0 * sum(1 for e in eps if e["recovery_triggered_count"] > 0) / n

        fired = [e["first_recovery_step"] for e in eps if e["first_recovery_step"] >= 0]
        avg_first_recovery = (sum(fired) / len(fired)) if fired else None

        summary.append({
            "condition":               cond,
            "success_rate":            success_rate,
            "avg_reward":              avg_reward,
            "avg_final_distance":      avg_final_dist,
            "recovery_pct":            recovery_pct,
            "avg_first_recovery_step": avg_first_recovery,
        })
    return summary


def fmt_pct(v):
    return f"{v:.1f}%"


def fmt_f(v, decimals=2):
    return f"{v:.{decimals}f}"


def fmt_step(v):
    return f"{v:.1f}" if v is not None else "N/A"


def print_table(summary):
    col_cond  = max(len(r["condition"]) for r in summary)
    col_cond  = max(col_cond, len("Condition"))

    header = (
        f"{'Condition':<{col_cond}} │ "
        f"{'Success Rate':>12} │ "
        f"{'Avg Reward':>10} │ "
        f"{'Avg Dist':>8} │ "
        f"{'Recovery (%)':>12} │ "
        f"{'Avg 1st Rec Step':>16}"
    )
    sep_inner = (
        f"{'─' * col_cond}─┼─"
        f"{'─' * 12}─┼─"
        f"{'─' * 10}─┼─"
        f"{'─' * 8}─┼─"
        f"{'─' * 12}─┼─"
        f"{'─' * 16}"
    )
    top = (
        f"┌{'─' * col_cond}─┬─"
        f"{'─' * 12}─┬─"
        f"{'─' * 10}─┬─"
        f"{'─' * 8}─┬─"
        f"{'─' * 12}─┬─"
        f"{'─' * 16}┐"
    )
    bot = (
        f"└{'─' * col_cond}─┴─"
        f"{'─' * 12}─┴─"
        f"{'─' * 10}─┴─"
        f"{'─' * 8}─┴─"
        f"{'─' * 12}─┴─"
        f"{'─' * 16}┘"
    )

    print(top)
    print(f"│ {header} │")
    print(f"├{sep_inner}┤")

    for r in summary:
        row = (
            f"{r['condition']:<{col_cond}} │ "
            f"{fmt_pct(r['success_rate']):>12} │ "
            f"{fmt_f(r['avg_reward']):>10} │ "
            f"{fmt_f(r['avg_final_distance']):>8} │ "
            f"{fmt_pct(r['recovery_pct']):>12} │ "
            f"{fmt_step(r['avg_first_recovery_step']):>16}"
        )
        print(f"│ {row} │")

    print(bot)


def write_csv(summary, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for r in summary:
            writer.writerow({
                "condition":               r["condition"],
                "success_rate":            f"{r['success_rate']:.1f}",
                "avg_reward":              f"{r['avg_reward']:.2f}",
                "avg_final_distance":      f"{r['avg_final_distance']:.4f}",
                "recovery_pct":            f"{r['recovery_pct']:.1f}",
                "avg_first_recovery_step": fmt_step(r["avg_first_recovery_step"]),
            })
    print(f"\nWrote {path}")


def main():
    if not os.path.exists(EPISODE_CSV):
        print(f"ERROR: {EPISODE_CSV} not found. Run scripts/run_failure_analysis.py first.")
        sys.exit(1)

    rows    = load_episodes(EPISODE_CSV)
    summary = compute_summary(rows)
    print_table(summary)
    write_csv(summary, SUMMARY_CSV)


if __name__ == "__main__":
    main()
