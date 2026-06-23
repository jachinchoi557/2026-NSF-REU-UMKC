"""
Print a summary table of the recovery failure analysis sweep.

Imports shared logic from print_failure_summary.py and points it at the
recovery CSV so both tables have identical formatting and column definitions.

Reads:  tairo_results/outputs/failure_analysis_recovery_episodes.csv
Writes: tairo_results/outputs/failure_summary_recovery_table.csv

Supports optional CLI overrides for --input and --output paths.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OUTPUTS_DIR

# Import shared table logic from print_failure_summary without modifying it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from print_failure_summary import load_episodes, compute_summary, print_table, write_csv

DEFAULT_INPUT  = os.path.join(OUTPUTS_DIR, "failure_analysis_recovery_episodes.csv")
DEFAULT_OUTPUT = os.path.join(OUTPUTS_DIR, "failure_summary_recovery_table.csv")


def main():
    parser = argparse.ArgumentParser(description="Summarise recovery failure analysis sweep.")
    parser.add_argument("--input",  default=DEFAULT_INPUT,  help="Path to episodes CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path for summary CSV output")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found. Run scripts/run_failure_analysis_recovery.py first.")
        sys.exit(1)

    rows    = load_episodes(args.input)
    summary = compute_summary(rows)
    print_table(summary)
    write_csv(summary, args.output)


if __name__ == "__main__":
    main()
