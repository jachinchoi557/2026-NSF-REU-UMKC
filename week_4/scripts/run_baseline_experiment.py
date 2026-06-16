"""
CLI script — TAIRO FetchReach-v4 baseline experiment.

Runs random and rule-based policies across all attack conditions and seeds,
computes TAIRO C1–C5 trustworthiness scores, and writes three CSV files:

    tairo_results/tairo_fetchreach_episode_results.csv
    tairo_results/tairo_fetchreach_step_logs.csv
    tairo_results/tairo_summary_with_scores.csv

Usage
-----
    # Run with defaults (3 seeds, all conditions)
    python scripts/run_baseline_experiment.py

    # Quick smoke test with 1 seed
    python scripts/run_baseline_experiment.py --seeds 0

    # Include SAC+HER (requires trained model)
    python scripts/run_baseline_experiment.py --model-path tairo_results/sac_her_fetchreach_model

    # Limit to specific conditions
    python scripts/run_baseline_experiment.py --conditions clean sensor_noise
"""

import argparse
import os
import sys
from dataclasses import asdict
from typing import List, Optional

import numpy as np
import pandas as pd

# Ensure project root is on path when called from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GYM_AVAILABLE, RANDOM_SEEDS, RESULT_DIR, SB3_AVAILABLE
from envs.fetchreach_env import make_env
from evaluation.episode_runner import run_episode
from evaluation.metrics import add_trustworthiness_scores, summarize_results
from policies.sac_her_policy import SACHerPolicy

# ---------------------------------------------------------------------------
# All conditions used in the benchmark
# ---------------------------------------------------------------------------
ALL_CONDITIONS = [
    ("clean", 0.00),
    ("sensor_noise", 0.01),
    ("sensor_noise", 0.05),
    ("action_noise", 0.05),
    ("action_scale", 0.50),
    ("action_delay", 0.00),
    ("target_shift", 0.03),
]

CONDITION_NAMES = {name for name, _ in ALL_CONDITIONS}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TAIRO FetchReach-v4 baseline experiment."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=RANDOM_SEEDS,
        metavar="SEED",
        help="Random seeds to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        metavar="COND",
        choices=sorted(CONDITION_NAMES),
        help="Subset of conditions to run (default: all).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["random", "rule_based"],
        metavar="METHOD",
        help="Baseline methods to run (default: random rule_based).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        metavar="PATH",
        help="Path to a saved SAC+HER model to include in evaluation.",
    )
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="Disable recovery-aware variants for attack conditions.",
    )
    parser.add_argument(
        "--output-dir",
        default=RESULT_DIR,
        metavar="DIR",
        help="Directory for output CSVs (default: %(default)s).",
    )
    return parser.parse_args()


def load_model(model_path: str):
    """Load a saved SB3 SAC model. Returns None and prints a warning on failure."""
    if not SB3_AVAILABLE:
        print("WARNING: Stable-Baselines3 not available — skipping model load.")
        return None
    try:
        from stable_baselines3 import SAC
        # HerReplayBuffer requires an env at load time to reconstruct the buffer.
        env = make_env(seed=0)
        model = SAC.load(model_path, env=env)
        env.close()
        print(f"Loaded model from: {model_path}")
        return model
    except Exception as exc:
        print(f"WARNING: Could not load model from {model_path}: {exc}")
        return None


def select_conditions(requested: Optional[List[str]]) -> List:
    if requested is None:
        return ALL_CONDITIONS
    return [(name, level) for name, level in ALL_CONDITIONS if name in requested]


def run_benchmark(
    seeds: List[int],
    conditions: List,
    methods: List[str],
    model=None,
    include_recovery: bool = True,
    output_dir: str = RESULT_DIR,
) -> None:
    if not GYM_AVAILABLE:
        print(
            "ERROR: Gymnasium Robotics is not available.\n"
            "Install with: pip install gymnasium gymnasium-robotics mujoco"
        )
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    all_steps = []

    sac_policy = SACHerPolicy(model) if model is not None else None

    total_runs = len(seeds) * len(methods) * len(conditions)
    run_idx = 0

    for seed in seeds:
        env = make_env(seed=seed)

        for method in methods:
            for condition, attack_level in conditions:
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] seed={seed} method={method} "
                      f"condition={condition} attack_level={attack_level}")

                # Choose policy_fn for SAC-based methods
                policy_fn = sac_policy if (
                    method in {"sac", "sac_her"} and sac_policy is not None
                ) else None

                result, step_df = run_episode(
                    env=env,
                    method=method,
                    seed=seed,
                    condition=condition,
                    attack_level=attack_level,
                    model=model,
                    policy_fn=policy_fn,
                    use_recovery=False,
                )
                all_results.append(asdict(result))
                all_steps.append(step_df)

                # Recovery-aware variant for non-clean conditions
                if include_recovery and condition != "clean":
                    recovery_method = method + "_recovery"
                    result_r, step_df_r = run_episode(
                        env=env,
                        method=recovery_method,
                        seed=seed,
                        condition=condition,
                        attack_level=attack_level,
                        model=model,
                        policy_fn=policy_fn,
                        use_recovery=True,
                    )
                    all_results.append(asdict(result_r))
                    all_steps.append(step_df_r)

        env.close()

    episode_df = pd.DataFrame(all_results)
    step_df = pd.concat(all_steps, ignore_index=True)

    # Compute metrics
    summary_df = summarize_results(episode_df)
    score_df = add_trustworthiness_scores(summary_df)

    # --- Per-run timestamped files (never overwritten) -----------------------
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    unique_conds = list(dict.fromkeys(name for name, _ in conditions))  # ordered, deduped
    cond_slug = "-".join(unique_conds) if len(unique_conds) <= 3 else "all-conditions"
    seed_slug = "-".join(str(s) for s in seeds)
    run_stem  = f"run_{cond_slug}_seed{seed_slug}_{ts}"
    run_ep_path    = os.path.join(output_dir, f"{run_stem}_episode_results.csv")
    run_step_path  = os.path.join(output_dir, f"{run_stem}_step_logs.csv")
    run_score_path = os.path.join(output_dir, f"{run_stem}_summary.csv")

    episode_df.to_csv(run_ep_path,    index=False)
    step_df.to_csv(run_step_path,     index=False)
    score_df.to_csv(run_score_path,   index=False)

    # --- Running combined CSVs (append, dedupe on key columns) ---------------
    EPISODE_KEYS = ["method", "condition", "attack_level", "seed"]
    SUMMARY_KEYS = ["method", "condition", "attack_level"]

    combined_ep_path    = os.path.join(output_dir, "combined_episode_results.csv")
    combined_step_path  = os.path.join(output_dir, "combined_step_logs.csv")
    combined_score_path = os.path.join(output_dir, "combined_summary.csv")

    def _append_deduped(new_df: pd.DataFrame, path: str, keys: list) -> pd.DataFrame:
        if os.path.exists(path):
            existing = pd.read_csv(path)
            merged = pd.concat([existing, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=keys, keep="last")
        else:
            merged = new_df
        merged.to_csv(path, index=False)
        return merged

    combined_ep    = _append_deduped(episode_df, combined_ep_path,    EPISODE_KEYS)
    _append_deduped(step_df,    combined_step_path,  EPISODE_KEYS + ["timestep"])
    combined_score = _append_deduped(score_df,    combined_score_path, SUMMARY_KEYS)

    print(f"\nDone. Results saved to {output_dir}/")
    print(f"  Run files:      {run_ep_path}")
    print(f"                  {run_step_path}")
    print(f"                  {run_score_path}")
    print(f"  Combined files: {combined_ep_path} ({len(combined_ep)} episode rows total)")
    print(f"                  {combined_score_path} ({len(combined_score)} summary rows total)")

    # Print quick summary table
    display_cols = [
        "method", "condition", "attack_level",
        "success_rate", "final_distance", "trustworthiness_score",
    ]
    print("\n--- Trustworthiness Summary ---")
    print(score_df[display_cols].to_string(index=False))


def main():
    args = parse_args()

    model = None
    if args.model_path is not None:
        model = load_model(args.model_path)

    conditions = select_conditions(args.conditions)

    run_benchmark(
        seeds=args.seeds,
        conditions=conditions,
        methods=args.methods,
        model=model,
        include_recovery=not args.no_recovery,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
