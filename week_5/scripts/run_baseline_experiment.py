"""
CLI script — TAIRO FetchReach-v4 baseline experiment.

Runs policies across all attack conditions and seeds, computes TAIRO C1–C5
trustworthiness scores, and writes CSV files:

    tairo_results/experiments/<cond_slug>/   — timestamped per-run snapshots
    tairo_results/canonical/                 — combined_*.csv (append, dedupe)

Usage
-----
    # Run with defaults (3 seeds, all conditions, random + rule_based)
    python scripts/run_baseline_experiment.py

    # Include SAC+HER (requires trained model)
    python scripts/run_baseline_experiment.py \\
        --methods random rule_based sac_her sac_plain \\
        --model-path tairo_results/models/sac_her_fetchreach_model

    # Week 5 new-attack sweep
    python scripts/run_baseline_experiment.py \\
        --conditions sensor_dropout sensor_bias goal_spoof_immediate goal_spoof_midep \\
        --methods random rule_based sac_her sac_plain \\
        --model-path tairo_results/models/sac_her_fetchreach_model

    # Quick smoke test with 1 seed
    python scripts/run_baseline_experiment.py --seeds 0
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

from config import (
    CANONICAL_DIR,
    EXPERIMENTS_DIR,
    GYM_AVAILABLE,
    RANDOM_SEEDS,
    RESULT_DIR,
    SB3_AVAILABLE,
)
from envs.fetchreach_env import make_env
from evaluation.episode_runner import run_episode
from evaluation.metrics import add_trustworthiness_scores, summarize_results
from policies.sac_her_policy import SACHerPolicy

# ---------------------------------------------------------------------------
# All conditions used in the benchmark
# ---------------------------------------------------------------------------
ALL_CONDITIONS = [
    # Week 4 baseline conditions
    ("clean",               0.00),
    ("sensor_noise",        0.01),
    ("sensor_noise",        0.05),
    ("action_noise",        0.05),
    ("action_scale",        0.50),
    ("action_delay",        0.00),
    ("action_reverse",      0.00),
    ("action_clipping",     0.30),
    ("target_shift",        0.03),
    # Week 5 new attack conditions
    ("sensor_dropout",      0.00),   # zeros obs["observation"]; magnitude unused
    ("sensor_bias",         0.10),   # constant offset on obs["observation"]
    ("goal_spoof_immediate", 0.10),  # goal shift from step 0
    ("goal_spoof_midep",    0.10),   # goal shift from step 20 of 50
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
        "--n-episodes",
        type=int,
        default=1,
        metavar="N",
        help="Number of episodes to run per (seed, method, condition) (default: 1).",
    )
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="Disable recovery-aware variants for attack conditions.",
    )
    parser.add_argument(
        "--include-clean-recovery",
        action="store_true",
        help="Also run _recovery method variants on the clean condition.",
    )
    parser.add_argument(
        "--output-dir",
        default=EXPERIMENTS_DIR,
        metavar="DIR",
        help=(
            "Root directory for timestamped per-run CSVs. "
            "Each condition slug gets a subdirectory. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--canonical-dir",
        default=CANONICAL_DIR,
        metavar="DIR",
        help="Directory for combined_*.csv canonical files (default: %(default)s).",
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
    include_clean_recovery: bool = False,
    n_episodes: int = 1,
    output_dir: str = EXPERIMENTS_DIR,
    canonical_dir: str = CANONICAL_DIR,
) -> None:
    if not GYM_AVAILABLE:
        print(
            "ERROR: Gymnasium Robotics is not available.\n"
            "Install with: pip install gymnasium gymnasium-robotics mujoco"
        )
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(canonical_dir, exist_ok=True)

    all_results = []
    all_steps = []

    sac_policy = SACHerPolicy(model) if model is not None else None

    total_runs = len(seeds) * len(methods) * len(conditions) * n_episodes
    run_idx = 0

    for seed in seeds:
        env = make_env(seed=seed)

        for method in methods:
            for condition, attack_level in conditions:
                for ep_idx in range(n_episodes):
                    run_idx += 1
                    # Each episode within a seed uses a distinct reset seed
                    episode_seed = seed * 100 + ep_idx
                    print(f"[{run_idx}/{total_runs}] seed={seed} ep={ep_idx} "
                          f"method={method} condition={condition} attack_level={attack_level}")

                    # Choose policy_fn for SAC-based methods
                    policy_fn = sac_policy if (
                        method in {"sac", "sac_her", "sac_plain"} and sac_policy is not None
                    ) else None

                    result, step_df = run_episode(
                        env=env,
                        method=method,
                        seed=episode_seed,   # env reset seed — unique per episode
                        condition=condition,
                        attack_level=attack_level,
                        model=model,
                        policy_fn=policy_fn,
                        use_recovery=False,
                    )
                    row = asdict(result)
                    row["seed"] = seed          # overwrite reset seed with seed group label
                    row["episode_idx"] = ep_idx
                    all_results.append(row)
                    step_df["seed"] = seed
                    step_df["episode_idx"] = ep_idx
                    all_steps.append(step_df)

                    # Recovery-aware variant for non-clean conditions (optionally clean too)
                    if include_recovery and (condition != "clean" or include_clean_recovery):
                        recovery_method = method + "_recovery"
                        result_r, step_df_r = run_episode(
                            env=env,
                            method=recovery_method,
                            seed=episode_seed,
                            condition=condition,
                            attack_level=attack_level,
                            model=model,
                            policy_fn=policy_fn,
                            use_recovery=True,
                        )
                        row_r = asdict(result_r)
                        row_r["seed"] = seed
                        row_r["episode_idx"] = ep_idx
                        all_results.append(row_r)
                        step_df_r["seed"] = seed
                        step_df_r["episode_idx"] = ep_idx
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

    # Write timestamped run files into experiments/<cond_slug>/
    run_subdir = os.path.join(output_dir, cond_slug)
    os.makedirs(run_subdir, exist_ok=True)
    run_ep_path    = os.path.join(run_subdir, f"{run_stem}_episode_results.csv")
    run_step_path  = os.path.join(run_subdir, f"{run_stem}_step_logs.csv")
    run_score_path = os.path.join(run_subdir, f"{run_stem}_summary.csv")

    episode_df.to_csv(run_ep_path,    index=False)
    step_df.to_csv(run_step_path,     index=False)
    score_df.to_csv(run_score_path,   index=False)

    # --- Running combined CSVs (append, dedupe on key columns) ---------------
    EPISODE_KEYS = ["method", "condition", "attack_level", "seed", "episode_idx"]
    SUMMARY_KEYS = ["method", "condition", "attack_level"]

    combined_ep_path    = os.path.join(canonical_dir, "combined_episode_results.csv")
    combined_step_path  = os.path.join(canonical_dir, "combined_step_logs.csv")
    combined_score_path = os.path.join(canonical_dir, "combined_summary.csv")

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
    _append_deduped(step_df, combined_step_path, EPISODE_KEYS + ["timestep"])
    combined_score = _append_deduped(score_df,    combined_score_path, SUMMARY_KEYS)

    print(f"\nDone.")
    print(f"  Run files (per-run):  {run_ep_path}")
    print(f"                        {run_step_path}")
    print(f"                        {run_score_path}")
    print(f"  Canonical combined:   {combined_ep_path} ({len(combined_ep)} episode rows total)")
    print(f"                        {combined_score_path} ({len(combined_score)} summary rows total)")

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
        include_clean_recovery=args.include_clean_recovery,
        n_episodes=args.n_episodes,
        output_dir=args.output_dir,
        canonical_dir=args.canonical_dir,
    )


if __name__ == "__main__":
    main()
