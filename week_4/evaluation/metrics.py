"""
TAIRO metric computation — summary statistics and trustworthiness scores.

Metric fields produced by add_trustworthiness_scores()
-------------------------------------------------------
reliability_score     — Does the robot succeed under its operating condition?
robustness_score      — How well does performance hold under disturbances?
cyber_resilience_score— How well does the robot resist adversarial attacks?
safety_score          — Does the robot avoid unsafe actuator behaviour?
recovery_score        — Does the robot detect and correct attack-induced drift?

These names are intentionally distinct from the project's TAIRO framework
component labels (C1–C6 defined in Week 3) to avoid a naming collision.
The C1–C6 component labels refer to pipeline modules; the fields here are
outcome metrics rolled up into the C6 trustworthiness aggregate.

Weight rationale (matches Week 3 framework component weights C1–C5 in order)
-----------------------------------------------------------------------------
C1 Perception & State Understanding      → reliability_score      0.10
C2 Uncertainty & Failure Detection       → robustness_score       0.25
C3 Cybersecurity-Aware Reasoning         → cyber_resilience_score 0.25
C4 RL-Based Adaptation                   → safety_score           0.05
C5 Failure Recovery & Safety Control     → recovery_score         0.35
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Trustworthiness weights — match Week 3 TAIRO framework C1–C5 (must sum 1.0)
# ---------------------------------------------------------------------------
WEIGHT_RELIABILITY = 0.10
WEIGHT_ROBUSTNESS = 0.25
WEIGHT_CYBER_RESILIENCE = 0.25
WEIGHT_SAFETY = 0.05
WEIGHT_RECOVERY = 0.35

assert abs(
    WEIGHT_RELIABILITY
    + WEIGHT_ROBUSTNESS
    + WEIGHT_CYBER_RESILIENCE
    + WEIGHT_SAFETY
    + WEIGHT_RECOVERY
    - 1.0
) < 1e-9, "TAIRO trustworthiness weights must sum to 1.0"

# Safety violation threshold: action norm above this is flagged as unsafe.
# Matches the threshold used in episode_runner.
SAFETY_ACTION_NORM_THRESHOLD = 1.5


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate episode-level records into per-(method, condition, attack_level) rows.

    Args:
        df: Episode-level DataFrame produced by run_episode / run_benchmark.

    Returns:
        Summary DataFrame with one row per (method, condition, attack_level)
        group, containing mean values for all key metrics.
    """
    group_cols = ["method", "condition", "attack_level"]
    summary = (
        df.groupby(group_cols)
        .agg(
            success_rate=("success", "mean"),
            avg_reward=("total_reward", "mean"),
            final_distance=("final_distance", "mean"),
            avg_episode_length=("episode_length", "mean"),
            action_smoothness=("action_smoothness", "mean"),
            action_magnitude=("action_magnitude", "mean"),
            safety_violation_rate=("safety_violation", "mean"),
            recovery_rate=("recovery_used", "mean"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    return summary


def add_trustworthiness_scores(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute TAIRO C1–C5 sub-scores and the composite trustworthiness score.

    Sub-score definitions
    ---------------------
    C1 Reliability        = success_rate (direct, clipped to [0, 1])

    C2 Robustness         = 0.6 * (1 - normalised_distance)
                          + 0.4 * (1 - normalised_smoothness)
                          Captures both how close the robot gets and how
                          smoothly it moves (attack-induced jitter degrades
                          smoothness).

    C3 Cyber Resilience   = success_rate
                          Reuses reliability because attack resilience is
                          directly observable as maintained task success.
                          (clean condition receives resilience = reliability)

    C4 Safety             = 1 - safety_violation_rate

    C5 Recovery           = recovery_rate for attacked conditions;
                          clean condition is set to 1.0 by convention since
                          there is nothing to recover from.

    Composite score       = sum(weight_i * score_i) for i in C1..C5

    Args:
        summary: Output of summarize_results().

    Returns:
        summary with C1–C5 sub-score columns and ``trustworthiness_score``
        appended.
    """
    out = summary.copy()

    # Reliability — maps to framework C1 (Perception & State Understanding)
    out["reliability_score"] = out["success_rate"].clip(0.0, 1.0)

    # Robustness — maps to framework C2 (Uncertainty & Failure Detection)
    max_dist = max(out["final_distance"].max(), 1e-9)
    max_smooth = max(out["action_smoothness"].max(), 1e-9)

    distance_score = 1.0 - (out["final_distance"] / max_dist)
    smoothness_score = 1.0 - (out["action_smoothness"] / max_smooth)
    out["robustness_score"] = (0.6 * distance_score + 0.4 * smoothness_score).clip(0.0, 1.0)

    # Cyber resilience — maps to framework C3 (Cybersecurity-Aware Reasoning)
    out["cyber_resilience_score"] = out["success_rate"].clip(0.0, 1.0)

    # Safety — maps to framework C4 (RL-Based Adaptation)
    out["safety_score"] = (1.0 - out["safety_violation_rate"]).clip(0.0, 1.0)

    # Recovery — maps to framework C5 (Failure Recovery & Safety Control)
    out["recovery_score"] = out["recovery_rate"].clip(0.0, 1.0)
    out.loc[out["condition"] == "clean", "recovery_score"] = 1.0

    # Composite trustworthiness score (C6 aggregate)
    out["trustworthiness_score"] = (
        WEIGHT_RELIABILITY * out["reliability_score"]
        + WEIGHT_ROBUSTNESS * out["robustness_score"]
        + WEIGHT_CYBER_RESILIENCE * out["cyber_resilience_score"]
        + WEIGHT_SAFETY * out["safety_score"]
        + WEIGHT_RECOVERY * out["recovery_score"]
    )

    return out
