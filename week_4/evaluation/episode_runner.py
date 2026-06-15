"""
Episode runner for TAIRO FetchReach-v4 benchmarks.

Provides:
    EpisodeResult — dataclass holding per-episode summary metrics.
    run_episode   — runs one episode under a given method and attack condition.

Supported method strings
------------------------
    "random"                  Random policy
    "rule_based"              Proportional reaching controller
    "sac" / "sac_her"         SB3 model (pass model= or policy_fn=)
    "recovery_aware_sac_her"  SB3 model with recovery damping enabled

All attack conditions, observation utilities, and recovery logic are
imported from their respective modules so this file contains only
orchestration logic.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import MAX_EPISODE_STEPS
from envs.fetchreach_env import distance_to_goal
from attacks.sensor_attacks import add_sensor_noise, shift_target
from attacks.action_attacks import manipulate_action
from policies.random_policy import random_policy
from policies.rule_based_policy import rule_based_reach_policy
from recovery.recovery_logic import maybe_apply_recovery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_smoothness(actions: List[np.ndarray]) -> float:
    """Mean step-to-step action-difference norm. Lower = smoother control."""
    if len(actions) < 2:
        return 0.0
    diffs = [np.linalg.norm(actions[i] - actions[i - 1]) for i in range(1, len(actions))]
    return float(np.mean(diffs))


def _action_magnitude(actions: List[np.ndarray]) -> float:
    """Mean action norm. Large values may indicate instability or attack amplification."""
    if not actions:
        return 0.0
    return float(np.mean([np.linalg.norm(a) for a in actions]))


# ---------------------------------------------------------------------------
# EpisodeResult
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Per-episode summary record. One row in the episode-level CSV."""
    method: str
    condition: str
    seed: int
    attack_level: float
    total_reward: float
    success: float           # 1.0 if is_success was ever True, else 0.0
    final_distance: float    # distance_to_goal at last step
    episode_length: int
    action_smoothness: float
    action_magnitude: float
    safety_violation: float  # 1.0 if any step exceeded action norm threshold
    recovery_used: float     # 1.0 if recovery was triggered at any step


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env,
    method: str,
    seed: int,
    condition: str = "clean",
    attack_level: float = 0.0,
    model=None,
    policy_fn: Optional[Callable] = None,
    use_recovery: bool = False,
    target_shift_step: int = 25,
) -> Tuple[EpisodeResult, pd.DataFrame]:
    """Run one episode and return a summary plus a step-level log DataFrame.

    Policy resolution order
    -----------------------
    1. ``policy_fn`` if provided (any callable ``(env, obs) -> action``).
    2. ``method`` string dispatch:
       - ``"random"``      → random_policy
       - ``"rule_based"``  → rule_based_reach_policy
       - ``"sac"``, ``"sac_her"``, ``"recovery_aware_sac_her"`` → model.predict
    3. Fallback: rule_based_reach_policy (with a warning).

    Attack condition dispatch
    -------------------------
    Observation-level:  ``sensor_noise``, ``target_shift``
    Action-level:       ``action_noise``, ``action_scale``,
                        ``action_reverse``, ``action_delay``

    Args:
        env:               Gymnasium environment (already created).
        method:            Policy identifier string.
        seed:              Random seed passed to env.reset().
        condition:         Attack condition name (see above).
        attack_level:      Float parameter for the attack (e.g., noise_std).
        model:             Trained SB3 model (required for sac / sac_her methods).
        policy_fn:         Optional callable that overrides ``method`` dispatch.
        use_recovery:      If True, apply TAIRO C5 recovery damping each step.
        target_shift_step: Step index at which target_shift is applied.

    Returns:
        Tuple of (EpisodeResult, step_log_DataFrame).
    """
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    actions: List[np.ndarray] = []
    step_logs: List[Dict] = []
    previous_action = np.zeros(env.action_space.shape, dtype=np.float32)
    any_recovery = False

    for t in range(MAX_EPISODE_STEPS):
        # -- Observation-level attacks ----------------------------------------
        policy_obs = obs
        if condition == "sensor_noise":
            policy_obs = add_sensor_noise(obs, noise_std=attack_level)
        elif condition == "target_shift" and t >= target_shift_step:
            policy_obs = shift_target(obs, shift_scale=attack_level)

        # -- Policy action -------------------------------------------------------
        if policy_fn is not None:
            action = policy_fn(env, policy_obs)
        elif method == "random":
            action = random_policy(env, policy_obs)
        elif method == "rule_based":
            action = rule_based_reach_policy(env, policy_obs)
        elif method in {"sac", "sac_her", "recovery_aware_sac_her"} and model is not None:
            action, _ = model.predict(policy_obs, deterministic=True)
        else:
            base = method.removesuffix("_recovery")
            if base == "random":
                action = random_policy(env, policy_obs)
            elif base == "rule_based":
                action = rule_based_reach_policy(env, policy_obs)
            elif base in {"sac", "sac_her"} and model is not None:
                action, _ = model.predict(policy_obs, deterministic=True)
            else:
                action = rule_based_reach_policy(env, policy_obs)

        intended_action = np.asarray(action, dtype=np.float32).copy()

        # -- Action-level attacks ------------------------------------------------
        executed_action = intended_action.copy()
        if condition == "action_noise":
            executed_action = manipulate_action(intended_action, "action_noise", noise_std=attack_level)
        elif condition == "action_scale":
            executed_action = manipulate_action(intended_action, "action_scale", scale=1.0 + attack_level)
        elif condition == "action_reverse":
            executed_action = manipulate_action(intended_action, "action_reverse")
        elif condition == "action_delay":
            executed_action = manipulate_action(intended_action, "action_delay", previous_action=previous_action)

        # -- Recovery (TAIRO C5) -------------------------------------------------
        recovery_triggered = False
        if use_recovery:
            executed_action, recovery_triggered = maybe_apply_recovery(
                executed_action, intended_action, policy_obs
            )
            if recovery_triggered:
                any_recovery = True

        previous_action = executed_action.copy()
        actions.append(executed_action.copy())

        # -- Environment step ----------------------------------------------------
        obs, reward, terminated, truncated, info = env.step(executed_action)
        total_reward += float(reward)

        current_distance = distance_to_goal(obs)
        is_success = float(info.get("is_success", 0.0))
        safety_violation_step = float(np.linalg.norm(executed_action) > 1.5)

        step_logs.append({
            "method": method,
            "condition": condition,
            "seed": seed,
            "attack_level": attack_level,
            "timestep": t,
            "reward": float(reward),
            "distance_to_goal": current_distance,
            "is_success": is_success,
            "action_norm": float(np.linalg.norm(executed_action)),
            "intended_action_norm": float(np.linalg.norm(intended_action)),
            "safety_violation": safety_violation_step,
            "recovery_triggered": float(recovery_triggered),
        })

        if terminated or truncated:
            break

    step_df = pd.DataFrame(step_logs)

    result = EpisodeResult(
        method=method,
        condition=condition,
        seed=seed,
        attack_level=float(attack_level),
        total_reward=float(total_reward),
        success=float(step_df["is_success"].max() if len(step_df) else 0.0),
        final_distance=float(step_df["distance_to_goal"].iloc[-1] if len(step_df) else float("nan")),
        episode_length=int(len(step_df)),
        action_smoothness=_action_smoothness(actions),
        action_magnitude=_action_magnitude(actions),
        safety_violation=float(step_df["safety_violation"].max() if len(step_df) else 0.0),
        recovery_used=float(any_recovery),
    )

    return result, step_df
