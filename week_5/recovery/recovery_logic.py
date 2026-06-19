"""
Recovery logic (TAIRO C5 — Recovery capability).

maybe_apply_recovery detects attack-induced deviation using three signals
and responds by substituting the rule-based reaching policy rather than
damping the original (potentially corrupted) action.

Detection signals
-----------------
1. Action divergence  : ||current_action - prev_action|| > 0.5
   Large step-to-step action changes are a hallmark of injected noise
   or scaling attacks.

2. Distance trend     : distance_to_goal has been increasing for 5+
   consecutive steps (monotone worsening).  Tracks that the robot is
   being driven away from the goal rather than toward it.

3. Jerk               : ||current_action - prev_action|| > 1.0
   (Same quantity as divergence but at a higher threshold, capturing
   sudden lurches that indicate action reversal or large-noise attacks.)

Recovery response
-----------------
When ANY signal fires, the executed action is replaced by the output of
rule_based_reach_policy, which always steers toward the true goal using
the *unattacked* obs.  This is strictly better than damping for goal-
conditioned tasks because damping just slows an already-corrupted
trajectory, whereas replanning re-anchors the robot to the correct goal.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from policies.rule_based_policy import rule_based_reach_policy

# ---------------------------------------------------------------------------
# Thresholds (module-level constants so callers can inspect / override)
# ---------------------------------------------------------------------------
ACTION_DIVERGENCE_THRESHOLD = 0.5   # ||current - prev|| to trigger
JERK_THRESHOLD              = 1.0   # same quantity, higher bar
DISTANCE_TREND_WINDOW       = 5     # consecutive worsening steps required


def maybe_apply_recovery(
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
    prev_obs: Optional[Dict[str, np.ndarray]],
    prev_action: Optional[np.ndarray],
    step_distances: List[float],
    step: int,
    env,
) -> Tuple[np.ndarray, bool]:
    """Detect attack-induced deviation and substitute rule-based control.

    Args:
        obs:            Current (possibly attacked) observation dict.
                        Must contain ``achieved_goal`` and ``desired_goal``.
        action:         Current executed action (after any attack manipulation).
        prev_obs:       Observation from the previous step (None at step 0).
        prev_action:    Executed action from the previous step (None at step 0).
        step_distances: Running list of distance_to_goal values recorded at
                        the END of each prior step.  Append to this list each
                        step so the trend detector has a history.
        step:           Current timestep index (0-indexed).
        env:            The Gymnasium environment (needed to call the
                        rule-based policy, which uses env.action_space for
                        clipping).

    Returns:
        Tuple of:
            - final_action  : ndarray — the action to actually execute.
                              Either the original ``action`` (no recovery) or
                              the rule-based policy output (recovery triggered).
            - triggered     : bool — True when recovery fired this step.
    """
    action = np.asarray(action, dtype=np.float32)

    if prev_action is None or step == 0:
        # No history yet — cannot compute divergence or jerk reliably.
        return action, False

    prev_action = np.asarray(prev_action, dtype=np.float32)

    # -- Signal 1 & 3: action divergence / jerk --------------------------------
    delta = float(np.linalg.norm(action - prev_action))
    divergence_triggered = delta > ACTION_DIVERGENCE_THRESHOLD
    jerk_triggered       = delta > JERK_THRESHOLD

    # -- Signal 2: distance trend ----------------------------------------------
    trend_triggered = False
    if len(step_distances) >= DISTANCE_TREND_WINDOW:
        window = step_distances[-DISTANCE_TREND_WINDOW:]
        # True when every consecutive pair in the window is increasing
        trend_triggered = all(
            window[i] < window[i + 1] for i in range(DISTANCE_TREND_WINDOW - 1)
        )

    triggered = divergence_triggered or jerk_triggered or trend_triggered

    if triggered:
        # Replace the (potentially corrupted) action with rule-based control
        # using the raw observation so the policy steers toward the real goal.
        recovery_action = rule_based_reach_policy(env, obs)
        return np.asarray(recovery_action, dtype=np.float32), True

    return action, False
