"""
Recovery logic (TAIRO C5 — Recovery capability).

maybe_apply_recovery detects signs of attack-induced deviation and applies
a conservative damping response: reduce the executed action magnitude so
the robot slows down and avoids amplifying a potentially corrupted command.

Detection heuristics
--------------------
1. Action divergence: the norm of (executed - intended) exceeds a threshold,
   indicating the action has been significantly altered by an attack.
2. Distance alarm: the robot is far from the goal, suggesting it has been
   driven off-course.

When either condition is met the executed action is scaled down by
``scale`` (default 0.5) — a simple but effective damping recovery that
reduces the risk of safety violations without requiring a separate
recovery policy.

This is the TAIRO baseline recovery. More sophisticated strategies
(e.g., anomaly-triggered replanning, goal reset) can be added here.
"""

from typing import Tuple, Dict
import numpy as np


def maybe_apply_recovery(
    executed_action: np.ndarray,
    intended_action: np.ndarray,
    obs: Dict[str, np.ndarray],
    *,
    action_divergence_threshold: float = 0.15,
    distance_threshold: float = 0.25,
    scale: float = 0.5,
) -> Tuple[np.ndarray, bool]:
    """Apply conservative damping if attack-induced deviation is detected.

    Args:
        executed_action:            Action after attack manipulation.
        intended_action:            Action originally chosen by the policy.
        obs:                        Current (possibly attacked) observation
                                    dict with ``achieved_goal`` and
                                    ``desired_goal`` keys.
        action_divergence_threshold: Trigger when
                                    ``||executed - intended|| > threshold``.
        distance_threshold:         Trigger when distance-to-goal exceeds
                                    this value (in metres).
        scale:                      Factor to multiply the executed action
                                    when recovery is triggered.

    Returns:
        Tuple of:
            - (possibly damped) action array
            - bool indicating whether recovery was triggered this step
    """
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
    desired = np.asarray(obs["desired_goal"], dtype=np.float32)
    distance = float(np.linalg.norm(achieved - desired))

    action_divergence = float(np.linalg.norm(
        np.asarray(executed_action, dtype=np.float32)
        - np.asarray(intended_action, dtype=np.float32)
    ))

    triggered = (
        action_divergence > action_divergence_threshold
        or distance > distance_threshold
    )

    if triggered:
        return scale * np.asarray(executed_action, dtype=np.float32), True

    return np.asarray(executed_action, dtype=np.float32), False
