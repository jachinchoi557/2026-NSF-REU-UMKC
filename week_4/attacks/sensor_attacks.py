"""
Observation-level attack functions (Scenario 1 — Sensor Attacks).

add_sensor_noise : Gaussian noise injected into every observation field.
shift_target     : Desired-goal spoofing / unexpected goal change.
"""

from typing import Dict
import numpy as np


def add_sensor_noise(
    obs: Dict[str, np.ndarray],
    noise_std: float,
) -> Dict[str, np.ndarray]:
    """Add Gaussian noise to every field of a goal-conditioned observation dict.

    Preserves the dictionary structure so the attacked observation is a
    drop-in replacement for the original.

    Args:
        obs:       Raw observation dict from FetchReach-v4.
        noise_std: Standard deviation of the zero-mean Gaussian noise.

    Returns:
        New dict with independent noise applied to each array field.
    """
    attacked: Dict[str, np.ndarray] = {}
    for key, value in obs.items():
        arr = np.asarray(value).copy()
        attacked[key] = arr + np.random.normal(loc=0.0, scale=noise_std, size=arr.shape)
    return attacked


def shift_target(
    obs: Dict[str, np.ndarray],
    shift_scale: float = 0.03,
) -> Dict[str, np.ndarray]:
    """Shift the desired_goal field to simulate target spoofing.

    Models an adversary that feeds the controller a false goal position,
    causing it to reach toward the wrong location.

    Args:
        obs:         Raw observation dict from FetchReach-v4.
        shift_scale: Maximum per-axis uniform shift in metres.

    Returns:
        New dict with desired_goal perturbed; all other fields are unchanged.
    """
    attacked = {key: np.asarray(value).copy() for key, value in obs.items()}
    attacked["desired_goal"] = attacked["desired_goal"] + np.random.uniform(
        low=-shift_scale,
        high=shift_scale,
        size=attacked["desired_goal"].shape,
    )
    return attacked
