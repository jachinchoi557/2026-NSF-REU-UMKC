"""
FetchReach-v4 environment factory and observation utilities.

Sensor/action attack functions have been moved to attacks/.
Action metric helpers have been moved to evaluation/episode_runner.py.
"""

from typing import Dict

import numpy as np

from config import ENV_ID, GYM_AVAILABLE, MAX_EPISODE_STEPS


def make_env(seed: int = 0):
    """Create and return a seeded FetchReach-v4 Gymnasium environment.

    Args:
        seed: Random seed passed to env.reset().

    Returns:
        Gymnasium environment instance.

    Raises:
        RuntimeError: If gymnasium_robotics is not installed.
    """
    if not GYM_AVAILABLE:
        raise RuntimeError(
            "Gymnasium Robotics is not available. "
            "Install with: pip install gymnasium gymnasium-robotics mujoco"
        )
    import gymnasium as gym  # deferred so import errors are caught by config.py
    env = gym.make(ENV_ID, max_episode_steps=MAX_EPISODE_STEPS)
    env.reset(seed=seed)
    return env


def flatten_goal_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate observation, achieved_goal, and desired_goal into one vector.

    Useful for logging or feeding to algorithms that expect a flat input.

    Args:
        obs: Goal-conditioned observation dict from FetchReach-v4.

    Returns:
        1-D float32 array.
    """
    return np.concatenate([
        np.asarray(obs["observation"], dtype=np.float32).ravel(),
        np.asarray(obs["achieved_goal"], dtype=np.float32).ravel(),
        np.asarray(obs["desired_goal"], dtype=np.float32).ravel(),
    ])


def distance_to_goal(obs: Dict[str, np.ndarray]) -> float:
    """Euclidean distance between achieved_goal and desired_goal.

    Args:
        obs: Goal-conditioned observation dict from FetchReach-v4.

    Returns:
        Scalar distance in metres.
    """
    return float(np.linalg.norm(
        np.asarray(obs["achieved_goal"]) - np.asarray(obs["desired_goal"])
    ))
