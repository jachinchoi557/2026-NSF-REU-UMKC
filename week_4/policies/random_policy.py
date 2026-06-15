"""
Random policy — samples uniformly from the environment action space.

Used as the sanity-check / lower-bound baseline. Expected to have the
lowest success rate of all methods.
"""

from typing import Dict
import numpy as np


def random_policy(env, obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Return a random action sampled from the environment's action space.

    Args:
        env: Gymnasium environment (must have action_space.sample()).
        obs: Current observation dict (unused, kept for uniform signature).

    Returns:
        Random action array.
    """
    return env.action_space.sample()
