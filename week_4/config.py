"""
Shared configuration for the TAIRO Week 4 benchmark.

All modules import constants and availability flags from here so there is
a single place to change seeds, the environment ID, or result paths.
"""

import os
import random

import numpy as np

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
RANDOM_SEEDS = [0, 1, 2]
ENV_ID = "FetchReach-v4"
MAX_EPISODE_STEPS = 50
RESULT_DIR = "tairo_results"

os.makedirs(RESULT_DIR, exist_ok=True)

# Global seeding — set once at import time so results are reproducible
np.random.seed(0)
random.seed(0)

# ---------------------------------------------------------------------------
# Optional dependency flags
# Modules guard their imports behind these flags so the rest of the pipeline
# degrades gracefully when MuJoCo or SB3 are not installed.
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym  # noqa: F401
    import gymnasium_robotics  # noqa: F401
    GYM_AVAILABLE = True
except Exception as _gym_err:
    GYM_AVAILABLE = False
    print(f"[config] Gymnasium Robotics not available: {_gym_err!r}")

try:
    from stable_baselines3 import SAC  # noqa: F401
    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer  # noqa: F401
    SB3_AVAILABLE = True
except Exception as _sb3_err:
    SB3_AVAILABLE = False
    print(f"[config] Stable-Baselines3 not available: {_sb3_err!r}")