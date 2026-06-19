"""
Plain SAC (no HER) training and evaluation for FetchReach-v4.

Identical to train_sac_her.py in every way EXCEPT:
- No HerReplayBuffer — uses SAC's default replay buffer
- Saves to tairo_results/sac_plain_fetchreach_model

This is used as a baseline to isolate HER's contribution on the sparse-reward
FetchReach task.  Without goal relabelling, the agent almost never encounters
a positive reward signal, so we expect significantly lower performance than
SAC+HER under clean conditions.  That gap is the finding, not a mistake.

Hyperparameters
---------------
total_timesteps : 20_000  (matches SAC+HER run)
seed            : 0
learning_rate   : 1e-3
buffer_size     : 100_000
batch_size      : 256
gamma           : 0.95
tau             : 0.05
"""

import os
import sys

# Ensure project root is on path when called from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional, Tuple

import pandas as pd

from config import (
    ENV_ID,
    GYM_AVAILABLE,
    MAX_EPISODE_STEPS,
    RANDOM_SEEDS,
    RESULT_DIR,
    SB3_AVAILABLE,
)
from envs.fetchreach_env import make_env
from evaluation.episode_runner import run_episode
from dataclasses import asdict

EVAL_CONDITIONS = [
    ("clean",        0.00),
    ("sensor_noise", 0.01),
    ("sensor_noise", 0.05),
    ("action_noise", 0.05),
    ("action_scale", 0.50),
    # action_delay intentionally omitted — zero-action bug confirmed 2026-06-16
    ("target_shift", 0.03),
]


def train_sac_plain(
    total_timesteps: int = 20_000,
    seed: int = 0,
    learning_rate: float = 1e-3,
    buffer_size: int = 100_000,
    batch_size: int = 256,
    gamma: float = 0.95,
    tau: float = 0.05,
    save_path: Optional[str] = None,
):
    """Train plain SAC (no HER) on FetchReach-v4 and save the model."""
    if not GYM_AVAILABLE:
        raise RuntimeError(
            "Gymnasium Robotics is not available. "
            "Install with: pip install gymnasium gymnasium-robotics mujoco"
        )
    if not SB3_AVAILABLE:
        raise RuntimeError(
            "Stable-Baselines3 is not available. "
            "Install with: pip install stable-baselines3"
        )

    from stable_baselines3 import SAC

    env = make_env(seed=seed)

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        seed=seed,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        # No HerReplayBuffer — default replay buffer used
    )

    print(f"Training plain SAC for {total_timesteps} timesteps (seed={seed}) ...")
    model.learn(total_timesteps=total_timesteps)
    env.close()

    if save_path is None:
        save_path = os.path.join(RESULT_DIR, "sac_plain_fetchreach_model")

    model.save(save_path)
    print(f"Model saved to: {save_path}")
    return model


def sanity_check(model, n_episodes: int = 5, seed: int = 0) -> float:
    """Run n_episodes headlessly and report success rate."""
    import gymnasium as gym
    import gymnasium_robotics
    import numpy as np

    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID, max_episode_steps=MAX_EPISODE_STEPS)

    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_success = False
        t = 0
        while not done and t < MAX_EPISODE_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("is_success", 0.0):
                ep_success = True
            done = terminated or truncated
            t += 1
        successes += int(ep_success)
        print(f"  Episode {ep}: {'SUCCESS' if ep_success else 'FAIL'}")

    env.close()
    rate = successes / n_episodes
    print(f"Sanity check: {successes}/{n_episodes} episodes succeeded ({rate:.0%})")
    return rate


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train plain SAC (no HER) on FetchReach-v4.")
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--sanity-check-only", action="store_true",
                        help="Skip training; load existing model and run sanity check.")
    args = parser.parse_args()

    if args.sanity_check_only:
        from stable_baselines3 import SAC
        path = args.save_path or os.path.join(RESULT_DIR, "sac_plain_fetchreach_model")
        env = make_env(seed=0)
        model = SAC.load(path, env=env)
        env.close()
    else:
        model = train_sac_plain(
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            save_path=args.save_path,
        )

    sanity_check(model, n_episodes=5, seed=0)
