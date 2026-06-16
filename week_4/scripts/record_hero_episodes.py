"""
Record three hero video episodes for TAIRO Week 4 slides.

Saves mp4 clips to tairo_results/videos/:
  clean_seed0/            — SAC+HER, no attack (reward -2, crisp success)
  sensor_noise_0.05_seed2/ — SAC+HER, Scenario 1 heavy noise (reward -32)
  action_noise_0.05_seed0/ — SAC+HER, Scenario 2 action noise (reward -2)

Usage:
    python scripts/record_hero_episodes.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC

from attacks.sensor_attacks import add_sensor_noise
from attacks.action_attacks import manipulate_action
from config import MAX_EPISODE_STEPS, RESULT_DIR

gym.register_envs(gymnasium_robotics)

VIDEO_DIR = os.path.join(RESULT_DIR, "videos")
MODEL_PATH = os.path.join(RESULT_DIR, "sac_her_fetchreach_model")

EPISODES = [
    dict(label="clean_seed0",              seed=0, condition="clean",        attack_level=0.0),
    dict(label="sensor_noise_0.05_seed2",  seed=2, condition="sensor_noise", attack_level=0.05),
    dict(label="action_noise_0.05_seed0",  seed=0, condition="action_noise", attack_level=0.05),
]


def run_and_record(model, label, seed, condition, attack_level):
    out_dir = os.path.join(VIDEO_DIR, label)
    os.makedirs(out_dir, exist_ok=True)

    env = gym.make("FetchReach-v4", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=out_dir, episode_trigger=lambda _: True,
                      name_prefix=label)

    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    previous_action = np.zeros(env.action_space.shape, dtype=np.float32)

    for _ in range(MAX_EPISODE_STEPS):
        policy_obs = obs
        if condition == "sensor_noise":
            policy_obs = add_sensor_noise(obs, noise_std=attack_level)

        action, _ = model.predict(policy_obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)

        executed = action.copy()
        if condition == "action_noise":
            executed = manipulate_action(action, "action_noise", noise_std=attack_level)

        previous_action = executed.copy()
        obs, reward, terminated, truncated, info = env.step(executed)
        total_reward += float(reward)

        if terminated or truncated:
            break

    env.close()
    success = info.get("is_success", 0.0)
    print(f"  [{label}] reward={total_reward:.0f}  success={success}  -> {out_dir}/")
    return total_reward, success


def main():
    # Load model once with a throwaway env for HerReplayBuffer init
    tmp_env = gym.make("FetchReach-v4")
    model = SAC.load(MODEL_PATH, env=tmp_env)
    tmp_env.close()
    print(f"Loaded model from {MODEL_PATH}\n")

    print("Recording hero episodes...")
    for ep in EPISODES:
        run_and_record(model, **ep)

    print(f"\nAll videos saved to {VIDEO_DIR}/")


if __name__ == "__main__":
    main()
