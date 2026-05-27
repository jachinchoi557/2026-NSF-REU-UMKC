import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics

env = gym.make("FetchReach-v4", render_mode="human")
env = RecordEpisodeStatistics(env, buffer_length=50)
obs, info = env.reset(seed=1)

for step in range(1000):
    action = env.action_space.sample()

    # Simulated sensor attack: add small noise to observation vector
    if isinstance(obs, dict):
        noisy_obs = obs.copy()
        noisy_obs["observation"] = obs["observation"] + np.random.normal(0, 0.01, size=obs["observation"].shape)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
