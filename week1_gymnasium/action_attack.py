import gymnasium as gym
import gymnasium_robotics
import numpy as np

env = gym.make("FetchReach-v4")
obs, info = env.reset(seed=2)

for step in range(100):
    action = env.action_space.sample()

    # Simulated control attack: perturb the robot command
    attacked_action = action + np.random.normal(0, 0.05, size=action.shape)
    attacked_action = np.clip(attacked_action, env.action_space.low, env.action_space.high)

    obs, reward, terminated, truncated, info = env.step(attacked_action)

    if terminated or truncated:
        obs, info = env.reset()
env.close()