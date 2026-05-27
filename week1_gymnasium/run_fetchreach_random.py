import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordEpisodeStatistics

env = gym.make("FetchReach-v4", render_mode="human")
env = RecordEpisodeStatistics(env, buffer_length=50)
obs, info = env.reset(seed=1)

for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()