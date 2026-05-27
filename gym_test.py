import gymnasium as gym
import gymnasium_robotics 

env = gym.make("FetchReach-v4", render_mode="human")
obs, info = env.reset(seed=1)

for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated,info = env.step(action)

    if terminated or truncated:
        obs,info = env.reset()

env.close
