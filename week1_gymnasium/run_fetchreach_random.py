import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordEpisodeStatistics
import logging
import pandas as pd

results = []

num_training_episodes = 100
logging.basicConfig(level=logging.INFO, format="%(message)s")
env = gym.make("FetchReach-v4", render_mode="human")

# Logging 
env = RecordEpisodeStatistics(env)
print(f"Starting training for {num_training_episodes} episodes")

obs,info = env.reset(seed=1)
# Main Loop
for episode_num in range(num_training_episodes):
    obs,info = env.reset(seed=episode_num + 1)
    episode_over = False

    while not episode_over:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_over = truncated or terminated
    
    # Log episode statistics (available in info after episode ends)
    if "episode" in info:
        episode_data = info["episode"]
        results.append({"episode": episode_num,
                        "seed": episode_num + 1,
                        "condition": "baseline",
                        "reward": episode_data['r'],
                        "time": episode_data['t'],
                        "length": episode_data['l'],
                        "success": info['is_success']})
pd.DataFrame(results).to_csv("/Users/yves/Documents/Github/2026-NSF-REU-UMKC/results/baseline_results.csv", index=False)
env.close()