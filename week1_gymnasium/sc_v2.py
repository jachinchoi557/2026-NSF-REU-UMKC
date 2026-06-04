import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordEpisodeStatistics
import pandas as pd
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
RESULTS_PATH = os.getenv("RESULTS_PATH")

def simple_controller(obs, gain=5.0):
    """Move gripper toward goal proportionally."""
    achieved = obs["achieved_goal"]  
    desired  = obs["desired_goal"]    
    error    = desired - achieved    
    action   = gain * error           
    action_full = np.append(action, 0.0)
    return np.clip(action_full, -1.0, 1.0)

results = []
gif_frames = []  

num_training_episodes = 100
RECORD_EPISODES = set(range(10)) 

env = gym.make("FetchReach-v4", render_mode="rgb_array")
env = RecordEpisodeStatistics(env)
print(f"Starting training for {num_training_episodes} episodes")

obs, info = env.reset(seed=1)

for episode_num in range(num_training_episodes):
    obs, info = env.reset(seed=episode_num + 1)
    episode_over = False
    recording = episode_num in RECORD_EPISODES

    while not episode_over:
        obs, reward, terminated, truncated, info = env.step(simple_controller(obs))
        episode_over = truncated or terminated

        if recording:
            frame = env.render()
            gif_frames.append(Image.fromarray(frame))

    if "episode" in info:
        episode_data = info["episode"]
        results.append({
            "episode": episode_num,
            "seed": episode_num + 1,
            "condition": "baseline",
            "reward": episode_data['r'],
            "time": episode_data['t'],
            "length": episode_data['l'],
            "success": info['is_success']
        })

gif_filename = "fetchreach_10episodes.gif"
gif_path = os.path.join(RESULTS_PATH, gif_filename)

os.makedirs(RESULTS_PATH, exist_ok=True)

if gif_frames:
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=33,
        loop=0
    )
    print(f"GIF saved to:{gif_path}")

pd.DataFrame(results).to_csv(
    os.path.join(RESULTS_PATH, "baseline_results_sc.csv"),
    index=False
)
env.close()