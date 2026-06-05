import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics
from PIL import Image
import pandas as pd
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

def apply_sensor_noise(obs, noise_std=0.05):
    noisy_obs = obs.copy()
    noisy_obs["observation"] = obs["observation"] + np.random.normal(0, noise_std, size=obs["observation"].shape)
    noisy_obs["achieved_goal"] = obs["achieved_goal"] + np.random.normal(0, noise_std, size=obs["achieved_goal"].shape)
    noisy_obs["desired_goal"]  = obs["desired_goal"]  + np.random.normal(0, noise_std, size=obs["desired_goal"].shape)
    return noisy_obs

results = []
gif_frames = []

num_episodes = 100
RECORD_EPISODES = set(range(10))
NOISE_STD = 0.01

env = gym.make("FetchReach-v4", render_mode="rgb_array")
env = RecordEpisodeStatistics(env)
print(f"Starting sensor noise attack for {num_episodes} episodes (noise_std={NOISE_STD})")

obs, info = env.reset(seed=1)

for episode_num in range(num_episodes):
    obs, info = env.reset(seed=episode_num + 1)
    episode_over = False
    recording = episode_num in RECORD_EPISODES

    while not episode_over:
        noisy_obs = apply_sensor_noise(obs, noise_std=NOISE_STD)
        obs, reward, terminated, truncated, info = env.step(simple_controller(noisy_obs))
        episode_over = truncated or terminated

        if recording:
            frame = env.render()
            gif_frames.append(Image.fromarray(frame))

    if "episode" in info:
        episode_data = info["episode"]
        results.append({
            "episode": episode_num,
            "seed": episode_num + 1,
            "condition": "sensor_noise",
            "noise_std": NOISE_STD,
            "reward": episode_data['r'],
            "time": episode_data['t'],
            "length": episode_data['l'],
            "success": info['is_success']
        })

# Save GIF
gif_filename = "sensor_noise_10episodes.gif"
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
    print(f"GIF saved to: {gif_path}")

# Save results
pd.DataFrame(results).to_csv(
    os.path.join(RESULTS_PATH, "sensor_noise_with_sc.csv"),
    index=False
)
print(f"Results saved to: {RESULTS_PATH}")
env.close()