import gymnasium as gym
import gymnasium_robotics
import pandas as pd
import numpy as np

gym.register_envs(gymnasium_robotics) 
results = []
env = gym.make("FetchReach-v4", render_mode="human")


# Is the robot-control system reliable across different initial conditions?

def simple_controller(obs, gain=5.0):
    """Move gripper toward goal proportionally."""
    achieved = obs["achieved_goal"]   # current end-effector xyz
    desired  = obs["desired_goal"]    # target xyz
    error    = desired - achieved     # direction to move
    action   = gain * error           # proportional control
    # FetchReach action is [dx, dy, dz, gripper] — clip to valid range
    action_full = np.append(action, 0.0)
    return np.clip(action_full, -1.0, 1.0)


for seed in range(50, 61):
    env = gym.make("FetchReach-v4")
    obs, info = env.reset(seed=seed)

    total_reward = 0

    for step in range(100):
        obs, reward, terminated, truncated, info = env.step(simple_controller(obs))
        total_reward += reward

        if terminated or truncated:
            break

    results.append({
        "seed": seed,
        "steps": step + 1,
        "total_reward": total_reward,
        "success": info.get("is_success", None)
    })

    env.close()

df = pd.DataFrame(results)
pd.DataFrame(results).to_csv("/Users/yves/Documents/Github/2026-NSF-REU-UMKC/results/simple_controller.csv", index=False)
print(df)