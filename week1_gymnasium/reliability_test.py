import gymnasium as gym
import gymnasium_robotics
import pandas as pd

results = []

# Is the robot-control system reliable across different initial conditions?

for seed in range(50, 61):
    env = gym.make("FetchReach-v4")
    obs, info = env.reset(seed=seed)

    total_reward = 0

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
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
pd.DataFrame(results).to_csv("/Users/yves/Documents/Github/2026-NSF-REU-UMKC/results/reliability_test_three.csv", index=False)
print(df)