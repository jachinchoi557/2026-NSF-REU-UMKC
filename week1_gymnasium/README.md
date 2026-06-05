# REU Trustworthy Robotics and Cybersecurity

## Project Title
Trustworthy Robot Control under Cyber-Physical Perturbations

## Team Members
- Jachin Choi
- Yves Velasquez Vega


## Week 1 Goal
Run Gymnasium Robotics FetchReach-v4 and evaluate reliability under normal and perturbed conditions.

## Platform
- Gymnasium Robotics
- MuJoCo
- Python 3.11

## Cybersecurity Scenario
We simulate sensor noise and action manipulation as simple cyber-physical attacks.

## Research Questions
1. How reliable is robot behavior across random seeds?

    In FetchReach-v4 with a random policy, the robot is consistently unreliable across all seeds but in a stable way. Because the policy just samples random actions, it never actually reaches the target goal. Reward stays near -1.0 and success rate stays near 0% regardless of which seed you use. Although the robot never succeeds, the consistency across seeds confirms our environment is set up correctly and gives us a baseline floor for comparison.

2. How sensitive is the robot to noisy observations?
    
    We injected sensor noise into ```obs["observation"]``` which contains things like the robot's joint velocities and end-effector position. But the simple controller and the random sampling policy never actually read that field. This means the attack surface is  **policy-dependent**. Sensor noise only threatens a policy that actually reads those sensors to make decisions. To make this attack meaningful, we would need to train a policy like SAC+HER that reads ```obs["observation"]``` then attack that.

3. How sensitive is the robot to manipulated actions?

    The robot is very sensitive to manipulated actions regardless of which policy is running because instead of corrupting what the robot actually reads we are corrupting what the robot actually does. The perturbed action gets sent directly to the physics engine regardless of what policy is running. Corrupting the action **directly degrades physical behavior**. 

## How to Run

python week1_gymnasium/run_fetchreach_random.py
python week1_gymnasium/reliability_test.py
python week1_gymnasium/sensor_noise_attack.py
python week1_gymnasium/action_attack.py