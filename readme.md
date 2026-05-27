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
2. How sensitive is the robot to noisy observations?
3. How sensitive is the robot to manipulated actions?

## How to Run

python week1_gymnasium/run_fetchreach_random.py
python week1_gymnasium/reliability_test.py
python week1_gymnasium/sensor_noise_attack.py
python week1_gymnasium/action_attack.py