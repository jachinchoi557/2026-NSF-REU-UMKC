# Evaluating Robot Control Reliability Under Sensor Noise Attacks in Gymnasium Robotics

Team Members: 
- Jachin Choi
- Yves Velasquez Vega

Platform: Gymnasium Robotics (FetchReach-v4), MuJoCo, Python 3.11

### Background
Robots rely on sensor data to make decisions, if an attacker manipulates those sensor readings even by small amounts the robot may behave unsafely or fail to complete its task. This is known as sensor spoofing or adversarial input attack and it's a real concern in cybersecurity systems like autonomous robots.

### Method
We will use the FetchReach-v4 environment from Gymnasium Robotics. We will run two sets of trials: one with clean observations and one where small Gaussian noise is added to the observation vector to simulate a sensor attack. We will compare total reward and success rate across multiple random seeds.

### Expected Outcome
We expect that increasing noise levels will reduce the robot's ability to reach its target, showing that even small sensor manipulations can meaningfully degrade performance and reliability.