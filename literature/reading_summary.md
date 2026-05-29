## Gymnasium Summary (FetchReach-v3)
This evironment is a learning task where a robotic arm must move its gripper to a random target point in a 3D space. The robot used in said simulation is a 7 Degree-of-Freedom Fetch robot arm with a two-fingered gripper that runs in the MuJoCo physics engine. 


### Spaces
The action space contains movement on the dx, dy, and dz axes of the end effector. The observation space contains information regarding two things: The desired_goal (the three cartesian coordinates of the final desired position) and achieved_goal (the final location/state of the end effector). 

### Rewards
This environment contains both sparse and dense reward structures. In the sparse setting, rewards are only given when the agent is within a target threshold. In the dense setting, rewards are instead given based on distance between the gripper and the goal. 
_________________________________________________________

FetchReach is currently used as an introductory tool for robotic reinforcement as it isolates the challenge of goal-oriented motion without object manipulation. Understanding this environment is an important step in learning core concepts of goal-conditioned learning, continuous control, and training used throughout modern learning research. 

## AI Source Summary


## Cybersecurity-Robotics Summary
