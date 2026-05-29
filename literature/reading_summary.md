## Gymnasium Summary (FetchReach-v3)
This evironment is a learning task where a robotic arm must move its gripper to a random target point in a 3D space. The robot used in said simulation is a 7 Degree-of-Freedom Fetch robot arm with a two-fingered gripper that runs in the MuJoCo physics engine. 


### Spaces
The action space contains movement on the dx, dy, and dz axes of the end effector. The observation space contains information regarding two things: The desired_goal (the three cartesian coordinates of the final desired position) and achieved_goal (the final location/state of the end effector). 

### Rewards
This environment contains both sparse and dense reward structures. In the sparse setting, rewards are only given when the agent is within a target threshold. In the dense setting, rewards are instead given based on distance between the gripper and the goal. 
_________________________________________________________

FetchReach is currently used as an introductory tool for robotic reinforcement as it isolates the challenge of goal-oriented motion without object manipulation. Understanding this environment is an important step in learning core concepts of goal-conditioned learning, continuous control, and training used throughout modern learning research. 

## AI Source Summary
AI RMF 1.0 is organized around four core functions: Govern, Map, Measure, and Manage. Govern establishes organizational policies, accountability, and oversight. Map focuses on understanding the context in which an AI system operates and identifying potential risks. Measure involves assessing and monitoring those risks using quantitative and qualitative methods. Manage focuses on prioritizing and mitigating risks while continuously monitoring system performance. Together, these functions create a continuous cycle of AI risk management.

The NIST Artificial Intelligence Risk Management Framework (AI RMF 1.0) provides a voluntary framework for identifying, assessing, and managing risks associated with AI systems throughout their lifecycle. Rather than prescribing specific technical solutions, the framework offers a structured approach that organizations can adapt to different industries and applications. Its primary goal is to encourage the development and deployment of trustworthy AI systems while minimizing potential harms to individuals, organizations, and society.

The framework defines several characteristics of trustworthy AI, including validity, reliability, safety, security, transparency, explainability, privacy protection, accountability, and fairness. These characteristics serve as guiding principles for evaluating AI systems and their potential impacts.
_________________________________________________________

This framework is particularly important because it recognizes that AI risks are not purely technical but also social and organizational. By providing a common structure for evaluating AI systems, the AI RMF helps organizations build more trustworthy, transparent, and responsible AI technologies while balancing innovation with risk management.

## Cybersecurity-Robotics Summary
