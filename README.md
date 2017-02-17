# RL-agent

This is an implementation of [PGQ: Combining policy gradient and Q-learning](https://arxiv.org/abs/1611.01626)
Also it contains additional hacks, including:
- n-step A3C update
- soft target network

This agent is implemented using distributed Tensorflow + Redis for synchronising experience replay and weights

Requirements:
-Numpy
-Scipy
-Tensorflow
-Redis (and redis server)
-Joblib
-Gym
-OpenCV (for screen preprocessing)

To run:
```
python3 run_agent.py
```

After the run you should kill redis-server process and all worker processes


