# DQN-Timeseries-Anomaly-Detection

This Repository focuses on Anomaly Detection with Reinforcement Learning. Starting with basic DQN Agents exploring Timeseries, modelled as an environment.
The idea is taken from https://github.com/chengqianghuang/exp-anomaly-detector but I went with Tensorflow 2 as backend. 

# Prerequesites

1. create a conda environment from req.txt by running:
  ```$ conda create --name <env> --file <this file>``` or by installing the environment via yml file 
  ```conda env create -f <environment-name>.yml```
2. This might be different for Linux Distributions or MacOS. I still need to test on my Linux system 
# Work in Progress

1. (NOW) Refactor the Environment, so that we can use the Binary State Function. This is important to recognize the complete
state space. The windowed agent cannot recognize anomaly transitions that are stopping. This is also because of the 
reward function

2. (NOW) Refactor the experience replay so that we replay at a Starting Point of X samples and fill up with every following 
received transition. ATM the replay buffer is emptied after the first replay of the MAX samples.

2. (FUTURE) Try to implement Prioritized Experience Replay, this is needed for bigger datasets later on when we cant sample a
complete datset.

3. (FUTURE) Maybe look into the behavior of LSTMS and if they can be used for our problem

4. (FUTURE) Refine the n-step Q Learning and verify if it is correct

# Documentation
## Agent
The agent is using tensorflow keras nn models to predict on batches. The below UML diagram describes the current setup.
![Agent Class Diagram](http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/jorekai/DQN-Timeseries-Anomaly-Detection/master/uml/agent.puml)
