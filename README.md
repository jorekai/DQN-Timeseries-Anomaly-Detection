# DQN-Timeseries-Anomaly-Detection

This Repository focuses on Anomaly Detection with Reinforcement Learning. Starting with basic DQN Agents exploring Timeseries, modelled as an environment.
The idea is taken from https://github.com/chengqianghuang/exp-anomaly-detector but I went with Tensorflow 2 as backend. The intermediate Goal is to run some Benchmarks on the most common Anomaly Detection Benchmark Datasets. In Future I might be able to support bi-/multivariate Timeseries. The Problem while working with univariate Timeseries is the small state space and strongly dependence on domain space of the Timeseries.

# Prerequesites

1. create a conda environment from req.txt by running:
  ```$ conda create --name <env> --file <this file>``` or by installing the environment via yml file 
  ```conda env create -f <environment-name>.yml```
2. This might be different for Linux Distributions or MacOS. I still need to test on my Linux system 
# Work in Progress

1. (NOW) Refactor the experience replay so that we replay at a Starting Point of X samples and fill up with every following 
received transition. ATM the replay buffer is emptied after the first replay of the MAX samples.

2. (FUTURE) Try to implement Prioritized Experience Replay, this is needed for bigger datasets later on when we cant sample a
complete datset.

3. (FUTURE) Maybe look into the behavior of LSTMS and if they can be used for our problem

4. (FUTURE) Refine the n-step Q Learning and verify if it is correct

5. (FUTURE) Include some testing for the environment and the agents

# Documentation
## Overview

The overview can be seen in the component diagram below. The Simulator is the Training Component used to train and to evaluate(WIP) agents in their environment.
The environment uses different state functions representing the internal state, also it needs to be instantiated with a Config. The data for the environment is found in the subfolder */ts_data*.

---
![Overview Diagram](http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/jorekai/DQN-Timeseries-Anomaly-Detection/master/uml/overview.puml)

## Environment
The Base Environment is a wrapable Object which must be contained by certain Custom Environments. The below UML diagram shows the necessary stuff.

![Environment Diagram](http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/jorekai/DQN-Timeseries-Anomaly-Detection/master/uml/environment.puml)

## Agent
The agent is using tensorflow keras nn models to predict on batches. The below UML diagram describes the current setup.

---
![Agent Class Diagram](http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/jorekai/DQN-Timeseries-Anomaly-Detection/master/uml/agent.puml)
