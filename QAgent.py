import matplotlib.pyplot as plt
import random as rnd
from collections import namedtuple
import numpy as np

from Config import ConfigTimeSeries
from TimeSeriesModel import TimeSeriesEnvironment

import seaborn as sb
from sklearn.preprocessing import MinMaxScaler


class QAgent:
    def __init__(self, environment=TimeSeriesEnvironment(), epsilon=0.2, alpha=0.1, gamma=0.95):
        self.env = environment
        self.env.reset()
        self.env.normalize_timeseries()
        self.rewardSum = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rewardList = [0]
        self.result = self.env.timeseries_labeled
        self.state = self.env.statefunction(self.env.timeseries_cursor)
        self.next_state = None
        self.action = [0, 1]
        self.actions = []
        self.Qtable = np.zeros((len(self.env.timeseries_labeled), len(self.action)), dtype=np.float32)
        self.scaler = MinMaxScaler()
        self.epsilon = epsilon

    def act_random(self):
        if rnd.random() < .5:
            return self.action[0]
        else:
            return self.action[1]

    def act_explore_exploit(self, state):
        if rnd.random() < self.epsilon:
            action = self.act_random()
            # print("WENT RANDOMLY")
        else:
            q_value_normal = self.Qtable[state, self.action[0]]
            # print("VAL NORMAL: " + str(q_value_normal))
            q_value_anomaly = self.Qtable[state, self.action[1]]
            # print("VAL ANOMAL: " + str(q_value_normal))
            # print("POSITION: " + str(self.env.timeseries_cursor))
            # print("ISANOMAL: " + str(self.env.isanomaly(self.env.timeseries_cursor)))
            if q_value_anomaly > q_value_normal:
                action = self.action[1]
            else:
                action = self.action[0]
        self.get_reward(action)

    def get_reward(self, action_in):
        current_state, action, reward, next_state, done = self.env.step(action_in)
        # print(current_state, action, reward, next_state, done)
        self.actions.append(action_in)
        self.Qtable[current_state, action] = self.Qtable[current_state, action] + self.alpha * (
                reward + self.gamma * np.argmax(self.Qtable[next_state, :]) - self.Qtable[current_state, action])
        self.rewardSum += reward
        self.rewardList.append(self.rewardSum)

    def simulate(self):
        while not self.env.done:
            self.act_explore_exploit(self.env.statefunction(self.env.timeseries_cursor))
        self.result["Reward"] = self.rewardList
        self.normalize_sum()
        print("Sum Rewards: " + str(self.rewardSum))
        # plot_reward(self.result)

    def normalize_sum(self):
        self.result["Reward"] = self.scaler.fit_transform(self.result[["Reward"]])

    def reset_agent(self, env):
        self.env = env
        self.env.reset()
        self.env.normalize_timeseries()
        self.rewardList = [0]
        self.result = self.env.timeseries_labeled
        self.state = self.env.statefunction(self.env.timeseries_cursor)
        self.next_state = None
        self.action = [0, 1]
        self.actions = []
        self.rewardSum = 0


def plot_actions(actions):
    plt.figure(figsize=(15, 7))
    sb.lineplot(
        data=actions,
    ).set_title("Actions vs Series")
    plt.ylabel('Reward Sum')
    plt.show()


def plot_learn(data):
    plt.figure(figsize=(15, 7))
    sb.lineplot(
        data=data,
    ).set_title("Learning")
    plt.ylabel('Reward Sum')
    plt.show()


def plot_reward(result):
    plt.figure(figsize=(15, 7))
    sb.lineplot(
        data=result,
    ).set_title("Reward Random vs Series")
    plt.ylabel('Reward Sum')
    plt.show()


if __name__ == '__main__':
    # Test on Standard TimeSeries
    # agent = QAgent(
    #    environment=TimeSeriesEnvironment(verbose=False, filename="Test/SmallData.csv"))
    config = ConfigTimeSeries(seperator=";")
    # Test on complete Timeseries from SwAT
    agent = QAgent(
        environment=TimeSeriesEnvironment(verbose=False, filename="Attack_FIT101csv.csv", config=config))
    i = 0
    rewards = []
    eps_start = 0.2
    eps_dec = 0.9993
    eps_end = 0.0
    agent.env.plot()
    while i < 10:
        agent.simulate()
        rewards.append(agent.rewardSum)
        agent.reset_agent(TimeSeriesEnvironment(verbose=False, filename="Attack_FIT101csv.csv", config=config))
        # agent.reset_agent(env=TimeSeriesEnvironment(verbose=False, filename="Test/SmallData.csv"))
        if eps_start >= eps_end:
            eps_start *= eps_dec
            agent.epsilon = eps_start
        i += 1

    # Testing
    print("Testing----------")
    agent.reset_agent(TimeSeriesEnvironment(verbose=False, filename="Attack_FIT101csv.csv", config=config))
    # agent.reset_agent(env=TimeSeriesEnvironment(verbose=False, filename="Test/SmallData.csv"))
    agent.epsilon = 0.0
    agent.simulate()
    print("REWARDS: " + str(len(rewards)))
    print(agent.actions)
    print(len(agent.actions))
    plot_reward(rewards)
    plot_actions(agent.actions)

    # print("AVG" + str(np.average(rewards)))
    plot_learn(rewards)
