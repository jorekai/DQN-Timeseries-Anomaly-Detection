import matplotlib.pyplot as plt
import random as rnd

from environment.Config import ConfigTimeSeries
from environment.TimeSeriesModel import TimeSeriesEnvironment
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler


class RandomAgent():
    def __init__(self, environment=TimeSeriesEnvironment()):
        self.env = environment
        self.rewardSum = 0
        self.rewardList = [0]
        self.result = self.env.timeseries_labeled
        self.scaler = MinMaxScaler()

    def act(self):
        if rnd.random() < .5:
            return 0
        else:
            return 1

    def getReward(self):
        action = self.act()
        reward = self.env.step(action)[2]
        self.rewardSum += reward
        self.rewardList.append(self.rewardSum)

    def simulate(self):
        self.env.reset()
        self.env.normalize_timeseries()
        while not self.env.done:
            self.getReward()
        self.result["Reward"] = self.rewardList
        self.normalize_sum()
        print(self.result)
        self.plot_reward()

    def plot_reward(self):
        plt.figure(figsize=(15, 7))
        sb.lineplot(
            data=self.result,
        ).set_title("Reward Random vs Series")
        plt.ylabel('Reward Sum')
        plt.show()

    def normalize_sum(self):
        self.result["Reward"] = self.scaler.fit_transform(self.result[["Reward"]])


if __name__ == '__main__':
    agent = RandomAgent(
        environment=TimeSeriesEnvironment(verbose=False,
                                          config=ConfigTimeSeries(normal=0, anomaly=1, reward_correct=1,
                                                                  reward_incorrect=-1,
                                                                  action_space=[0, 1], seperator=",", boosted=False),
                                          filename="./Test/SmallData.csv"))
    agent.simulate()
