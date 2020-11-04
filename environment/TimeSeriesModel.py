import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

from resources import Plots
from environment.Config import ConfigTimeSeries


class TimeSeriesEnvironment:
    def __init__(self, directory="../ts_data/",
                 config=ConfigTimeSeries(normal=0, anomaly=1, reward_correct=1, reward_incorrect=-1,
                                         action_space=[0, 1], seperator=",", boosted=False),
                 filename="Test/SmallData.csv", verbose=False,
                 scaler=MinMaxScaler(), window=False):
        """
        Initialization of one TimeSeries, should have suitable methods to prepare DataScience
        :param directory: Path to The Datasets
        :param config: Config class Object with specified Reward Settings
        :param filename: The Name of the specified TimeSeries we are working on
        :param verbose: if True, then prints __str__
        """
        self.filename = filename
        self.file = os.path.join(directory + self.filename)
        self.cfg = config
        self.sep = config.seperator
        self.boosted = config.boosted
        self.window = window
        self.scaler = scaler

        self.action_space_n = len(self.cfg.action_space)

        self.timeseries_cursor = -1
        self.timeseries_cursor_init = 0
        self.timeseries_states = []
        self.done = False

        self.statefunction = self._get_state_q
        self.rewardfunction = self._get_reward_q
        self.isdone = False

        self.timeseries_labeled = pd.read_csv(self.file, usecols=[1, 2], header=0, sep=self.sep,
                                              names=['value', 'anomaly'],
                                              encoding="utf-8")

        self.timeseries_unlabeled = pd.read_csv(self.file, usecols=[1], header=0, sep=self.sep,
                                                names=['value'],
                                                encoding="utf-8")

        if verbose:
            print(self.__str__())

    def __str__(self):
        """
        :return: String Representation of the TimeSeriesEnvironment Class, mainly for debug information
        """
        return "TimeSeries from: {}\n Header(labeled):\n {} \nHeader(unlabeled):\n {} \nRows:\n " \
               "{}\nMeanValue:\n {}\nMaxValue:\n {}\nMinValue:\n {}".format(
            self.filename,
            self.timeseries_labeled.head(
                3),
            self.timeseries_unlabeled.head(
                3),
            self.timeseries_labeled.shape[0],
            round(self.timeseries_labeled["value"].mean(), 2),
            round(self.timeseries_labeled["value"].max(), 2),
            round(self.timeseries_labeled["value"].min(), 2))

    def _get_state_q(self, timeseries_cursor, qtableagent=False):
        """
        :param timeseries_cursor: the position where in the TimeSeries we are currently
        :return: The Value of the current position, states with the same value are treated the same way
        """
        if qtableagent:
            return self.timeseries_labeled.index[timeseries_cursor]
        return np.float64(self.timeseries_labeled['value'][timeseries_cursor])

    def _get_reward_q(self, timeseries_cursor, action):
        """
        :param timeseries_cursor: the position where in the TimeSeries we are currently
        :param action: the chosen action (the label we put on the state, Anomaly or Normal)
        :return: Rewards shaped inside the Config File
        """
        if action == self.timeseries_labeled['anomaly'][timeseries_cursor]:
            return self.cfg.reward_correct
        else:
            return self.cfg.reward_incorrect

    # reset the environment
    def reset(self, cursor_init=0):
        self.timeseries_cursor = cursor_init
        self.done = False
        self.normalize_timeseries()
        if not self.window:
            init_state = self.statefunction(self.timeseries_cursor)
        else:
            init_state = self.statefunction(self.timeseries_labeled, self.timeseries_cursor)
        return init_state

    # take a step and gain a reward
    def step(self, action):
        current_state = self.statefunction(timeseries_cursor=self.timeseries_cursor)
        action_in = action
        reward = self.rewardfunction(timeseries_cursor=self.timeseries_cursor, action=action_in)
        self.update_cursor()
        next_state = self.statefunction(timeseries_cursor=self.timeseries_cursor)
        return current_state, action, reward, next_state, self.is_done(self.timeseries_cursor)

    # take a step and gain a reward
    def step_window(self, action):

        current_state = self.statefunction(self.timeseries_labeled, self.timeseries_cursor)
        # 1. get the reward of the action
        reward = self.rewardfunction(self.timeseries_labeled, self.timeseries_cursor, action)

        # 2. get the next state and the done flag after the action
        self.timeseries_cursor += 1

        if self.timeseries_cursor >= self.timeseries_labeled['value'].size:
            self.isdone = 1
            next_state = []
        else:
            self.isdone = 0
            next_state = self.statefunction(self.timeseries_labeled, self.timeseries_cursor)

        return current_state, action, reward, next_state, self.is_done(self.timeseries_cursor)

    def update_cursor(self):
        self.timeseries_cursor += 1

    def is_done(self, cursor):
        if cursor >= len(self.timeseries_labeled) - 1:
            self.done = True
            return True
        else:
            self.done = False
            return False

    def normalize_timeseries(self):
        self.timeseries_labeled["value"] = self.scaler.fit_transform(self.timeseries_labeled[["value"]])

    def isanomaly(self, cursor):
        if self.timeseries_labeled['anomaly'][cursor] == 1:
            return 1
        else:
            return 0

    def get_series(self, labelled=True):
        if labelled:
            return self.timeseries_labeled
        return self.timeseries_unlabeled

    def get_name(self):
        return self.filename

    def __len__(self):
        return self.timeseries_labeled['value'].size


if __name__ == '__main__':
    ts = TimeSeriesEnvironment(verbose=False,
                               config=ConfigTimeSeries(normal=0, anomaly=1, reward_correct=1, reward_incorrect=-1,
                                                       action_space=[0, 1], seperator=",", boosted=False),
                               filename="./Test/SmallData.csv")
    ts.reset()
    count = 0
    ts.normalize_timeseries()
    while count < len(ts.timeseries_labeled) - 1:
        count += 1
    Plots.plot_series(ts.get_series(), name=ts.get_name())
