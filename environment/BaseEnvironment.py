import pandas as pd
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

    def __state(self):
        """
        Every environment should implement a state function to return the current state
        :return: current state
        """
        pass

    def __reward(self, action):
        """
        Every environment should implement a reward function to return rewards for the current state and action chosen
        :return: reward
        """
        pass

    def reset(self):
        """
        Every environment should implement a reset method to return a initial State
        :return: init state
        """
        pass

    def step(self, action):
        """
        Every environment should implement a step method to return the S,A,R,S_,D tuples
        :return: state, action, reward, next_state, done
        """
        pass

    def update_cursor(self):
        """
        Increment the cursor
        :return: void
        """
        self.timeseries_cursor += 1

    def is_done(self, cursor):
        """
        Are we done with the current timeseries?
        :param cursor: position in dataframe
        :return: boolean
        """
        if cursor >= len(self.timeseries_labeled) - 1:
            self.done = True
            return True
        else:
            self.done = False
            return False

    def normalize_timeseries(self):
        """
        Min,Max Normalization between 0,1
        :return: void
        """
        self.timeseries_labeled["value"] = self.scaler.fit_transform(self.timeseries_labeled[["value"]])

    def is_anomaly(self, cursor):
        """
        Is the current position a anomaly?
        :param cursor: position in dataframe
        :return: boolean
        """
        if self.timeseries_labeled['anomaly'][cursor] == 1:
            return True
        else:
            return False

    def get_series(self, labelled=True):
        """
        Return the current series labelled or unlabelled
        :param labelled: boolean
        :return: pandas dataframe
        """
        if labelled:
            return self.timeseries_labeled
        return self.timeseries_unlabeled

    def get_name(self):
        """
        Get the current Filename if needed
        :return: String
        """
        return self.filename

    def __len__(self):
        """
        Get the length of the current dataframe
        :return: int
        """
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
