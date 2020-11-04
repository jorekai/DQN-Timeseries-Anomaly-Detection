from environment.Config import ConfigTimeSeries
from environment.TimeSeriesModel import TimeSeriesEnvironment
import numpy as np


class WindowStateEnvironment:
    def __init__(self, environment: TimeSeriesEnvironment, window_size=25):
        self.env = environment
        self.window_size = window_size

    SLIDE_WINDOW_SIZE = 25  # size of the slide window for SLIDE_WINDOW state and reward functions

    def __state(self):
        if self.env.timeseries_cursor >= self.window_size:
            return [self.env.timeseries_labeled['value'][i + 1]
                    for i in range(self.env.timeseries_cursor - self.window_size, self.env.timeseries_cursor)]
        else:
            return np.zeros(self.window_size)

    def __reward(self, action):
        if self.env.timeseries_cursor >= self.window_size:
            sum_anomaly = np.sum(self.env.timeseries_labeled['anomaly']
                                 [self.env.timeseries_cursor - self.window_size + 1:self.env.timeseries_cursor + 1])
            if sum_anomaly == 0:
                if action == 0:
                    return 1  # 0.1      # true negative
                else:
                    return -1  # 0.5     # false positive, error alarm

            if sum_anomaly > 0:
                if action == 0:
                    return -5  # false negative, miss alarm
                else:
                    return 5  # 10      # true positive
        else:
            return 0

    def reset(self):
        """
        Reset the current Series to the first Value.
        :return: initial state
        """
        self.env.timeseries_cursor = self.env.timeseries_cursor_init
        self.env.normalize_timeseries()
        self.env.done = False
        init_state = self.__state()
        return init_state

    def step(self, action):
        current_state = self.__state()
        # 1. get the reward of the action
        reward = self.__reward(action)

        # 2. get the next state and the done flag after the action
        self.env.update_cursor()

        if self.env.timeseries_cursor >= len(self.env):
            self.env.isdone = 1
            next_state = []
        else:
            self.env.isdone = 0
            next_state = self.__state()

        return current_state, action, reward, next_state, self.env.is_done(self.env.timeseries_cursor)

    def __len__(self):
        return len(self.env)

    def __getattr__(self, item):
        return getattr(self.env, item)


if __name__ == '__main__':
    config = ConfigTimeSeries(seperator=",", window=1)
    env = WindowStateEnvironment(
        TimeSeriesEnvironment(verbose=True, filename="./Test/SmallData.csv", config=config, window=True))
    env.reset()
    idx = 1
    while True:
        idx += 1
        s, a, r, s_, d = env.step(1)
        print(s, a, r, s_, d)
        if d:
            print(idx)
            break