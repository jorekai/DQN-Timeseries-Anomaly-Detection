from environment.Config import ConfigTimeSeries
from environment.BaseEnvironment import TimeSeriesEnvironment
import numpy as np


class WindowStateEnvironment:
    """
    This Environment is sliding a window over the timeseries step by step. The window size can be configured
    The Shape of the states is therefore of the form (1, window_size).
    """

    def __init__(self, environment: TimeSeriesEnvironment, window_size=25):
        """
        Initialize the WindowStateEnvironment and wrapping the base environment
        :param environment: TimeSeriesEnvironment
        :param window_size: int
        """
        self.env = None
        self.window_size = window_size
        if environment is None:
            raise TypeError("Base environment must be instantiated")
        elif isinstance(environment, TimeSeriesEnvironment):
            self.env = environment
        else:
            raise TypeError("Input is not of type TimeSeriesEnvironment")

    def __state(self):
        """
        The Statefunction returning an array of the window states
        :return:
        """
        if self.env.timeseries_cursor >= self.window_size:
            return [self.env.timeseries_labeled['value'][i + 1]
                    for i in range(self.timeseries_cursor - self.window_size, self.timeseries_cursor)]
        else:
            return np.zeros(self.window_size)

    def __reward(self, action):
        """
        The Rewardfunction returning rewards for certain actions in the environment
        :param action: type of action
        :return: arbitrary reward
        """
        if self.timeseries_cursor >= self.window_size:
            if self.timeseries_labeled['anomaly'][self.timeseries_cursor] == 1:
                if action == 0:
                    return -5  # false negative, miss alarm
                else:
                    return 5  # 10      # true positive
            if self.timeseries_labeled['anomaly'][self.timeseries_cursor] == 0:
                if action == 1:
                    return -1
                if action == 0:
                    return 1
        return 0

    def reset(self):
        """
        Reset the current Series to the first Value.
        :return: initial state
        """
        self.env.timeseries_cursor = self.timeseries_cursor_init
        self.normalize_timeseries()
        self.env.done = False
        init_state = self.__state()
        return init_state

    def step(self, action):
        """
        Taking a step inside the base environment with the action input
        :param action: certain action value
        :return: S,A,R,S_,D tuple
        """
        current_state = self.__state()
        reward = self.__reward(action)

        self.update_cursor()
        next_state = self.__state()

        return current_state, action, reward, next_state, self.is_done()

    def __len__(self):
        """
        Length of the current Timeseries
        :return: int
        """
        return len(self.env)

    def __getattr__(self, item):
        """
        Get the attribute of the base environment
        :param item: String of field key
        :return: attribute item of the base environment
        """
        return getattr(self.env, item)


if __name__ == '__main__':
    env = WindowStateEnvironment(
        TimeSeriesEnvironment(verbose=True, filename="./Test/SmallData.csv", config=ConfigTimeSeries()))
    env.reset()
    idx = 1
    while True:
        idx += 1
        s, a, r, s_, d = env.step(1)
        print(s, a, r, s_, d)
        if d:
            print(idx)
            break
