import numpy as np

from environment.Config import ConfigTimeSeries
from environment.TimeSeriesModel import TimeSeriesEnvironment


class BinaryStateEnvironment:
    """
    The class environment is a wrapper around the TimeSeriesEnvironment. We are changing the interface of the environment by
    overriding the methods to use for RL-Computations.
    """

    def __init__(self, environment: TimeSeriesEnvironment):
        """
        Initializing the BinaryStateEnv by wrapping the BaseEnvironment
        :param environment: TimeSeriesEnvironment
        """
        self.env = environment

    def __state(self, action):
        """
        :param timeseries:
        :param cursor: the position where in the TimeSeries we are currently
        :return: The Value of the current position, states with the same value are treated the same way
        """
        state = np.asarray(
            [np.float64(self.env.timeseries_labeled['value'][self.env.timeseries_cursor]), 1 if action == 1 else 0])
        return state

    def __reward(self, action):
        """
        :param action: The reward depends on the action we are taking in the environment
        :return: action-reward value
        """
        state = self.__state(action)
        label = self.env.timeseries_labeled['anomaly'][self.env.timeseries_cursor]
        if state[1] == 1 and label == 1:
            if action == 1:
                return 5
            if action == 0:
                return -5
        if state[1] == 1 and label == 0:
            if action == 1:
                return -1
            if action == 0:
                return 1
        if state[1] == 0 and label == 1:
            if action == 1:
                return 5
            if action == 0:
                return -5
        if state[1] == 0 and label == 0:
            if action == 1:
                return -1
            if action == 0:
                return 1

    def reset(self):
        """
        Reset the current Series to the first Value.
        :return: initial state
        """
        self.env.timeseries_cursor = self.env.timeseries_cursor_init
        self.env.normalize_timeseries()
        self.env.done = False
        init_state = self.__state(0)
        return init_state

    def step(self, action):
        """
        Taking a step inside the current Series.
        :param action: a valid action value
        :return: (S,A,R,S',Done)
        """
        current_state = self.__state(action)
        reward = self.__reward(action)
        self.env.update_cursor()
        next_state = self.__state(action)
        return current_state, action, reward, next_state, self.env.is_done(self.env.timeseries_cursor)

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
    config = ConfigTimeSeries(seperator=",", window=1)
    env = BinaryStateEnvironment(
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
