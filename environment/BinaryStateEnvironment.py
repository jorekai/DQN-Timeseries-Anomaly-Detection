import numpy as np

from environment.Config import ConfigTimeSeries
from environment.BaseEnvironment import TimeSeriesEnvironment


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
        self.env = None
        if environment is None:
            raise TypeError("Base environment must be instantiated")
        elif isinstance(environment, TimeSeriesEnvironment):
            self.env = environment
        else:
            raise TypeError("Input is not of type TimeSeriesEnvironment")

    def __state(self, action):
        """
        :param action: Our state Value consists of the timeseries value and the previous taken action: [0.123, 1] or [0.123, 0]
        :return: The Value of the current position, states with the same value are treated the same way
        """
        state = np.asarray(
            [self.timeseries_labeled['value'][self.timeseries_cursor], 1 if action == 1 else 0], dtype=float)
        return state

    def __reward(self, action):
        """
        :param action: The reward depends on the action we are taking in the environment
        :return: action-reward value
        """
        state = self.__state(action)
        label = self.timeseries_labeled['anomaly'][self.timeseries_cursor]
        if state[1] == 1 and label == 1:  # previous action was "anomaly" and current label is "anomaly"
            if action == 1:  # we did label correctly so true positive
                return 5
            if action == 0:  # we did not label correctly so false negative
                return -5
        if state[1] == 1 and label == 0:  # previous action was "anomaly" and current label is "normal"
            if action == 1:  # we did not label correctly so false positive
                return -1
            if action == 0:  # we did label correctly so true negative
                return 1
        if state[1] == 0 and label == 1:  # previous action was "normal" and current label is "anomaly"
            if action == 1:  # we did label correctly so true positive
                return 5
            if action == 0:  # we did not label correctly so false negative
                return -5
        if state[1] == 0 and label == 0:  # previous action was "normal" and current label is "normal"
            if action == 1:  # we did not label correctly so false positive
                return -1
            if action == 0:  # we did label correctly so true negative
                return 1

    def reset(self):
        """
        Reset the current Series to the first Value.
        :return: initial state, starting action was 0
        """
        self.env.timeseries_cursor = self.timeseries_cursor_init
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
        self.update_cursor()
        next_state = self.__state(action)
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
    env = BinaryStateEnvironment(
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
