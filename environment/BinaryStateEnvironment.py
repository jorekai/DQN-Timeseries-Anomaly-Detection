import numpy as np

from environment.Config import ConfigTimeSeries
from environment.BaseEnvironment import TimeSeriesEnvironment


class BinaryStateEnvironment:
    """
    The class environment is a wrapper around the TimeSeriesEnvironment. We are changing the interface of the environment by
    overriding the methods to use for RL-Computations.
    """

    def __init__(self, environment: TimeSeriesEnvironment, steps=5):
        """
        Initializing the BinaryStateEnv by wrapping the BaseEnvironment
        :param environment: TimeSeriesEnvironment
        """
        self.env = None
        self.steps = steps
        if environment is None:
            raise TypeError("Base environment must be instantiated")
        elif isinstance(environment, TimeSeriesEnvironment):
            self.env = environment
        else:
            raise TypeError("Input is not of type TimeSeriesEnvironment")
        self.env.timeseries_cursor_init = steps

    def __state(self, previous_state=[]):
        """
        :param action: Our state Value consists of the timeseries value and the previous taken action: [0.123, 1] or [0.123, 0]
        :return: The Value of the current position, states with the same value are treated the same way
        """
        if self.timeseries_cursor == self.steps:
            state = []
            for i in range(self.timeseries_cursor):
                state.append([self.timeseries_labeled['value'][i], 0])

            state.pop(0)
            state.append([self.timeseries_labeled['value'][self.timeseries_cursor], 1])

            return np.array(state, dtype='float32')

        if self.timeseries_cursor > self.steps:
            state0 = np.concatenate((previous_state[1:self.steps],
                                     [[self.timeseries_labeled['value'][self.timeseries_cursor], 0]]))
            state1 = np.concatenate((previous_state[1:self.steps],
                                     [[self.timeseries_labeled['value'][self.timeseries_cursor], 1]]))

            return np.array([state0, state1], dtype='float32')

    def __reward(self):
        """
        :param action: The reward depends on the action we are taking in the environment
        :return: action-reward value
        """
        if self.timeseries_cursor >= self.steps:
            if self.timeseries_labeled['anomaly'][self.timeseries_cursor] == 0:
                return [1, -1]

            if self.timeseries_labeled['anomaly'][self.timeseries_cursor] == 1:
                return [-5, 5]
        else:
            return [0, 0]

    def reset(self):
        self.env.timeseries_cursor = self.timeseries_cursor_init
        self.normalize_timeseries()
        self.env.done = False
        # 2. return the first state, containing the first element of the time series
        self.env.timeseries_states = self.__state()

        return self.timeseries_states

    def step(self, action):
        # assert(action in action_space)
        # assert(self.timeseries_curser >= 0)

        # 1. get the reward of the action
        reward = self.__reward()

        # 2. get the next state and the done flag after the action
        self.update_cursor()

        if self.timeseries_cursor >= self.timeseries_labeled['value'].size:
            state = np.array([self.timeseries_states, self.timeseries_states])
        else:
            state = self.__state(self.timeseries_states)

        if len(np.shape(state)) > len(np.shape(self.timeseries_states)):
            self.env.timeseries_states = state[action]
        else:
            self.env.timeseries_states = state

        return state, reward, self.is_done(), []

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
        state, reward, done, [] = env.step(1)
        print(state, reward, done, [])
        print(state.shape)
        if done:
            print(idx)
            break
