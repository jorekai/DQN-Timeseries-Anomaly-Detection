import numpy as np


def defaultState(timeseries, cursor, action):
    """
    :param timeseries:
    :param cursor: the position where in the TimeSeries we are currently
    :return: The Value of the current position, states with the same value are treated the same way
    """
    state = np.asarray([np.float64(timeseries['value'][cursor]), 1 if action == 1 else 0])
    return state


def defaultReward(state, timeseries, cursor, action):
    if state[1] == 1 and timeseries['anomaly'][cursor] == 1:
        if action == 1:
            return 1
        if action == 0:
            return -1
    if state[1] == 1 and timeseries['anomaly'][cursor] == 0:
        if action == 1:
            return -1
        if action == 0:
            return 1
    if state[1] == 0 and timeseries['anomaly'][cursor] == 1:
        if action == 1:
            return 1
        if action == 0:
            return -1
    if state[1] == 0 and timeseries['anomaly'][cursor] == 0:
        if action == 1:
            return -1
        if action == 0:
            return 1
