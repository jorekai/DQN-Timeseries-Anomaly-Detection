import numpy as np
import sys

# import the library in the sub-folder env
from environment.Config import ConfigTimeSeries

if "../" not in sys.path:
    sys.path.append("../../")
from environment.TimeSeriesModel import TimeSeriesEnvironment

SLIDE_WINDOW_SIZE = 2  # size of the slide window for SLIDE_WINDOW state and reward functions


def SlideWindowStateFuc(timeseries, timeseries_cursor, timeseries_states=None, action=None):
    if timeseries_cursor >= SLIDE_WINDOW_SIZE:
        return [timeseries['value'][i + 1]
                for i in range(timeseries_cursor - SLIDE_WINDOW_SIZE, timeseries_cursor)]
    else:
        return np.zeros(SLIDE_WINDOW_SIZE)


def SlideWindowRewardFuc(timeseries, timeseries_cursor, action):
    if timeseries_cursor >= SLIDE_WINDOW_SIZE:
        if np.sum(timeseries['anomaly']
                  [timeseries_cursor - SLIDE_WINDOW_SIZE + 1:timeseries_cursor + 1]) == 0:
            if action == 0:
                return 1  # 0.1      # true negative
            else:
                return -1  # 0.5     # false positive, error alarm

        if np.sum(timeseries['anomaly']
                  [timeseries_cursor - SLIDE_WINDOW_SIZE + 1:timeseries_cursor + 1]) > 0:
            if action == 0:
                return -5  # false negative, miss alarm
            else:
                return 5  # 10      # true positive
    else:
        return 0


if __name__ == '__main__':
    ts = TimeSeriesEnvironment(verbose=False,
                               config=ConfigTimeSeries(normal=0, anomaly=1, reward_correct=1, reward_incorrect=-1,
                                                       action_space=[0, 1], seperator=",", boosted=False, window=2),
                               window=True)
    ts.timeseries_cursor = ts.timeseries_cursor_init
    ts.normalize_timeseries()
    ts.statefunction = SlideWindowStateFuc
    ts.rewardfunction = SlideWindowRewardFuc
    ts.reset()
    count = 0
    while count <= len(ts.timeseries_labeled) - 1:
        count += 1
        state, action, reward, next_state, done = ts.step_window(1)
        print(state, action, reward, next_state, done)
    print(count)
