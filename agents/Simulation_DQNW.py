from agents.DQNWAgent import DDQNWAgent
from environment import BatchLearning
from environment.Config import ConfigTimeSeries
from environment.TimeSeriesModel import TimeSeriesEnvironment


class Simulator:
    """
    This class is used to train and to test the agent in its environment
    """

    def __init__(self, max_episodes, agent, environment):
        self.max_episodes = max_episodes
        self.episode = 1
        self.agent = agent
        self.environment = environment

    def run(self):
        """
        This method is for scheduling training before testing
        :return: True if finished
        """
        while True:
            if self.train():
                print("Training {}".format(self.episode))
                self.next()
            if not self.train():
                print("Testing {}".format(self.episode))
                break
        return True

    def train(self):
        if self.episode >= self.max_episodes:
            return False
        return True

    def next(self):
        self.episode += 1


if __name__ == '__main__':
    config = ConfigTimeSeries(seperator=",", window=BatchLearning.SLIDE_WINDOW_SIZE)
    env = TimeSeriesEnvironment(verbose=True, filename="./Test/SmallData_1.csv", config=config, window=True)
    env.statefunction = BatchLearning.SlideWindowStateFuc
    env.rewardfunction = BatchLearning.SlideWindowRewardFuc
    env.timeseries_cursor_init = BatchLearning.SLIDE_WINDOW_SIZE
    dqn = DDQNWAgent(env.action_space_n, 0.001, 0.9, 1, 0, 0.9)
    simulation = Simulator(10, dqn, env)
    simulation.run()
