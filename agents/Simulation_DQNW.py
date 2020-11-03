import os
import tensorflow as tf
# custom modules
from agents.DQNWAgent import DDQNWAgent
from agents.MemoryBuffer import MemoryBuffer
from agents.NeuralNetwork import build_model
from agents.SlidingWindowAgent import SlidingWindowAgent
from environment import WindowStateFunctions
from environment.Config import ConfigTimeSeries
from environment.TimeSeriesModel import TimeSeriesEnvironment
from resources import Utils as utils
from resources.Plots import plot_actions


class Simulator:
    """
    This class is used to train and to test the agent in its environment
    """

    def __init__(self, max_episodes, agent, environment, update_steps):
        """
        Initialize the Simulator with parameters

        :param max_episodes: How many episodes we want to learn, the last episode is used for evaluation
        :param agent: the agent which should be trained
        :param environment: the environment to evaluate and train in
        :param update_steps: the update steps for the Target Q-Network of the Agent
        """
        self.max_episodes = max_episodes
        self.episode = 0
        self.agent = agent
        self.env = environment
        self.update_steps = update_steps

        # information variables
        self.training_scores = []
        self.test_rewards = []
        self.test_actions = []

    def run(self):
        """
        This method is for scheduling training before testing
        :return: True if finished
        """
        while True:
            start = utils.start_timer()
            start_testing = self.__can_test()
            if not start_testing:
                info = self.__training_iteration()
                print("Training episode {} took {} seconds {}".format(self.episode, utils.get_duration(start), info))
                self.__next__()
            if start_testing:
                self.__testing_iteration()
                print("Testing episode {} took {} seconds".format(self.episode, utils.get_duration(start)))
                break
            self.agent.anneal_epsilon()
        plot_actions(self.test_actions[0], self.env.timeseries_labeled)
        return True

    def __training_iteration(self):
        """
        One training iteration is through the complete timeseries, maybe this needs to be changed for
        bigger timeseries datasets.

        :return: Information of the training episode, if update episode or normal episode
        """
        rewards = 0
        state = self.env.reset()
        for idx in range(len(
                self.env.timeseries_labeled)):
            action = self.agent.action(state)
            state, action, reward, nstate, done = self.env.step_window(action)
            rewards += reward
            self.agent.memory.store(state, action, reward, nstate, done)
            state = nstate
            if done:
                self.training_scores.append(rewards)
                break
            # Experience Replay
            if len(self.agent.memory) > self.agent.batch_size:
                self.agent.experience_replay()
        # Target Model Update
        if self.episode % self.update_steps == 0:
            self.agent.update_target_model()
            return "Update Target Model"
        return ""

    def __testing_iteration(self):
        """
        The testing iteration with greedy actions only.
        """
        rewards = 0
        actions = []
        state = self.env.reset()
        self.agent.epsilon = 0
        for idx in range(len(
                self.env.timeseries_labeled)):
            action = self.agent.action(state)
            actions.append(action)
            state, action, reward, nstate, done = self.env.step_window(action)
            rewards += reward
            state = nstate
            if done:
                actions.append(action)
                self.test_rewards.append(rewards)
                self.test_actions.append(actions)
                break

    def __can_test(self):
        """
        :return: True if last episode, False before
        """
        if self.episode >= self.max_episodes:
            return True
        return False

    def __next__(self):
        # increment episode counter
        self.episode += 1


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    # Create the agent
    config = ConfigTimeSeries(seperator=",", window=WindowStateFunctions.SLIDE_WINDOW_SIZE)
    # Test on complete Timeseries from SwAT
    # for subdir, dirs, files in os.walk("../ts_data/A1Benchmark"):
    #     for file in files:
    #         if file.find('.csv') != -1:
    #             env = TimeSeriesEnvironment(verbose=True, filename="./A1Benchmark/{}".format(file), config=config,
    #                                         window=True)
    #             env.statefunction = BatchLearning.SlideWindowStateFuc
    #             env.rewardfunction = BatchLearning.SlideWindowRewardFuc
    #             env.timeseries_cursor_init = BatchLearning.SLIDE_WINDOW_SIZE
    #
    #             dqn = DDQNWAgent(env.action_space_n, 0.001, 0.9, 1, 0, 0.9)
    #             dqn.memory.init_memory(env)
    #             simulation = Simulator(11, dqn, env, 5)
    #             simulation.run()

    env = TimeSeriesEnvironment(verbose=True, filename="./Test/SmallData_1.csv", config=config, window=True)

    env.statefunction = WindowStateFunctions.SlideWindowStateFuc
    env.rewardfunction = WindowStateFunctions.SlideWindowRewardFuc

    agent = SlidingWindowAgent(dqn=build_model(), memory=MemoryBuffer(max=50000, id="sliding_window"), alpha=0.001,
                               gamma=0.99, epsilon=1.0,
                               epsilon_end=0.0, epsilon_decay=0.9, fit_epoch=2, action_space=2, batch_size=512)
    simulation = Simulator(10, agent, env, 5)
    agent.memory.init_memory(env=env)
    simulation.run()
