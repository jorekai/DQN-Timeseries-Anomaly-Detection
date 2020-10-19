import os

import tensorflow as tf
from agents.DQNWAgent import DDQNWAgent
from environment import BatchLearning
from environment.Config import ConfigTimeSeries
from environment.TimeSeriesModel import TimeSeriesEnvironment
from resources import Utils as utils
from resources.Plots import plot_actions


class Simulator:
    """
    This class is used to train and to test the agent in its environment
    """

    def __init__(self, max_episodes, agent, environment, update_steps):
        self.max_episodes = max_episodes
        self.episode = 1
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
            start_testing = self.can_test()
            if not start_testing:
                info = self.training_iteration()
                print("Training episode {} took {} seconds {}".format(self.episode, utils.get_duration(start), info))
                self.next()
            if start_testing:
                self.testing_iteration()
                print("Testing episode {} took {} seconds".format(self.episode, utils.get_duration(start)))
                break
            self.agent.anneal_eps()
        plot_actions(self.test_actions[0], self.env.timeseries_labeled)
        return True

    def can_test(self):
        if self.episode >= self.max_episodes:
            return True
        return False

    def next(self):
        self.episode += 1

    def training_iteration(self):
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
                self.agent.experience_replay(self.agent.batch_size, lstm=False)
        # Target Model Update
        if self.episode % self.update_steps == 0:
            self.agent.update_target_from_model()
            return "Update Target Model"
        return ""

    def testing_iteration(self):
        rewards = 0
        actions = []
        state = self.env.reset()
        self.agent.epsilon = 0
        for idx in range(len(
                self.env.timeseries_labeled)):
            action = self.agent.action(state)
            actions.append(action)
            # print("At Timestamp: " + str(idx))
            state, action, reward, nstate, done = self.env.step_window(action)
            # print("State: \n " + str(state))
            # print("Action: " + str(action))
            # print("Reward: " + str(reward))

            rewards += reward
            state = nstate
            if done:
                actions.append(action)
                self.test_rewards.append(rewards)
                self.test_actions.append(actions)
                break


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    # Create the agent
    config = ConfigTimeSeries(seperator=",", window=BatchLearning.SLIDE_WINDOW_SIZE)
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

    env.statefunction = BatchLearning.SlideWindowStateFuc
    env.rewardfunction = BatchLearning.SlideWindowRewardFuc
    env.timeseries_cursor_init = BatchLearning.SLIDE_WINDOW_SIZE

    dqn = DDQNWAgent(env.action_space_n, 0.001, 0.9, 1, 0, 0.9)
    dqn.memory.init_memory(env)
    simulation = Simulator(11, dqn, env, 5)
    simulation.run()
