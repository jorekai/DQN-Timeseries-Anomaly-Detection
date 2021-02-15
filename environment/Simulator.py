# custom modules
from collections import deque

from resources import Utils as utils
from resources.Plots import plot_actions
import numpy as np


class Simulator:
    """
    This class is used to train and to test the agent in its environment
    If only Testing, one can specify the testing param
    """

    def __init__(self, max_episodes, agent, environment, update_steps, testing=False):
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
        self.testing = testing

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
                print("Training episode {} took {} seconds, score {}".format(self.episode, utils.get_duration(start),
                                                                             self.training_scores[-1]))
                self.__next__()
            if start_testing:
                self.__testing_iteration()
                print("Testing episode {} took {} seconds".format(self.episode, utils.get_duration(start)))
                break
            self.agent.anneal_epsilon()
        # Appending 5 values because idk -- SORT OUT IF FIXED
        self.test_actions[0].extendleft([0 for x in range(self.env.steps)])
        plot_actions(self.test_actions[0], getattr(self.env, "timeseries_labeled"))
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
                self.env)):
            action = self.agent.action(state)
            nstate, reward, done, [] = self.env.step(action)
            rewards += reward[action]
            self.agent.memory.store(state, action, reward, nstate, done)
            state = nstate[action]
            if done:
                self.training_scores.append(rewards)
                self.agent.update_target_model()
                break

        for i in range(10):
            self.agent.experience_replay()

        # Target Model Update - IF WORKING WITH BIGGER DATASETS SET TO NOT ONLY UPDATE
        # ON END OF EPISODE (END OF FILE)
        # if self.episode % self.update_steps == 0:
        #     self.agent.update_target_model()
        #     return "Update Target Model"
        return ""

    def __testing_iteration(self):
        """
        The testing iteration with greedy actions only.
        """
        rewards = 0
        actions = deque([])
        # self.env.timeseries_cursor_init = 0
        state = self.env.reset()
        self.agent.epsilon = 0
        for idx in range(len(
                self.env)):
            action = self.agent.action(state)
            actions.append(action)
            nstate, reward, done, [] = self.env.step(action)
            rewards += reward[action]
            state = nstate[action]
            if done:
                actions.append(action)
                print("Testing Score: ", np.sum(rewards))
                self.test_rewards.append(rewards)
                self.test_actions.append(actions)
                break

    def __can_test(self):
        """
        :return: True if last episode, False before
        """
        if self.episode >= self.max_episodes or self.testing:
            return True
        return False

    def __next__(self):
        # increment episode counter
        self.episode += 1
