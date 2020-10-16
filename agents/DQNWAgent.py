import pickle
from collections import deque
import numpy as np
import random

import time
import tensorflow as tf

from agents.MemoryBuffer import MemoryBuffer
from agents.NeuralNetwork import build_model
from environment import BatchLearning
from environment.Config import ConfigTimeSeries
from environment.TimeSeriesModel import TimeSeriesEnvironment

import logging

# Global Variables
from resources.Plots import plot_actions
from resources.Utils import store_object, load_object

EPISODES = 2
TRAIN_END = 0
DISCOUNT_RATE = 0.9
LEARNING_RATE = 0.001
BATCH_SIZE = 512


class DeepQNetwork():
    def __init__(self, states, actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.nS = states
        self.nA = actions
        self.memory = MemoryBuffer(max=50000)
        self.batch_size = BATCH_SIZE
        self.alpha = alpha
        self.gamma = gamma
        # Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = build_model()
        self.model_target = build_model()  # Second (target) neural network
        self.update_target_from_model()  # Update weights
        self.hist = None
        self.loss = []

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA)  # Explore
        action_vals = self.model.predict([state])  # Exploit: Use the NN to predict the correct action from this state
        return np.argmax(action_vals)

    def test_action(self, state):  # Exploit
        action_vals = self.model_target.predict(np.array(state).reshape(1, BatchLearning.SLIDE_WINDOW_SIZE))
        return np.argmax(action_vals)

    def experience_replay(self, batch_size, lstm):
        # get the batches as list so we can build tuples
        minibatch = self.memory.get_exp(batch_size)
        # Execute the experience replay

        # Convert to numpy for speed by vectorization
        x = []
        y = []
        st = np.array(list(list(zip(*minibatch))[0]))
        nst = np.array(list(list(zip(*minibatch))[3]))

        st_predict = self.model.predict(st)  # Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst)
        nst_predict_target = self.model_target.predict(nst)  # Predict from the TARGET

        index = 0
        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            # Predict from state
            nst_action_predict_target = nst_predict_target[index]
            nst_action_predict_model = nst_predict[index]
            if done == True:  # Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = reward
            else:  # Non terminal
                target = reward + self.gamma * nst_action_predict_target[
                    np.argmax(nst_action_predict_model)]  # Using Q to get T is Double DQN
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1
        # Reshape for Keras Fit
        x_reshape = np.array(x)
        y_reshape = np.array(y)
        epoch_count = 1
        self.hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)

    def train(self):
        rewards_training = []
        actions_training = []

        state = env.reset()
        tot_rewards = 0
        for time in range(len(
                env.timeseries_labeled)):
            logging.debug("State at time: {}".format(time) + " " + str(state))
            action = dqn.action(state)
            actions_training.append(action)
            state, action, reward, nstate, done = env.step(action)
            logging.debug(
                "Step Results at {}: S {}, A {}, R {}, S_ {}, D {}".format(env.timeseries_cursor, state, action,
                                                                           reward,
                                                                           nstate, done))
            tot_rewards += reward
            dqn.store(state, action, reward, nstate, done)  # Resize to store in memory to pass to .predict
            state = nstate
            if done:
                # logging.debug("DONE")
                rewards.append(tot_rewards)
                epsilons.append(dqn.epsilon)
                logging.debug("episode: {}/{}, score: {}, e: {}"
                              .format(e, EPISODES, tot_rewards, dqn.epsilon))
                break
            # Experience Replay
            if len(dqn.memory) > batch_size:
                dqn.experience_replay(batch_size)

    def test(self, env, agent, rewards, actions):
        print("TESTING STAGE")
        state, experience, test, test_states, done, total_rewards = self.init_test(env, agent)
        env.reset()
        print("---TEST INIT RESULTS")
        print(state, experience, test, test_states, done, total_rewards)
        for time in range(len(
                env.timeseries_labeled)):  # 200 is when you "solve" the game. This can continue forever as far as I know
            logging.debug("---ITERATION START--{}----".format(time))
            # logging.debug("State at time: {}".format(time) + " " + str(state))
            if done:
                # logging.debug("DONE")
                rewards.append(total_rewards)
                epsilons.append(dqn.epsilon)
                test_states.append(env.statefunction(env.timeseries_labeled, env.timeseries_cursor))
                # test_states.append(env.statefunction(env.timeseries_labeled, env.timeseries_cursor))
                actions_episode.append(action)
                logging.debug("Testing Performance: {}/{}, score: {}"
                              .format(e, EPISODES, total_rewards))
                break
            action = dqn.test_action(state)
            actions_episode.append(action)
            state, action, reward, nstate, done = env.step_window(action)
            # state, action, reward, nstate, done = env.step_window(action)
            test.append("State: {} Action: {} Reward: {} State_: {}".format(state, action, reward, nstate))
            test_states.append(state)
            logging.debug(
                "Step Results at {}: S {}, A {}, R {}, S_ {}, D {}".format(env.timeseries_cursor, state, action,
                                                                           reward,
                                                                           nstate, done))
            # nstate = np.reshape(nstate, [1, nS])
            total_rewards += reward
            state = nstate
            logging.debug("---ITERATION END--{}----".format(time))
        logging.debug("--------")
        logging.debug(len(env.timeseries_labeled))
        logging.debug(len(test_states))
        logging.debug(len(actions_episode))
        logging.debug("--------")
        logging.debug(env.timeseries_labeled)
        logging.debug("--------")
        logging.debug(test_states)
        logging.debug("--------")
        # logging.debug(test)

        actions.append(actions_episode)

    def init_test(self, env, agent):
        """
        :param env: Reset the given environment
        :param agent: Reset the given agents action
        :return: state, action, experience, test, test_states, done, total_rewards
        """
        # Reset the given environment and agent
        state = env.reset()
        print(state)
        agent.epsilon = 0
        experience = []
        test = []
        test_states = []
        done = False
        total_rewards = 0
        return state, experience, test, test_states, done, total_rewards

    def update_target_from_model(self):
        # Update the target model from the base model
        print("Updating Target Model.")
        self.model_target.set_weights(self.model.get_weights())


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    logging.basicConfig(filename='../Logs/window_debug.log', level=logging.DEBUG, filemode='w')
    # Create the agent
    config = ConfigTimeSeries(seperator=",", window=BatchLearning.SLIDE_WINDOW_SIZE)
    # Test on complete Timeseries from SwAT
    env = TimeSeriesEnvironment(verbose=True, filename="./Test/SmallData_1.csv", config=config, window=True)
    # env = TimeSeriesEnvironment(verbose=True, filename="./Attack_FIT101csv.csv", config=config, window=True)
    env.statefunction = BatchLearning.SlideWindowStateFuc
    env.rewardfunction = BatchLearning.SlideWindowRewardFuc

    nS = env.timeseries_labeled.shape[1]  # This is only 4
    nA = env.action_space_n  # Actions
    dqn = DeepQNetwork(nS, nA, LEARNING_RATE, DISCOUNT_RATE, 1, 0, 0.9)
    dqn.memory.init_memory(env)
    batch_size = BATCH_SIZE
    # Training
    rewards = []  # Store rewards for graphing
    epsilons = []  # Store the Explore/Exploit
    actions = []
    test = False
    for e in range(EPISODES):
        print("LEARNING EPISODE: {}".format(e))
        actions_episode = []
        if e >= EPISODES - 1:
            logging.debug("e/E: {}/{}".format(e, EPISODES))
            test = True
        # logging.debug("--------------EPISODE: {}-----------".format(e))
        state = env.reset()
        # logging.debug("TypeOf State at INIT: {}".format(type(state)))
        # logging.debug("State at INIT: {}".format(state) + " Shape at INIT: {}".format(state.shape))
        # state = np.reshape(1, 2)  # Resize to store in memory to pass to .predict
        tot_rewards = 0
        if not test:
            for time in range(len(
                    env.timeseries_labeled)):  # 200 is when you "solve" the game. This can continue forever as far as I know
                # logging.debug("State at time: {}".format(time) + " " + str(state))
                action = dqn.action(state)
                actions_episode.append(action)
                state, action, reward, nstate, done = env.step_window(action)
                # state, action, reward, nstate, done = env.step_window(action)
                logging.debug(
                    "Step Results at {}: S {}, A {}, R {}, S_ {}, D {}".format(env.timeseries_cursor, state, action,
                                                                               reward,
                                                                               nstate, done))
                # nstate = np.reshape(nstate, [1, nS])
                tot_rewards += reward
                dqn.memory.store(state, action, reward, nstate, done)  # Resize to store in memory to pass to .predict
                state = nstate
                if done:
                    # logging.debug("DONE")
                    rewards.append(tot_rewards)
                    epsilons.append(dqn.epsilon)
                    logging.debug("episode: {}/{}, score: {}, e: {}"
                                  .format(e, EPISODES, tot_rewards, dqn.epsilon))
                    break
                # Experience Replay
                if len(dqn.memory) > batch_size:
                    dqn.experience_replay(batch_size, lstm=False)
            if e % 5 == 0:
                dqn.update_target_from_model()
        if test:
            dqn.test(env, dqn, rewards=rewards, actions=actions)
    logging.debug(actions[-1])
    print(dqn.hist)
    plot_actions(actions[-1], env.timeseries_labeled)
