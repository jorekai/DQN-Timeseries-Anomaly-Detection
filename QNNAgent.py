import itertools

import matplotlib.pyplot as plt
import random as rnd
from collections import namedtuple, deque
import numpy as np
import random
import doctest

import tensorflow as tf
from tensorflow import keras

import BatchLearning
from Config import ConfigTimeSeries
from TimeSeriesModel import TimeSeriesEnvironment

import seaborn as sb
from sklearn.preprocessing import MinMaxScaler
import logging

# Global Variables
EPISODES = 26
TRAIN_END = 0


# Hyper Parameters
def discount_rate():  # Gamma
    return 0.9


def learning_rate():  # Alpha
    return 0.001


def batch_size():  # Size of the batch used in the experience replay
    return 256


def plot_actions(actions, series):
    plt.figure(figsize=(15, 7))
    plt.plot(series.index, actions, label="Actions", linestyle="solid")
    plt.plot(series.index, series["anomaly"], label="True Label", linestyle="dotted")
    plt.plot(series.index, series["value"], label="Series", linestyle="dashed")
    plt.legend()
    # sb.lineplot(
    #     data=actions,
    # ).set_title("Actions vs Series")
    plt.ylabel('Reward Sum')
    plt.show()


def plot_learn(data):
    plt.figure(figsize=(15, 7))
    sb.lineplot(
        data=data,
    ).set_title("Learning")
    plt.ylabel('Reward Sum')
    plt.show()


def plot_reward(result):
    plt.figure(figsize=(15, 7))
    sb.lineplot(
        data=result,
    ).set_title("Reward Random vs Series")
    plt.ylabel('Reward Sum')
    plt.show()


class DeepQNetwork():
    def __init__(self, states, actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=5000)
        self.batch_size = batch_size()
        self.alpha = alpha
        self.gamma = gamma
        # Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.model_target = self.build_model()  # Second (target) neural network
        self.update_target_from_model()  # Update weights
        self.hist = None
        self.loss = []

    def build_model(self):
        model = keras.Sequential()  # linear stack of layers https://keras.io/models/sequential/
        model.add(keras.layers.Dense(24, input_dim=1, activation='relu'))  # [Input] -> Layer 1
        model.add(keras.layers.Dense(48, activation='relu'))  # Layer 3 -> [output]
        # model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(48, activation='relu'))  # Layer 3 -> [output]
        model.add(keras.layers.Dense(self.nA, activation='linear'))  # Layer 3 -> [output]
        model.compile(loss='mean_squared_error',  # Loss function: Mean Squared Error
                      optimizer=keras.optimizers.RMSprop(
                          lr=self.alpha))  # Optimaizer: Adam (Feel free to check other options)
        return model

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA)  # Explore
        # logging.debug("State at ACTION: {}".format([state]) + " Shape at ACTION: {}".format(state.shape))
        # logging.debug("TypeOf State at ACTION: {}".format(type(state)))
        action_vals = self.model.predict([state])  # Exploit: Use the NN to predict the correct action from this state
        # logging.debug("Action Values: " + str(action_vals))
        return np.argmax(action_vals)

    def test_action(self, state):  # Exploit
        action_vals = self.model_target.predict([state])
        # logging.debug("ARGMAX: {} at State: {}".format(action_vals, state))
        # logging.debug("ARGMAX 0 : {} 01: {} 10. {}".format(action_vals[0], action_vals[0][1], action_vals[1][0]))
        return np.argmax(action_vals)

    def store(self, state, action, reward, nstate, done):
        # Store the experience in memory
        # logging.debug("STORING IN MEMORY: {}".format((state, action, reward, nstate, done)))
        self.memory.append((state, action, reward, nstate, done))

    def init_memory(self, env):
        # resetting environment once
        env.reset()
        while True:
            # break if memory is full
            if len(self.memory) >= self.memory.maxlen:
                break
            # check if we need to reset env
            if env.is_done(env.timeseries_cursor):
                env.reset()
            # get random action
            action = random.randrange(self.nA)
            # take step in env and append
            state, action, reward, nstate, done = env.step(action)
            # logging.debug("INIT MEMORY: {}".format((state, action, reward, nstate, done)))
            self.store(state, action, reward, nstate, done)
        print("Memory is full")

    def get_exp(self, batch_size):
        return [self.memory.popleft() for _i in range(batch_size)]  # Popping from the Memory Queue

    def experience_replay(self, batch_size, lstm):
        # get the batches as list so we can build tuples
        batch_list = self.get_exp(batch_size)
        # print(batch_list)
        # get states and next state as [[state_1],...,[state_batch_size]]
        sars_list = np.array(batch_list[:])
        state_list = []
        next_state_list = []
        next_state_list_tnet = []
        for sars in sars_list:
            state_list.append([sars[0]])
            next_state_list.append([sars[3]])
        # if we are using an lstm we need different input shapes (Target Network not yet)
        if lstm:
            state_list = np.asarray(state_list).reshape(1, batch_size, 1)
            next_state_list = np.asarray(next_state_list).reshape(1, batch_size, 1)
        else:
            state_list = np.asarray(state_list)
            next_state_list = np.asarray(next_state_list)
            next_state_list_tnet = next_state_list  # Predict from the TARGET Network
        # get predictions as [[prediction(0), prediction(1)],...,[prediction(0), prediction(1)]]
        logging.debug("STATE LIST: {} ".format(state_list.shape))
        logging.debug("NEXT STATE LIST: {} ".format(next_state_list.shape))
        pred_states = self.model.predict(state_list)
        pred_next_states = self.model.predict(next_state_list)
        pred_next_states_tnet = self.model_target.predict(next_state_list_tnet)

        # preparing batch input for neural network function approximator
        inputs = []
        targets = []
        debug_actions = []
        idx = 0
        for state, action, reward, nstate, done in batch_list:
            inputs.append(state)

            if lstm:
                nstate_predictions = pred_next_states[0][idx]
            else:
                nstate_predictions = pred_next_states[idx]
                nstate_predictions_target = pred_next_states_tnet[idx]  # TARGET NETWORK
            logging.debug("PREDICTION TARGET: {}".format(nstate_predictions_target))
            logging.debug("AMAX NET: {}".format(np.amax(nstate_predictions)))
            if done:
                target = reward
            elif not done:
                # target = reward + self.gamma * np.amax(nstate_predictions) NOT TARGET
                target = reward + self.gamma * nstate_predictions_target[
                    np.argmax(nstate_predictions)]  # TARGET NETWORK
            if lstm:
                target_f = pred_states[0][idx]
            else:
                target_f = pred_states[idx]
            # logging.debug("PREDICTION STATE: {}".format(target_f))
            target_f[action] = target
            targets.append(target_f)
            debug_actions.append(action)
            idx += 1
        if lstm:
            inputs = np.array(inputs).reshape(1, batch_size, 1)
            targets = np.array(targets).reshape(1, batch_size, 2)
        else:
            inputs = np.array(inputs).reshape(batch_size, 1)
            targets = np.array(targets)
        logging.debug("INPUTS NN: {}".format(inputs.shape))
        logging.debug("ACTION BATCH IDX: {}".format(debug_actions))
        logging.debug("TARGETS NN: {}".format(targets.shape))
        self.hist = self.model.fit(inputs, targets, epochs=10, verbose=0)
        # plt.title("Loss of Batch")
        # plt.plot(self.hist.history['loss'], label='train')
        # plt.show()

    # def experience_replay(self, batch_size):
    #     # Execute the experience replay
    #     minibatch = self.get_exp(batch_size)
    #     print("LEN MEM: {}".format(len(self.memory)))
    #     # Convert to numpy for speed by vectorization
    #     x = []
    #     y = []
    #     np_array = np.array(minibatch)
    #     st = np.zeros((0, self.nS))  # States
    #     nst = np.zeros((0, self.nS))  # Next States
    #     logging.debug(st)
    #     logging.debug(nst)
    #     logging.debug(np_array)
    #     for i in range(len(np_array)):  # Creating the state and next state np arrays
    #         logging.debug(np_array[i, 0])
    #         logging.debug(np_array[i, 3])
    #         st = np.append(st, np_array[i, 0])
    #         nst = np.append(nst, np_array[i, 3])
    #     logging.debug("State: {} Shape: {}".format(st, st.shape))
    #     logging.debug("Next State: {} Shape: {}".format(nst, nst.shape))
    #     st_predict = self.model.predict(
    #         st)  # Here is the speedup! I can predict on the ENTIRE batch
    #     nst_predict = self.model.predict(nst)
    #     index = 0
    #     for state, action, reward, nstate, done in minibatch:
    #         x.append(state)
    #         # Predict from state
    #         nst_action_predict_model = nst_predict[index]
    #         if done == True:  # Terminal: Just assign reward much like {* (not done) - QB[state][action]}
    #             target = reward
    #         else:  # Non terminal
    #             target = reward + self.gamma * np.amax(nst_action_predict_model)
    #         target_f = st_predict[index]
    #         target_f[action] = target
    #         y.append(target_f)
    #         index += 1
    #     # Reshape for Keras Fit
    #     x_reshape = np.array(x).reshape(batch_size, 1)
    #     y_reshape = np.array(y)
    #     logging.debug("Input Reshaped: {} / Length: {} / Shape: {}".format(x_reshape, len(x_reshape), x_reshape.shape))
    #     logging.debug("Target Reshaped: {} / Length: {} / Shape: {}".format(y_reshape, len(y_reshape), y_reshape.shape))
    #     epoch_count = 10  # Epochs is the number or iterations
    #     self.hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
    #     # Graph Losses
    #     for i in range(epoch_count):
    #         self.loss.append(self.hist.history['loss'][i])
    #     # Decay Epsilon
    #     # if self.epsilon > self.epsilon_min:
    #     #    self.epsilon *= self.epsilon_decay

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
        state, action, experience, test, test_states, done, total_rewards = self.init_test(env, agent)
        for time in range(len(
                env.timeseries_labeled)):  # 200 is when you "solve" the game. This can continue forever as far as I know
            logging.debug("---ITERATION START--{}----".format(time))
            # logging.debug("State at time: {}".format(time) + " " + str(state))
            if done:
                # logging.debug("DONE")
                rewards.append(total_rewards)
                epsilons.append(dqn.epsilon)
                test_states.append(env.statefunction(env.timeseries_cursor))
                # test_states.append(env.statefunction(env.timeseries_labeled, env.timeseries_cursor))
                actions_episode.append(action)
                logging.debug("Testing Performance: {}/{}, score: {}"
                              .format(e, EPISODES, total_rewards))
                break
            actions_episode.append(action)
            action = dqn.test_action(state)
            state, action, reward, nstate, done = env.step(action)
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
        action = agent.test_action(state)
        agent.epsilon = 0
        experience = []
        test = []
        test_states = []
        done = False
        total_rewards = 0
        return state, action, experience, test, test_states, done, total_rewards

    def update_target_from_model(self):
        # Update the target model from the base model
        self.model_target.set_weights(self.model.get_weights())


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    # logging.basicConfig(filename='./Logs/debug.log', level=logging.DEBUG, filemode='w')
    # Create the agent
    config = ConfigTimeSeries(seperator=",", window=True)
    # Test on complete Timeseries from SwAT
    env = TimeSeriesEnvironment(verbose=True, filename="./Test/SmallData_1.csv", config=config)
    env.statefunction = BatchLearning.SlideWindowStateFuc
    env.rewardfunction = BatchLearning.SlideWindowRewardFuc
    # env = TimeSeriesEnvironment(verbose=True, filename="./Attack_FIT101csv.csv", config=config)

    nS = env.timeseries_labeled.shape[1]  # This is only 4
    nA = env.action_space_n  # Actions
    dqn = DeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1, 0, 0.9)
    dqn.init_memory(env)
    batch_size = batch_size()
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
                state, action, reward, nstate, done = env.step(action)
                # state, action, reward, nstate, done = env.step_window(action)
                logging.debug(
                    "Step Results at {}: S {}, A {}, R {}, S_ {}, D {}".format(env.timeseries_cursor, state, action,
                                                                               reward,
                                                                               nstate, done))
                # nstate = np.reshape(nstate, [1, nS])
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
                    dqn.experience_replay(batch_size, lstm=False)
            if e % 5 == 0:
                dqn.update_target_from_model()
        if test:
            dqn.test(env, dqn, rewards=rewards, actions=actions)
    logging.debug(actions[-1])
    print(dqn.hist)
    plot_actions(actions[-1], env.timeseries_labeled)
