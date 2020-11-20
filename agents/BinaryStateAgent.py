import numpy as np
import random as pyrand
from agents.AbstractAgent import AbstractAgent
from resources.SafetyChecks import verifyBatchShape


class BinaryStateAgent(AbstractAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def action(self, state):
        """
        The action function chooses according to epsilon-greedy policy actions in our environment
        :param state: The state, representing the Sliding Window of datapoints in our Timeseries
        :return: returns an action in our action space ∈ (0, 1)
        """
        if np.random.rand() <= self.epsilon:
            return pyrand.randrange(self.action_space)  # Explore Actions by percentage epsilon
        if self.epsilon == 0:  # greedy action choice for testing the agents performance in our environment
            # action_values = self.target_dqn.predict(np.array(state).reshape(1, 2))
            lstm_input = np.expand_dims(state, axis=0)
            action_values = self.dqn.predict(lstm_input)
            print(action_values)
        else:
            # print(state.shape)
            # comment in for not lstm
            # action_values = self.dqn.predict(np.array(state).reshape(1, 2))
            lstm_input = np.expand_dims(state, axis=0)
            action_values = self.dqn.predict(lstm_input)
        return np.argmax(action_values)

    def experience_replay(self):
        """
        Experience Replay is necessary for our off-policy algorithm to derive the optimal policy out of the
        Transition-Set:(state, action, reward, next_state, done) which is of size: batch_size.
        DQN: The algorithm estimates the return (total discounted future reward) for state-action pairs assuming a greedy
        policy was followed despite the fact that it's not following a greedy policy.

        :param batch_size: number of transitions to be fitted at once
        """
        # get a sample from the memory buffer
        minibatch = self.memory.get_exp(self.batch_size)

        # print(minibatch)

        # create default input arrays for the fitting of the model
        x = []
        y = []

        state_predict, nextstate_predict, nextstate_predict_target = self.predict_on_batch(minibatch)

        index = 0
        # we must iterate through the batch and calculate all target values for the models q values
        for state, action, reward, nextstate, done in minibatch:
            x.append(state)
            # Predict from state
            nextstate_action_predict_target = nextstate_predict_target[index]
            nextstate_action_predict_model = nextstate_predict[index]
            # now we calculate the target Q-Values for our input batch of transitions
            if done:  # Terminal State: Just assign reward
                target = reward
            else:  # Non terminal State: Update the Q-Values
                target = reward + self.gamma * nextstate_action_predict_target[
                    np.argmax(nextstate_action_predict_model)]  # Using Q-Value from our Target Network follows DDQN
            # print("TARGET")
            # print(target)
            target_f = state_predict
            # print(target_f)
            # target_f[action] = target
            y.append(target)
            # if 5 in reward or -5 in reward:
            # print(state, target)
            index += 1
        # Reshape for Keras Fitting Function
        x_reshape = np.array(x)
        y_reshape = np.array(y)
        # print(x_reshape)
        # print(y_reshape)
        self.hist = self.dqn.fit(x_reshape, y_reshape, epochs=self.fit_epoch, verbose=0)

    def predict_on_batch(self, batch):
        """
        Helper method to get the predictions on out batch of transitions in a shape that we need
        :param batch: n batch size
        :return: NN predictions(state), NN predictions(next_state), Target NN predictions(next_state)
        """
        # Convert to numpy for speed by vectorization
        st = np.array(list(list(zip(*batch))[0]))
        actions = np.array(list(list(zip(*batch))[1]))
        nst = np.array(list(list(zip(*batch))[3]))

        # next state shape is binary tree format so we need to split between the action 0 and 1 prediction
        nst = np.squeeze(np.split(nst, 2, axis=1))
        nst_a0 = nst[0]
        nst_a1 = nst[1]
        # print(st.shape)
        # print(actions.shape)
        # print(nst_a0.shape)
        # print(nst_a1.shape)
        # safety check if every input consists of a binary feature
        verifyBatchShape(st, np.zeros((self.batch_size, st[0].shape[0], 2)).shape)

        # predict on the batches with the model as well as the target values
        st_predict = self.dqn.predict(st)
        nst0_predict = self.dqn.predict(nst_a0)
        nst0_predict_target = self.target_dqn.predict(nst_a0)
        nst1_predict = self.dqn.predict(nst_a1)
        nst1_predict_target = self.target_dqn.predict(nst_a1)

        nst_predict = np.stack((np.amax(nst0_predict, axis=1),
                                np.amax(nst1_predict, axis=1)),
                               axis=-1)

        nst_predict_target = np.stack((np.amax(nst0_predict_target, axis=1),
                                       np.amax(nst1_predict_target, axis=1)),
                                      axis=-1)

        # print(nst_predict)
        # print(nst_predict_target)

        return st_predict, nst_predict, nst_predict_target
