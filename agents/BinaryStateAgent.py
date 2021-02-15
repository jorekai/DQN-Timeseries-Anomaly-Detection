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
            action_values = self.target_dqn.predict(lstm_input)
            # print(action_values)
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

        nextstate_predict_target = self.predict_on_batch(minibatch)

        states = np.array(list(list(zip(*minibatch))[0]))
        rewards = np.array(list(list(zip(*minibatch))[2]))
        # print(rewards)
        if self.gamma > 0:
            targets = rewards + self.gamma * nextstate_predict_target
        else:
            targets = rewards
        self.hist = self.dqn.fit(states, targets, epochs=self.fit_epoch, verbose=0)

    def predict_on_batch(self, batch):
        """
        Helper method to get the predictions on out batch of transitions in a shape that we need
        :param batch: n batch size
        :return: NN predictions(state), NN predictions(next_state), Target NN predictions(next_state)
        """
        # Convert to numpy for speed by vectorization
        nst = np.array(list(list(zip(*batch))[3]))

        # next state shape is binary tree format so we need to split between the action 0 and 1 prediction
        nst = np.squeeze(np.split(nst, 2, axis=1))
        nst_a0 = nst[0]
        nst_a1 = nst[1]

        # safety check if every input consists of a binary feature
        # verifyBatchShape(nst, np.zeros((self.batch_size, nst[0].shape[0], 2)).shape)

        nst0_predict_target = self.target_dqn.predict(nst_a0)
        nst1_predict_target = self.target_dqn.predict(nst_a1)

        nst_predict_target = np.stack((np.amax(nst0_predict_target, axis=1),
                                       np.amax(nst1_predict_target, axis=1)),
                                      axis=-1)

        # print(nst_predict)
        # print(nst_predict_target)

        return nst_predict_target
