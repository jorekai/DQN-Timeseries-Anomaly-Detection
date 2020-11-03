import numpy as np
import random as pyrand
from agents.AbstractAgent import AbstractAgent

from environment import WindowStateFunctions


class SlidingWindowAgent(AbstractAgent):
    """
    The Sliding Window Agent uses a Sliding Window as state representation of Timeseries.
    One can use all superclass hooks for training the agent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def action(self, state):
        """
        The action function chooses according to epsilon-greedy policy actions in our environment
        :param state: The state, representing the Sliding Window of datapoints in our Timeseries
        :return: returns an action in our action space ∈ (0, 1)
        """
        if np.random.rand() <= self.epsilon:
            # explore environment
            return pyrand.randrange(self.action_space)
        elif self.epsilon == 0:
            # exploit from target net (evaluation)
            action_values = self.target_dqn.predict(np.array(state).reshape(1,
                                                                            WindowStateFunctions.SLIDE_WINDOW_SIZE))
        else:
            # exploit from train net (training)
            action_values = self.dqn.predict(np.array(state).reshape(1, WindowStateFunctions.SLIDE_WINDOW_SIZE))
        # return index of max action in action list
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
            target_f = state_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1
        # Reshape for Keras Fitting Function
        x_reshape = np.array(x)
        y_reshape = np.array(y)
        self.hist = self.dqn.fit(x_reshape, y_reshape, epochs=self.fit_epoch, verbose=0)

    def predict_on_batch(self, batch):
        """
        Helper method to get the predictions on out batch of transitions in a shape that we need
        :param batch: n batch size
        :return: NN predictions(state), NN predictions(next_state), Target NN predictions(next_state)
        """
        # Convert to numpy for speed by vectorization
        st = np.array(list(list(zip(*batch))[0]))
        nst = np.array(list(list(zip(*batch))[3]))

        try:
            #  check for equivalence of array shapes
            expected_shape = np.zeros((self.batch_size, WindowStateFunctions.SLIDE_WINDOW_SIZE)).shape
            msg = "Shape mismatch for Experience Replay, shape expected: {}, shape received: {}".format(st.shape,
                                                                                                        expected_shape)
            assert st.shape == expected_shape, msg
        except AssertionError:
            raise

        # predict on the batches with the model as well as the target values
        st_predict = self.dqn.predict(st)
        nst_predict = self.dqn.predict(nst)
        nst_predict_target = self.target_dqn.predict(nst)

        return st_predict, nst_predict, nst_predict_target
