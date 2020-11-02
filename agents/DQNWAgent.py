import numpy as np
import random

# custom modules
from agents.MemoryBuffer import MemoryBuffer
from agents.NeuralNetwork import build_model, build_lstm
from environment import WindowStateFunctions

# Global Variables
BATCH_SIZE = 512


class DDQNWAgent:
    """
    This Agent is using a Sliding Window Approach to estimate the Q(s,a) Values.
    The Sliding Window is moved over the Timeseries input to create subsequent states of it.
    The Batch Size describes how many transitions are updated with the NN function approximation at once
    """

    def __init__(self, actions, alpha, gamma, epsilon, epsilon_end, epsilon_decay):
        """
        :param actions: The amount of actions which are possible in our environment
        :param alpha: The hyperparameter for Q-Learning to choose the learning rate
        :param gamma: The hyperparameter for Q-Learning to choose the discount rate
        :param epsilon: The hyperparameter for Q-Learning to choose the exploration percentage starting point
        :param epsilon_end: The hyperparameter for Q-Learning to choose the exploration percentage end point
        :param epsilon_decay: The hyperparameter for Q-Learning to choose the exploration decay per episode
        """
        self.nA = actions
        self.batch_size = BATCH_SIZE
        self.alpha = alpha
        self.gamma = gamma
        self.memory = MemoryBuffer(max=50000, id="ddqnw")
        # Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        # fitting param
        self.epoch_count = 2
        self.model = build_model()
        self.model_target = build_model()  # target neural network, updated every n epsiode
        self.update_target_from_model()  # update weights of target neural network
        self.hist = None  # history object can be used for tensorflow callback
        self.loss = []  # the loss array which can be used for callbacks from tensorflow

    def action(self, state):
        """
        The action function chooses according to epsilon-greedy policy actions in our environment
        :param state: The state, representing the Sliding Window of datapoints in our Timeseries
        :return: returns an action in our action space ∈ (0, 1)
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA)  # Explore Actions by percentage epsilon
        if self.epsilon == 0:  # greedy action choice for testing the agents performance in our environment
            action_vals = self.model_target.predict(np.array(state).reshape(1,
                                                                            WindowStateFunctions.SLIDE_WINDOW_SIZE))
        else:
            action_vals = self.model.predict(np.array(state).reshape(1,
                                                                     WindowStateFunctions.SLIDE_WINDOW_SIZE))
        return np.argmax(action_vals)

    def experience_replay(self, batch_size):
        """
        Experience Replay is necessary for our off-policy algorithm to derive the optimal policy out of the
        Transition-Set:(state, action, reward, next_state, done) which is of size: batch_size.
        DQN: The algorithm estimates the return (total discounted future reward) for state-action pairs assuming a greedy
        policy was followed despite the fact that it's not following a greedy policy.

        :param batch_size: number of transitions to be fitted at once
        """
        # get a sample from the memory buffer
        minibatch = self.memory.get_exp(batch_size)

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
        self.hist = self.model.fit(x_reshape, y_reshape, epochs=self.epoch_count, verbose=0)

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
            expected_shape = np.zeros((BATCH_SIZE, WindowStateFunctions.SLIDE_WINDOW_SIZE)).shape
            msg = "Shape mismatch for Experience Replay, shape expected: {}, shape received: {}".format(st.shape,
                                                                                                        expected_shape)
            assert st.shape == expected_shape, msg
        except AssertionError:
            raise

        # predict on the batches with the model as well as the target values
        st_predict = self.model.predict(st)
        nst_predict = self.model.predict(nst)
        nst_predict_target = self.model_target.predict(nst)

        return st_predict, nst_predict, nst_predict_target

    def update_target_from_model(self):
        """
        Update the target model from the base model
        """
        self.model_target.set_weights(self.model.get_weights())

    def anneal_eps(self):
        """
        Anneal our epsilon factor by the decay factor
        """
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
