import numpy as np
import random

# custom modules
from agents.MemoryBuffer import MemoryBuffer
from agents.NeuralNetwork import build_model, build_lstm
from environment import BatchLearning

# Global Variables
BATCH_SIZE = 512


class DDQNWAgent:
    def __init__(self, actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.nA = actions
        self.memory = MemoryBuffer(max=50000)
        self.batch_size = BATCH_SIZE
        self.alpha = alpha
        self.gamma = gamma
        # Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # fitting param
        self.epoch_count = 5
        self.model = build_lstm()
        self.model_target = build_lstm()  # Second (target) neural network
        self.update_target_from_model()  # Update weights
        self.hist = None
        self.loss = []

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA)  # Explore
        if self.epsilon == 0:
            action_vals = self.model_target.predict(np.array(state).reshape(1, 1,
                                                                            BatchLearning.SLIDE_WINDOW_SIZE))
        else:
            action_vals = self.model.predict(np.array(state).reshape(1, 1,
                                                                     BatchLearning.SLIDE_WINDOW_SIZE))  # Exploit: Use the NN to predict the correct action from this state
        return np.argmax(action_vals)

    def experience_replay(self, batch_size, lstm):
        # get a sample from the memory buffer
        minibatch = self.memory.get_exp(batch_size)

        # create default input arrays for the fitting of the model
        x = []
        y = []

        st_predict, nst_predict, nst_predict_target = self.predict_on_batch(minibatch)

        # print(st_predict.shape)
        # print(nst_predict.shape)
        # print(nst_predict_target.shape)

        index = 0
        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            # Predict from state
            nst_action_predict_target = nst_predict_target[index]
            nst_action_predict_model = nst_predict[index]
            # print(nst_action_predict_target.shape)
            # print(nst_action_predict_model.shape)
            if done == True:  # Terminal: Just assign reward
                target = reward
            else:  # Non terminal
                # print(np.unravel_index(np.argmax(nst_action_predict_model), nst_action_predict_model.shape))
                target = reward + self.gamma * nst_action_predict_target[
                    np.unravel_index(np.argmax(nst_action_predict_model),
                                     nst_action_predict_model.shape)]  # Using Q to get T is Double DQN
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1
        # Reshape for Keras Fit
        x_reshape = np.array(x)
        y_reshape = np.array(y)

        x_reshape = np.reshape(x_reshape, (BATCH_SIZE, 1, BatchLearning.SLIDE_WINDOW_SIZE))

        self.hist = self.model.fit(x_reshape, y_reshape, epochs=self.epoch_count, verbose=0)

    def predict_on_batch(self, batch):
        # Convert to numpy for speed by vectorization
        st = np.array(list(list(zip(*batch))[0]))
        nst = np.array(list(list(zip(*batch))[3]))

        st = np.reshape(st, (BATCH_SIZE, 1, BatchLearning.SLIDE_WINDOW_SIZE))
        nst = np.reshape(nst, (BATCH_SIZE, 1, BatchLearning.SLIDE_WINDOW_SIZE))

        # predict on the batches with the model as well as the target values
        st_predict = self.model.predict(st)
        nst_predict = self.model.predict(nst)
        nst_predict_target = self.model_target.predict(nst)

        return st_predict, nst_predict, nst_predict_target

    def update_target_from_model(self):
        # Update the target model from the base model
        self.model_target.set_weights(self.model.get_weights())

    def anneal_eps(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
