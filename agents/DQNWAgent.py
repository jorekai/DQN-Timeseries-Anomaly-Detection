import numpy as np
import random

# custom modules
from agents.MemoryBuffer import MemoryBuffer
from agents.NeuralNetwork import build_model
from environment import BatchLearning

# Global Variables
EPISODES = 21
TRAIN_END = 0
DISCOUNT_RATE = 0.5
LEARNING_RATE = 0.001
BATCH_SIZE = 256


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
        self.model = build_model()
        self.model_target = build_model()  # Second (target) neural network
        self.update_target_from_model()  # Update weights
        self.hist = None
        self.loss = []

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA)  # Explore
        if self.epsilon == 0:
            action_vals = self.model_target.predict(np.array(state).reshape(1,
                                                                            BatchLearning.SLIDE_WINDOW_SIZE))
        else:
            action_vals = self.model.predict(np.array(state).reshape(1,
                                                                     BatchLearning.SLIDE_WINDOW_SIZE))  # Exploit: Use the NN to predict the correct action from this state
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
        epoch_count = 2
        self.hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)

    def update_target_from_model(self):
        # Update the target model from the base model
        self.model_target.set_weights(self.model.get_weights())

    def anneal_eps(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
