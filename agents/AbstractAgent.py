from abc import ABC, abstractmethod
import tensorflow as tf


class AbstractAgent(ABC):
    """
    The abstract base class should implement abstract common methods for every DQN Agent type, regardless of their
    choice of algorithm.
    """

    @abstractmethod
    def __init__(self, dqn, memory, alpha, gamma, epsilon,
                 epsilon_end, epsilon_decay, fit_epoch, action_space, batch_size):
        """
        Initializing an agent which shall learn an optimal Q(s, a) value function.
        :param dqn: model -> keras.model
        :param memory: MemoryBuffer
        :param alpha: float ∈ [0.0,1.0], learning factor
        :param gamma: float ∈ [0.0,1.0], discount factor
        :param epsilon: float ∈ [0.0,1.0], exploration factor
        :param epsilon_end: float ∈ [0.0,1.0], exploration end factor
        :param epsilon_decay: float ∈ [0.0,1.0], exploration decay per epoch
        :param fit_epoch: int > 0, keras fit param
        """
        # learning
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        # network
        self.dqn = dqn
        self.target_dqn = tf.keras.models.clone_model(dqn)
        self.fit_epoch = fit_epoch
        # memory
        self.memory = memory
        # state
        self.action_space = action_space
        self.batch_size = batch_size
        # evaluation training
        self.hist = None
        self.loss = []

    @abstractmethod
    def action(self, state):
        """
        The action function chooses according to a certain policy actions in our environment
        :return: action ∈ (action_space)
        """
        pass

    @abstractmethod
    def experience_replay(self):
        """
        Experience Replay is a major feature of DQNs. The tuples stored as transitions are of form:
        (state, action, reward, next_state, done). Agents can use different shapes of states or actions and therefore
        can have different implementations of experience replay.
        :return: void
        """
        pass

    @abstractmethod
    def predict_on_batch(self, batch):
        """
        Every agent which is using experience replay probably needs to predict on batches of samples.
        Agents can use different shapes of states or actions and therefore can have different prediction implementations
        :return: batch_pred(state), batch_pred(next_state), batch_pred_target_net(next_state)
        """
        pass

    def update_target_model(self):
        """
        Update the target model from the base model
        """
        self.target_dqn.set_weights(self.dqn.get_weights())

    def anneal_epsilon(self):
        """
        Anneal our epsilon factor by the decay factor
        """
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)
