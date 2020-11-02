import os
import random
import time
from collections import deque
from resources.Utils import load_object, store_object


class MemoryBuffer:
    def __init__(self, max, id):
        self.memory = deque([], maxlen=max)
        self.id = "memory_{}.obj".format(id)

    def store(self, state, action, reward, nstate, done):
        # Store the experience in memory
        self.memory.append((state, action, reward, nstate, done))

    def init_memory(self, env):
        # time measurement for memory initialization
        init_time = time.time()
        # resetting environment once
        env.reset()
        # try to load memory from local file
        if os.path.isfile(self.id):
            self.memory = load_object(self.id)
        # try to init memory by taking random steps in our environment until the deque is full
        else:
            while True:
                # break if memory is full
                if len(self.memory) >= self.memory.maxlen:
                    break
                # check if we need to reset env and still fill our memory
                if env.is_done(env.timeseries_cursor):
                    env.reset()
                # get random action
                action = random.randrange(env.action_space_n)
                # take step in env and append
                state, action, reward, nstate, done = env.step_window(action)
                # store our memory in class
                self.store(state, action, reward, nstate, done)
            # store our memory locally to reduce loading time on next run
            store_object(self.memory, self.id)
            print("Memory is full, {} Samples stored. It took {} seconds".format(len(self.memory),
                                                                                 time.time() - init_time))

    def get_exp(self, batch_size):
        # Popping from the Memory Queue which should be filled randomly beforehand
        return [self.memory.popleft() for _i in range(batch_size)]

    def __len__(self):
        return len(self.memory)


class PrioritizedMemoryBuffer(MemoryBuffer):
    """
    WIP
    """

    def __init__(self, max):
        # Initialize Buffer for PER
        super().__init__(max)

    def store(self, state, action, reward, nstate, done):
        # Basically store as in root class
        # Add the error for the sample by taking them of the Q Estimator, like below
        # self.qnetwork_local.sample_noise()
        # Q_expected = self.qnetwork_local(states).gather(1, actions)
        # max_next_actions = self.get_max_next_actions(next_states)
        # self.qnetwork_target.sample_noise()
        # max_next_q_values = self.qnetwork_target(next_states).gather(1, max_next_actions)
        # Q_targets = rewards + (self.GAMMA * max_next_q_values * (1 - dones))
        # 
        # errors = Q_expected - Q_targets
        # store the tuple (state, action, reward, nstate, done, errors) in memory
        pass

    def get_exp(self, batch_size):
        # return sample with highest priority yet in tree
        pass

    def update_priority(self):
        # update priority of sample
        pass
