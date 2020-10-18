import os
import random
import time
from collections import deque

from environment import BatchLearning
from resources.Utils import load_object, store_object


class MemoryBuffer:
    def __init__(self, max):
        self.memory = deque([], maxlen=max)

    def store(self, state, action, reward, nstate, done):
        # Store the experience in memory
        self.memory.append((state, action, reward, nstate, done))

    def init_memory(self, env):
        # time measurement for memory initialization
        init_time = time.time()
        # resetting environment once
        env.reset()
        # try to load memory from local file
        if os.path.isfile("memory.obj"):
            self.memory = load_object("memory.obj")
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
            store_object(self.memory, "memory.obj")
            print("Memory is full, {} Samples stored. It took {} seconds".format(len(self.memory),
                                                                                 time.time() - init_time))

    def get_exp(self, batch_size):
        # Popping from the Memory Queue which should be filled randomly beforehand
        return [self.memory.popleft() for _i in range(batch_size)]
        # return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
