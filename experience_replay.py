from collections import deque
import random

class ReplayMemory:
    #create FIFO queue -> experience replay
    def __init__(self, maxlen, seed=None):
        self.memory=deque([], maxlen=maxlen)

    def append(self, state, action, next_state, reward, terminated):
        self.memory.append((state, action, next_state, reward, terminated))

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    #current buffer size
    def __len__(self):
        return len(self.memory)
