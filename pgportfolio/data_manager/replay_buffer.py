""" 
Data structure for implementing experience replay

Author: Huang Yanqi
"""
from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size, config, random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self._config = config
        random.seed(random_seed)

    def add(self, s, close, stamp):
        experience = (s, close, stamp)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def geometric_sampling(self, l, n, p):
        res = []
        while len(res)<n:
            ind = np.random.geometric(p, 1)[0]
            if ind<=len(l):
                res.append(l[-ind])
        return res

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = self.geometric_sampling(self.buffer, self.count, self._config["training"]["buffer_biased"])
        else:
            batch = self.geometric_sampling(self.buffer, batch_size, self._config["training"]["buffer_biased"])

        s_batch = np.array([_[0] for _ in batch])
        close_batch = np.array([_[1] for _ in batch])
        stamp_batch = np.array([_[2] for _ in batch])

        return s_batch, close_batch, stamp_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


