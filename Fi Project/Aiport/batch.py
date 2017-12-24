"""
batch functions
"""
import random as rnd


class stochastic:
    @staticmethod
    def b(x, y=None):
        c = rnd.choice(range(x.shape[1]))
        if y is None:
            return x[:, c]
        else:
            return x[:, c], y[:, c]


class mini_batch:
    def __init__(self, batch_size):
        self.k = batch_size

    def b(self, x, y=None):
        c = rnd.sample(range(x.shape[1]), self.k)
        if y is None:
            return x[:, c]
        else:
            return x[:, c], y[:, c]


class full:
    @staticmethod
    def b(x, y=None):
        if y is None:
            return x
        else:
            return x, y
