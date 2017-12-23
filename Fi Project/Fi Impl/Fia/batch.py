import random as rnd


class stochastic:
    def b(self, data):
        return data[rnd.choice(range(data.shape[0])), :]


class mini_batch:
    def __init__(self, batch_size):
        self.k = batch_size

    def b(self, data):
        return data[rnd.sample(range(data.shape[0]), self.k), :]


class full:
    def b(self, data):
        return data
