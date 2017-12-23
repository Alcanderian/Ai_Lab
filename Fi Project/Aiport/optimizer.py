"""
Optimizers
"""
import gnumpy as gpu


class gradient_optimizer:
    @staticmethod
    def o(g):
        return g


class adam_optimizer:
    def __init__(self, beta1=0.9, beta2=0.999):
        self.beta1, self.beta2 = beta1, beta2
        self.m, self.v, self.t = None, None, 0
        return

    def o(self, g):
        if self.m is None or self.v is None:
            m, v = gpu.zeros(g.shape), gpu.zeros(g.shape)
        self.t += 1
        self.m = self.beta1 * self.m + (1. - self.beta1) * g
        self.v = self.beta2 * self.v + (1. - self.beta2) * (g ** 2)
        r = gpu.sqrt(1. - self.beta2 ** self.t) / (1. - self.beta1 ** self.t)
        return r * self.m / gpu.sqrt(self.v)
