"""
activation functions
"""
import gnumpy as gpu


class sigmoid:
    @staticmethod
    def a(x):
        return gpu.logistic(x)

    @staticmethod
    def d(x):
        a = sigmoid.a(x)
        return a * (1. - a)


class tanh:
    @staticmethod
    def a(x):
        return gpu.tanh(x)

    @staticmethod
    def d(x):
        a = tanh.a(x)
        return 1. - a ** 2


class identity:
    @staticmethod
    def a(x):
        return x

    @staticmethod
    def d(x):
        return gpu.ones(x.shape)


class leaky_relu:
    def __init__(self, alpha):
        self.alpha = alpha
        return

    def a(self, x):
        r = x.copy()
        r -= (x < 0) * r * (1. - self.alpha)
        return r

    def d(self, x):
        r = gpu.ones(x.shape)
        r -= (x < 0) * (1. - self.alpha)
        return r
