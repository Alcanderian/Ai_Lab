"""
activation functions
"""
from config import mk, cpu, gpu


class sigmoid:
    @staticmethod
    def a(x):
        if mk is gpu:
            return mk.logistic(x)
        elif mk is cpu:
            return 1. / (1. + mk.exp(-x))
        else:
            raise TypeError('only cpu and gpu are supported')

    @staticmethod
    def d(x):
        a = sigmoid.a(x)
        return a * (1. - a)


class tanh:
    @staticmethod
    def a(x):
        return mk.tanh(x)

    @staticmethod
    def d(x):
        return 1. - mk.tanh(x) ** 2


class identity:
    @staticmethod
    def a(x):
        return x

    @staticmethod
    def d(x):
        return mk.ones(x.shape)


class leaky_relu:
    def __init__(self, alpha):
        self.alpha = alpha
        return

    def a(self, x):
        r = x.copy()
        r -= (x < 0) * r * (1. - self.alpha)
        return r

    def d(self, x):
        r = mk.ones(x.shape)
        r -= (x < 0) * (1. - self.alpha)
        return r
