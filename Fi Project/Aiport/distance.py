"""
distance functions
"""
from config import mk


class norm_distance:
    def __init__(self, p):
        self.p = p
        return

    def d(self, x, y):
        if self.p == 1:
            return mk.sum(mk.abs(x - y), axis=0)
        if self.p == 2:
            return mk.sqrt(mk.sum((x - y) ** 2, axis=0))
        return mk.sum(mk.abs(x - y) ** self.p, axis=0) ** (1. / self.p)
