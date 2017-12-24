"""
evaluation functions
"""
from config import mk


class mse:
    @staticmethod
    def l(y, h):
        n, _ = h.shape
        return (0.5 * mk.mean((y - h) ** 2, axis=1)).reshape((n, 1))

    @staticmethod
    def d(y, h):
        return h - y


class xent:
    @staticmethod
    def l(y, h):
        n, _ = h.shape
        return -mk.mean(y * mk.log(h) + (1. - y) * mk.log(1. - h), axis=1).reshape((n, 1))

    @staticmethod
    def d(y, h):
        return (h - y) / (h * (1. - h) + 1e-38)


class rmse:
    @staticmethod
    def l(y, h):
        n, _ = h.shape
        return mk.sqrt(0.5 * mk.mean((y - h) ** 2, axis=1)).reshape((n, 1))

    @staticmethod
    def d(y, h):
        raise NotImplemented('derivation function of \'rmse\' has no significance')


class nf1:
    @staticmethod
    def l(y, h):
        n, _ = h.shape
        t = h > 0.5
        e = mk.zeros((n, 1))
        for i in xrange(n):
            s = y[i] + 2 * t[i]
            tp = mk.sum(s == 3.)
            fn = mk.sum(s == 1.)
            fp = mk.sum(s == 2.)
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f1 = 2 * precision * recall / (precision + recall)
            e[i][0] = 1 - f1
        return e

    @staticmethod
    def d(y, h):
        raise NotImplemented('\'nf1\' has no derivation function')
