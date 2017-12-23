"""
evaluation functions
"""
import gnumpy as gpu


class mse:
    @staticmethod
    def l(y, h):
        n, _ = h.shape
        return (0.5 * gpu.mean((y - h) ** 2, axis=1)).reshape((n, 1))

    @staticmethod
    def d(y, h):
        return y - h


class xent:
    @staticmethod
    def l(y, h):
        n, _ = h.shape
        return -gpu.mean(y * gpu.log(h) + (1. - y) * gpu.log(1. - h), axis=1).reshape((n, 1))

    @staticmethod
    def d(y, h):
        return (h - y) / (h * (1. - h) + 1e-38)


class rmse:
    @staticmethod
    def l(y, h):
        n, _ = h.shape
        return gpu.sqrt(0.5 * gpu.mean((y - h) ** 2, axis=1)).reshape((n, 1))

    @staticmethod
    def d(y, h):
        raise NotImplemented('derivation function of \'rmse\' has no significance')


class nf1:
    @staticmethod
    def l(y, h):
        n, _ = h.shape
        t = h > 0.5
        e = gpu.zeros((n, 1))
        for i in xrange(n):
            s = y[i] + 2 * t[i]
            tp = gpu.sum(s == 3.)
            fn = gpu.sum(s == 1.)
            fp = gpu.sum(s == 2.)
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f1 = 2 * precision * recall / (precision + recall)
            e[i][0] = 1 - f1
        return e

    @staticmethod
    def d(y, h):
        raise NotImplemented('\'nf1\' has no derivation function')
