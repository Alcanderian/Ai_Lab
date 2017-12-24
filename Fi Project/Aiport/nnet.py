"""
Neural Network
"""
import pickle
from config import mk, cpu, gpu
from batch import full
from loss import mse
from tqdm import tqdm


class bpnn:
    def __init__(self, layers, activations, optimizers, alphas, lambdas, loss=mse, batch=full,
                 handle=None):
        """
        :param layers: [list]list of number of units for each layer
        :param activations: [list]list of activation for layers
        :param optimizers: [list[dict]]list of dict of optimizer like {'w': optA, 'b': optB} for layers
        :param alphas: [list]list of learning rate factor for layers
        :param lambdas: [list]list of regularization term factor for layers
        :param loss: [class]loss function of output
        :param batch: [class]batch function of nnet
        :param handle: [function array(array)]handle function for transforming raw output to formatted output
        """
        self.w, self.b, self.a, self.z, self.g, self.d = {}, {}, {}, {}, {}, {}
        self.act, self.opt, self.alp, self.lbd = activations, optimizers, alphas, lambdas
        self.loss, self.batch, self.handle, self.nl = loss, batch, handle, len(layers)
        for l in xrange(self.nl - 1):
            if mk is gpu:
                self.w[l] = mk.randn((layers[l + 1], layers[l]))
                self.b[l] = mk.randn((layers[l + 1], 1))
            elif mk is cpu:
                self.w[l] = mk.random.uniform(-1, 1, (layers[l + 1], layers[l]))
                self.b[l] = mk.random.uniform(-1, 1, (layers[l + 1], 1))
            else:
                raise TypeError('only cpu and gpu are supported')
        return

    def propagate(self, x):
        self.a[0] = x
        for l in xrange(self.nl - 1):
            self.z[l + 1] = mk.dot(self.w[l], self.a[l]) + self.b[l]
            self.a[l + 1] = self.act[l].a(self.z[l + 1])
        return

    def compute_gradient(self, y):
        self.d[self.nl - 1] = self.loss.d(y, self.a[self.nl - 1]) * \
                              self.act[self.nl - 2].d(self.z[self.nl - 1])
        for l in xrange(self.nl - 2, 0, -1):
            self.d[l] = mk.dot(self.w[l].transpose(), self.d[l + 1]) * \
                        self.act[l - 1].d(self.z[l])
        for l in xrange(self.nl - 1):
            if l not in self.g:
                self.g[l] = {}
            n, m = self.d[l + 1].shape
            self.g[l]['w'] = mk.dot(self.d[l + 1], self.a[l].transpose()) / m
            self.g[l]['b'] = mk.mean(self.d[l + 1], axis=1).reshape((n, 1))
        return

    def back_propagate(self):
        for l in xrange(self.nl - 1):
            self.w[l] -= self.alp[l] * self.opt[l]['w'].o(self.g[l]['w'] + self.lbd[l] * self.w[l])
            self.b[l] -= self.alp[l] * self.opt[l]['b'].o(self.g[l]['b'])
        return

    def train(self, epoch, tx, ty, vx=None, vy=None):
        """
        :param epoch: number of iteration
        :param tx: [array]training feature set
        :param ty: [array]training expectation set
        :param vx: [array]validation feature set
        :param vy: [array]validation expectation set
        :return: [list]training loss and [list]validation loss(if validation sets are exist)
        """
        tl = []
        if vx is not None and vy is not None:
            vl = []
        else:
            vl = None
        it = tqdm(range(epoch))
        it.set_description('bpnn train')
        for _ in it:
            x, y = self.batch.b(tx, ty)
            self.propagate(x)
            self.compute_gradient(y)
            self.back_propagate()
            hy = self.predict(tx)
            tl.append(self.loss.l(ty, hy))
            if vl is not None:
                hy = self.predict(vx)
                vl.append(self.loss.l(vy, hy))
                it.set_description('bpnn train, [tl: %.0f, vl: %.0f]' % (tl[-1][0][0], vl[-1][0][0]))
            else:
                it.set_description('bpnn train, [tl: %.0f]' % (tl[-1][0][0]))
        if vl is None:
            return tl
        else:
            return tl, vl

    def predict(self, x):
        """
        :param x: [array]testing feature set
        :return: [array]formatted result
        """
        self.propagate(x)
        if self.handle is None:
            return self.a[self.nl - 1]
        else:
            return self.handle(self.a[self.nl - 1])

    def save(self, filename):
        pickle.dump(self, filename)
        return

    @staticmethod
    def load(filename):
        o = pickle.load(filename)
        if isinstance(o, bpnn):
            return o
        else:
            TypeError('file is not an object of \'bpnn\'')


class cluster_bpnn:
    def __init__(self, cluster, layers, activations, optimizers, alphas, lambdas, loss, batch):
        return
