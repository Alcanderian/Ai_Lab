"""
Neural Network
"""
import gnumpy as gpu
import pickle
from tqdm import tqdm


class bpnn:
    def __init__(self, layers, activations, optimizers, alphas, lambdas, loss, batch, handle):
        """
        :param layers: [list]list of number of units for each layer
        :param activations: [list]list of activation for layers
        :param optimizers: [list[dict]]list of dict of optimizer like {'w': optA, 'b': optB} for layers
        :param alphas: [list]list of learning rate factor for layers
        :param lambdas: [list]list of regularization term factor for layers
        :param loss: [class]loss function of output
        :param batch: [class]batch function of nnet
        :param handle: [function gpu.garray(gpu.garray)]handle function for transforming raw output to formatted output
        """
        self.w, self.b, self.a, self.z, self.g, self.d = {}, {}, {}, {}, {}, {}
        self.act, self.opt, self.alp, self.lbd = activations, optimizers, alphas, lambdas
        self.loss, self.batch, self.handle, self.nl = loss, batch, handle, len(layers)
        for l in xrange(self.nl - 1):
            self.w[l] = gpu.randn((layers[l + 1], layers[l]))
            self.b[l] = gpu.randn((layers[l + 1], 1))
        return

    def propagate(self, x):
        self.a[0] = x
        for l in xrange(self.nl - 1):
            self.z[l + 1] = gpu.dot(self.w[l], self.a[l]) + self.b[l]
            self.a[l + 1] = self.act[l].a(self.z[l + 1])
        return

    def compute_gradient(self, y):
        self.d[self.nl - 1] = self.loss.d(y, self.a[self.nl - 1]) * \
                              self.act[self.nl - 2].d(self.z[self.nl - 1])
        for l in xrange(self.nl - 2, 1, -1):
            self.d[l] = gpu.dot(self.w[l].transpose(), self.d[l + 1]) * \
                        self.act[l - 1].d(self.z[l])
        for l in xrange(self.nl - 1):
            if self.g[l] is None:
                self.g[l] = {}
            n, _ = self.d[l + 1].shape
            self.g[l]['w'] = gpu.dot(self.d[l + 1], self.a[l].transpose())
            self.g[l]['b'] = gpu.mean(self.d[l + 1], axis=1).reshape((n, 1))
        return

    def back_propagate(self):
        for l in xrange(self.nl - 1):
            self.w[l] -= self.alp[l] * self.opt[l]['w'].o(self.g[l]['w'] + self.lbd[l] * self.w[l])
            self.b[l] -= self.alp[l] * self.opt[l]['b'].o(self.g[l]['b'])
        return

    def train(self, epoch, tx, ty, vx=None, vy=None):
        """
        :param epoch: number of iteration
        :param tx: [gpu.garray]training feature set
        :param ty: [gpu.garray]training expectation set
        :param vx: [gpu.garray]validation feature set
        :param vy: [gpu.garray]validation expectation set
        :return: [list]training loss and [list]validation loss(if validation sets are exist)
        """
        tl = []
        if vx is not None and vy is not None:
            vl = []
        else:
            vl = None
        for _ in tqdm(range(epoch)):
            x, y = self.batch.b(tx, ty)
            self.propagate(x)
            self.compute_gradient(y)
            self.back_propagate()
            hy = self.predict(tx)
            tl.append(self.loss.l(ty, hy))
            if vl is not None:
                hy = self.predict(vx)
                vl.append(self.loss.l(vy, hy))
        if vl is None:
            return tl
        else:
            return tl, vl

    def predict(self, x):
        """
        :param x: [gpu.garray]testing feature set
        :return: [gpu.garray]formatted result
        """
        self.propagate(x)
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
