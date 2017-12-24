from nnet import bpnn
from config import gpu
import numpy as np
import batch
from activation import tanh, sigmoid, leaky_relu
from optimizer import gradient_optimizer as gd, adam_optimizer as ad
from matplotlib import pyplot as plt

data = np.loadtxt('../../Nnet/Data/urain.csv', delimiter=',')
x = gpu.garray(data[:, :-1])
y = gpu.garray(data[:, -1:])
ratio = 9.3 / 10.
samples, _ = data.shape
tx = x[:int(ratio * samples) - 1].T
ty = y[:int(ratio * samples) - 1].T
vx = x[int(ratio * samples):samples - 1].T
vy = y[int(ratio * samples):samples - 1].T

nn = bpnn(layers=[45, 45, 45, 1],
          activations=[
              sigmoid,
              leaky_relu(0.1),
              leaky_relu(0.01)],
          optimizers=[
              {'w': ad(), 'b': ad()},
              {'w': ad(), 'b': ad()},
              {'w': ad(), 'b': ad()}
          ],
          alphas=[0.01, 0.01, 0.01],
          lambdas=[0.002, 0.001, 0.001],
          batch=batch.mini_batch(400))
tl, vl = nn.train(3000, x.T, y.T, vx, vy)

tl = [l[0][0] for l in tl]
vl = [l[0][0] for l in vl]
plt.figure()
plt.plot(tl, label='training loss')
plt.plot(vl, label='validation loss')
plt.legend()

py = gpu.as_numpy_array(nn.predict(tx)[0])
ay = gpu.as_numpy_array(ty[0])
plt.figure()
plt.plot(ay, label='training data')
plt.plot(py, label='predict')
plt.legend()

py = gpu.as_numpy_array(nn.predict(vx)[0])
ay = gpu.as_numpy_array(vy[0])
plt.figure()
plt.plot(ay, label='validation data')
plt.plot(py, label='predict')
plt.legend()

data = np.loadtxt('../../Nnet/Data/test.csv', delimiter=',')
x = data[:, :-1]
sx = gpu.garray(x.T)
py = gpu.as_numpy_array(nn.predict(sx)[0])
plt.figure()
plt.plot(py, label='predict')
plt.legend()

plt.show()
