import numpy as np
import gnumpy as gpu
import loss
import distance
import batch
import activation

a = gpu.garray([[1, 2, 1], [1, 0, 1], [7, 8, 9]])
b = gpu.garray([[0, 0, 1], [1, -4, 1], [-10, 12, 14]])
c = gpu.garray([[1], [2], [3]])

print(a)
print(a.transpose())
print(a + c)
for l in xrange(6):
    print(l)
