import gnumpy
import numpy
from tqdm import tqdm
from time import sleep

ox = [range(1000) for x in range(1000)]
oy = [range(1000) for y in range(1000)]

m = gnumpy.garray(ox).astype('float64')
n = gnumpy.garray(oy).astype('float64')

p = numpy.array(ox).astype('float64')
q = numpy.array(oy).astype('float64')


def run_gnumpy(a, b):
    it = tqdm(range(1000))
    for i in it:
        gnumpy.dot(a, b)
    it.close()
    return


def run_numpy(a, b):
    it = tqdm(range(1000))
    for i in it:
        numpy.dot(a, b)
    it.close()
    return


print('start')
sleep(0.1)
run_gnumpy(m, n)
sleep(0.1)
run_numpy(p, q)
sleep(0.1)
print('finish')
