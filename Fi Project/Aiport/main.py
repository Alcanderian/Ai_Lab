import gnumpy as gpu
import numpy as np
from time import time
from tqdm import tqdm

ox = [range(1000) for x in range(1000)]
oy = [range(1000) for y in range(1000)]

m = gpu.garray(ox).astype(np.float64)
n = gpu.garray(oy).astype(np.float64)

p = np.array(ox).astype(np.float64)
q = np.array(oy).astype(np.float64)


def run_gnumpy(a, b):
    st_g = time()
    for _ in tqdm(xrange(1000)):
        gpu.dot(a, b)
    et_g = time()
    return et_g - st_g


def run_numpy(a, b):
    st_n = time()
    for _ in tqdm(xrange(1000)):
        np.dot(a, b)
    et_n = time()
    return et_n - st_n


run_numpy(p, q)
run_gnumpy(m, n)
