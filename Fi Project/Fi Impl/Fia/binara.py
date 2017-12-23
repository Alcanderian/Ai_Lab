import csv
import numpy as np

ft = '../../Data/bc/train.csv'
fs = '../../Data/bc/test.csv'

rt = '../../Data/bc/urain.csv'
rs = '../../Data/bc/uest.csv'


def convert_a(src, dst, usecols):
    str2i = {b'type_A': 0, b'type_B': 1, b'type_C': 2, b'type_D': 3, b'type_E': 4}

    rm = np.loadtxt(src, delimiter=',', converters={1: lambda s: str2i[s]}, usecols=usecols)
    ty = np.zeros((rm.shape[0], 5))
    for i, r in enumerate(rm):
        ty[i][int(r[0])] = 1.0
    rm = np.column_stack((ty, rm[:, 1:]))
    np.savetxt(dst, rm, delimiter=',')


convert_a(ft, rt, range(1, 14))
convert_a(fs, rs, range(1, 13))

rt = '../../Data/bc/vrain.csv'
rs = '../../Data/bc/vest.csv'


def convert_b(src, dst, usecols):
    str2i = {b'type_A': 0, b'type_B': 1, b'type_C': 2, b'type_D': 3, b'type_E': 4}

    rm = np.loadtxt(src, delimiter=',', converters={1: lambda s: str2i[s]}, usecols=usecols)
    np.savetxt(dst, rm, delimiter=',')


convert_b(ft, rt, range(1, 14))
convert_b(fs, rs, range(1, 13))
