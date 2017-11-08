"""
train de tree.
"""
import numpy as np
import toolkit
import pickle
import de_tree
import random
m = np.loadtxt('../../Data/train.csv', delimiter=',')
perm = list(range(len(m)))
random.shuffle(perm)
perm = np.array(perm)
mt = m[perm[range(int(len(perm) * 4 / 5))]]
mv = m[perm[range(int(len(perm) * 4 / 5), len(perm))]]
t = de_tree.train(mt, toolkit=toolkit.complex.cart)
print(de_tree.validation(t, mv))
with open('../../Data/tree.pyv', 'wb') as val:
    pickle.dump(t, val)
