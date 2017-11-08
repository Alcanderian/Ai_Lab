"""
train de tree.
"""
import numpy as np
import datatools as dt
import toolkit
import pickle
import de_tree
import random
m = np.loadtxt('../../Data/train.csv', delimiter=',')
perm = list(range(len(m)))
for i in range(30):
    random.shuffle(perm)
perm = np.array(perm)
mt = m[perm[range(int(len(perm) * 0.6))]]
mv = m[perm[range(int(len(perm) * 0.6), len(perm))]]
t = de_tree.train(m)
print(de_tree.validation(t, m))
with open('../../Data/tree.pyv', 'wb') as val:
    pickle.dump(t, val)
