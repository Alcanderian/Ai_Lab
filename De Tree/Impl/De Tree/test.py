"""
test param of tree and forest.
"""
import numpy as np
import datatools as dt
import toolkit
import pickle
import decision_tree
import random_forest
import random
ma = np.loadtxt('../../Data/train.csv', delimiter=',')
perm = list(range(len(ma)))
random.shuffle(perm)
perm = np.array(perm)
#mt = ma[perm[range(int(len(perm) * 0.7))]]
#mv = ma[perm[range(int(len(perm) * 0.7), len(perm))]]
mt = np.loadtxt('../../Data/magictrain.csv', delimiter=',')
mv = np.loadtxt('../../Data/magicvalid.csv', delimiter=',')
f = random_forest.train(ma, toolkit.multiple.cart, 6, 1024)
t = decision_tree.train(ma, toolkit=toolkit.binary.discrete.cart)
# predict
with open('../../Data/test.csv', 'r') as src:
    s = src.read().replace('?', 'NaN')
with open('../../Data/test.csv', 'w') as dst:
    dst.write(s)
ms = np.loadtxt('../../Data/test.csv', delimiter=',')
# forest
print(random_forest.validation(f, mt))
print(random_forest.validation(f, mv))
r = random_forest.predict(f, ms)
print({-1: len(r[r == -1]), 1: len(r[r == 1])})
# tree
print(decision_tree.validation(t, mt))
print(decision_tree.validation(t, mv))
r = decision_tree.predict(t, ms)
print({-1: len(r[r == -1]), 1: len(r[r == 1])})
