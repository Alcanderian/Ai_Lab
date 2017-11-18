"""
train de tree.
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
#mt = m[perm[range(int(len(perm) * 0.7))]]
#mv = m[perm[range(int(len(perm) * 0.7), len(perm))]]
mt = np.loadtxt('../../Data/magictrain.csv', delimiter=',')
mv = np.loadtxt('../../Data/magicvalid.csv', delimiter=',')
f = random_forest.train(ma, toolkit.multiple.cart, 6, 1024)
t = decision_tree.train(ma, toolkit=toolkit.binary.discrete.cart)
# validation
print(random_forest.validation(f, mt))
print(random_forest.validation(f, mv))
print(decision_tree.validation(t, mt))
print(decision_tree.validation(t, mv))
# write binary
with open('../../Data/forest.pyv', 'wb') as val:
    pickle.dump(f, val)
with open('../../Data/tree.pyv', 'wb') as val:
    pickle.dump(t, val)
