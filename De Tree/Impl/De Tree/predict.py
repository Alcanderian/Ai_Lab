"""
predict by de tree.
"""
import pickle
import decision_tree
import random_forest
import numpy as np
with open('../../Data/forest.pyv', 'rb') as val:
    f = pickle.load(val)
with open('../../Data/tree.pyv', 'rb') as val:
    t = pickle.load(val)
with open('../../Data/test.csv', 'r') as src:
    s = src.read().replace('?', 'NaN')
with open('../../Data/test.csv', 'w') as dst:
    dst.write(s)
m = np.loadtxt('../../Data/test.csv', delimiter=',')
# forest
r = random_forest.predict(f, m)
print({-1: len(r[r == -1]), 1: len(r[r == 1])})
with open('../../Data/forestresult.csv', 'w') as dst:
    for tag in r:
        dst.write('%d\n'%(tag))
# tree
r = decision_tree.predict(t, m)
print({-1: len(r[r == -1]), 1: len(r[r == 1])})
with open('../../Data/treeresult.csv', 'w') as dst:
    for tag in r:
        dst.write('%d\n'%(tag))
