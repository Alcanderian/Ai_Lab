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
ma = np.loadtxt('../../Data/ystrain.csv', delimiter=',')
t = decision_tree.train(ma, toolkit=toolkit.multiple.id3)
# predict
with open('../../Data/ystest.csv', 'r') as src:
    s = src.read().replace('?', 'NaN')
with open('../../Data/ystest.csv', 'w') as dst:
    dst.write(s)
ms = np.loadtxt('../../Data/ystest.csv', delimiter=',')
# tree
print(decision_tree.plotable(t))
r = decision_tree.predict(t, ms)
print(r)