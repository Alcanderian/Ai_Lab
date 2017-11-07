"""
train de tree.
"""
import numpy as np
import toolkit
import pickle
import de_tree
m = np.loadtxt('../../Data/train.csv', delimiter=',')
t = de_tree.train(m, toolkit=toolkit.id3)
with open('../../Data/tree.pyv', 'wb') as val:
    pickle.dump(t, val)
