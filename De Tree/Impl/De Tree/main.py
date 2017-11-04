"""
main of de tree.
"""
import numpy as np
import toolkit
import pickle
import de_tree
if False:
    with open('../../Data/test.csv', 'r') as src:
        s = src.read().replace('?', 'nan')
    with open('../../Data/test.csv', 'w') as dst:
        dst.write(s)
m = np.loadtxt('../../Data/train.csv', delimiter=',')
t = de_tree.train(m, toolkit=toolkit.sdcart)
print(t)

with open('../../Data/tree.pyv', 'wb') as val:
    pickle.dump(t, val)
