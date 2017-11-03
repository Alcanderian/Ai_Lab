"""
main of de tree.
"""
import numpy as np
import toolkit
import de_tree
if False:
    with open('../../Data/test.csv', 'r') as src:
        s = src.read().replace('?', 'NaN')
    with open('../../Data/test.csv', 'w') as dst:
        dst.write(s)
m = np.loadtxt('../../Data/train.csv', delimiter=',')
print(de_tree.train(m, toolkit=toolkit.ccart))
