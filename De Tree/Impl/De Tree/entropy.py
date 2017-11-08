"""
entropy module.
"""
import numpy as np
from datatools import count_each_tag

def entropy(mat):
    """
    get entropy.
    """
    entropy_r = 0.0
    for cnt in count_each_tag(mat).values():
        prob = cnt / len(mat)
        entropy_r -= prob * np.log2(prob)
    return entropy_r
