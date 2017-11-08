"""
gini module.
"""
from datatools import count_each_tag


def gini(mat):
    """
    get gini.
    """
    gini_r = 1.0
    for cnt in count_each_tag(mat).values():
        prob = cnt / len(mat)
        gini_r -= prob * prob
    return gini_r
