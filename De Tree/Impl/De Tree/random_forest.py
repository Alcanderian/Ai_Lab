"""
random forest module.
"""
import numpy as np
import decision_tree
import random
import datatools
from toolkit import multiple

def train(mat, toolkit=multiple.id3, n_attr=None, m_tree=None,
          max_depth=float('inf')):
    """
    train random forest with 
    n_attr = sqrt(cnt_attr)
    and
    m_tree = square(cnt_attr)
    """
    cnt_attr, cnt_row = len(mat[0]) - 1, len(mat)
    if n_attr is None:
        n_attr = int(np.sqrt(cnt_attr))
    if m_tree is None:
        m_tree = int(np.square(cnt_attr))
    col_list = list(range(cnt_attr))
    forest = []
    for m in range(m_tree): 
        row_idx = [[random.randint(0, cnt_row - 1)] for i in range(cnt_row)]
        col_idx = np.append(random.choices(col_list, k=n_attr), [cnt_attr])
        subset = mat[row_idx, col_idx]
        forest.append(decision_tree.train(subset, col_idx,
                                          toolkit, max_depth))
    return forest


def plotable(forest):
    """
    let trees in forest be plotable by treePlotter.py
    """
    plotable_forest = [decision_tree.plotable(tree) for tree in forest]
    return plotable_forest


def predict(forest, mat):
    """
    predict wtih forest by voting.
    """
    tagm = []
    for tree in forest:
        tagm.append(decision_tree.predict(tree, mat))
    tagm = np.array(tagm)
    tags = [decision_tree.max_tag(tagm[:, i:i + 1])
            for i in range(len(tagm[0]))]
    return np.array(tags, int)


def validation(forest, mat):
    """
    validate and evalute the forest.
    """
    actual, result = mat[:, -1], predict(forest, mat)
    return datatools.evalution(actual, result)
