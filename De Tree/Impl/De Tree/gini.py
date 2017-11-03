"""
gini module, for CART.
"""
import numpy as np
import data_op


def gini(mat):
    """
    get gini.
    """
    sums = {}
    for vec in mat:
        if vec[-1] not in sums:
            sums[vec[-1]] = 0.0
        sums[vec[-1]] += 1.0
    gini_r = 1.0
    for cnt in sums.values():
        prob = cnt / len(mat)
        gini_r -= prob * prob
    return gini_r


def gini_continuous_split(mat, i, val):
    """
    get gini split of i-th col split by val.
    """
    gini_split_r = 0.0
    for op in [lambda x, y: x > y, lambda x, y: x < y]:
        subset = data_op.select_except_i(mat, i, val, op)
        prob = len(subset) / float(len(mat))
        gini_split_r += prob * gini(subset)
    return gini_split_r


def choose_gini_continuous(mat):
    """
    choose best col and best split by gini slipt.
    """
    gini_m = gini(mat)
    min_gini, min_index, min_split = float('-inf'), None, None
    for i in range(len(mat[0]) - 1):
        vals = sorted(np.unique(mat[:, i]))
        if len(vals) == 1:
            continue
        splits = [(vals[j] + vals[j + 1]) / 2 for j in range(len(vals) - 1)]
        ginis = [gini_continuous_split(mat, i, split) for split in splits]
        index = np.argmin(ginis)
        if ginis[index] > min_gini:
            min_gini, min_index, min_split = ginis[index], i, splits[index]
    return {'gini': min_gini, 'index': min_index, 'split': min_split}


def gini_continuous_subset(mat, args):
    """
    get subsets.
    partition func: val < split
    """
    index, split, subsets = args['index'], args['split'], []
    for op in [lambda x, y: x > y, lambda x, y: x < y]:
        subsets.append(data_op.select_except_i(mat, index, split, op))
    return subsets, ['x > ' + str(split), 'x < ' + str(split)]


def gini_continuous_partition(val, args):
    """
    get partition label of val.
    """
    split = args['split']
    if val < split:
        return 'x < ' + str(split)
    else:
        return 'x > ' + str(split)
