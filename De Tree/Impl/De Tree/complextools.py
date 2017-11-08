"""
data tools for complex tree.
"""
import numpy as np
import datatools as dt


def condition_eval(mat, i, elems, efunc):
    """
    for complex tree.
    get conditional information evalution of i-th col.

    param:
    :elems: elem set of data
    :efunc: evalution function of data
    """
    condition_eval_r = 0.0
    for elem in elems:
        subset = dt.select_except_i(mat, i, elem, dt.eq)
        prob = len(subset) / float(len(mat))
        condition_eval_r += prob * efunc(subset)
    return condition_eval_r


def choose(mat, efunc, cmethod):
    """
    for complex tree.
    choose best col by information evalution gain.

    param:
    :efunc: evalution function of data
    :cmethod: choose method, 'gain' or 'gain_rate'
    """
    max_diff, max_index = float('-inf'), None
    if cmethod != 'gain' and cmethod != 'gain_rate':
        {cmethod: max_diff, 'index': max_index}
    eval_m = efunc(mat)
    for i in range(len(mat[0]) - 1):
        splits = np.unique(mat[:, i])
        diff = eval_m - condition_eval(mat, i, splits, efunc)
        if diff == 0:
            continue
        if cmethod == 'gain_rate':
            diff = diff / efunc(mat[:, i:i + 1])
        if diff > max_diff:
            max_diff, max_index = diff, i
    return {cmethod: max_diff, 'index': max_index}


def subset(mat, args):
    """
    get complex subsets.
    """
    subsets, index, op = [], args['index'], dt.eq
    elems = np.unique(mat[:, index])
    for elem in elems:
        subsets.append(dt.select_except_i(mat, index, elem, op))
    return subsets, elems


def partition(val, args):
    """
    get complex partition label of val.
    """
    return val
