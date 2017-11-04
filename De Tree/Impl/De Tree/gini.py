"""
gini module, for CART.
"""
import numpy as np
import data_op
import itertools


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


def gini_split(mat, i, val, op):
    """
    get gini split of i-th col split by val.

    param:
    :op: operator to select subset
    """
    gini_split_r = 0.0
    for lbd in [lambda x, y: op(x, y), lambda x, y: not op(x, y)]:
        subset = data_op.select_except_i(mat, i, val, lbd)
        prob = len(subset) / float(len(mat))
        gini_split_r += prob * gini(subset)
    return gini_split_r


def choose_gini(mat, get_splits):
    """
    choose best col and best split by gini slipt.

    param:
    :get_splits: function to get subset spliter
    """
    gini_m = gini(mat)
    min_gini, min_index = float('-inf'), None
    min_split, operator = None, None
    for i in range(len(mat[0]) - 1):
        vals = np.unique(mat[:, i])
        if len(vals) == 1:
            continue
        splits, op = get_splits(vals)
        ginis = [gini_split(mat, i, split, op) for split in splits]
        index = np.argmin(ginis)
        if ginis[index] > min_gini:
            min_gini, min_index = ginis[index], i
            min_split, operator = splits[index], op
    return {'gini': min_gini, 'index': min_index, 
            'split': min_split, 'operator': operator}


def gini_get_splits_continuous(vals):
    """
    get split thesholds of continuous values.

    return: subset spliter, operator to select subset
    """
    vals = sorted(vals)
    splits = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
    return splits, data_op.lt


def gini_get_splits_smart_discrete(vals):
    """
    get splits smartly.
    if len(vals) > 6, treat vals as continuous values.
    otherwise treat them as discrete values and
    get split set of vals.

    return: subset spliter, operator to select subset
    """
    if len(vals) == 2:
        return [(vals[0],)], data_op.contains
    if len(vals) > 10:
        return gini_get_splits_continuous(vals)
    splits = []
    for k in range(1, int(len(vals) / 2) + 1):
        splits.extend(itertools.combinations(vals, k))
    return splits, data_op.contains


def gini_subset(mat, args):
    """
    get subsets.
    """
    index, split = args['index'], args['split']
    op, subsets = args['operator'], []
    fmt = (op.__name__, str(split))
    for lbd in [lambda x, y: op(x, y), lambda x, y: not op(x, y)]:
        subsets.append(data_op.select_except_i(mat, index, split, lbd))
    return subsets, ['%s(x, %s)'%fmt, '~%s(x, %s)'%fmt]


def gini_partition(val, args):
    """
    get partition label of val.
    """
    fmt = (args['operator'].__name__, str(args['split']))
    if args['operator'](val, args['split']):
        return '%s(x, %s)'%fmt
    else:
        return '~%s(x, %s)'%fmt
