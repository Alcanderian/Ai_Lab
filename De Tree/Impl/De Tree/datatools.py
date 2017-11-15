"""
handle data set.
"""
import numpy as np


def count_each_tag(mat):
    """
    get count of each tag.
    """
    cnts = {}
    for vec in mat:
        if vec[-1] not in cnts:
            cnts[vec[-1]] = 0.0
        cnts[vec[-1]] += 1.0
    return cnts


def select(mat, i, val, op):
    """
    SQL: SELECT \* FROM mat WHERE op(mat[\*][i], val) = True
    """
    subset = []
    for vec in mat:
        if op(vec[i], val):
            subset.append(vec)
    return np.array(subset)


def select_except_i(mat, i, val, op):
    """
    SQL: SELECT \* FROM mat - mat[\*][i] WHERE op(mat[\*][i], val) = True
    """
    subset = []
    for vec in mat:
        if op(vec[i], val):
            subset.append(np.append(vec[:i], vec[i + 1:]))
    return np.array(subset)


def is_continuous_split(args):
    """
    return: True if args match continuous split method

    only binary tree's args has 'split'
    and only continuous split is float.
    """
    return 'split' in args and isinstance(args['split'], float)


def evalution(actual, result):
    """
    evalution of result.
    """
    sums = {1: {1: 0.0, -1: 0.0}, -1: {1: 0.0, -1: 0.0}}
    for i, j in zip(actual, result):
        sums[int(i)][int(j)] += 1.0
    eval = {'accuracy': (sums[1][1] + sums[-1][-1]) / len(actual),
            'precision': sums[1][1] / (sums[1][1] + sums[-1][1]),
            'recall': sums[1][1] / (sums[1][1] + sums[1][-1])}
    eval['f1'] = (2 * eval['precision'] * eval['recall']
                  / (eval['precision'] + eval['recall']))
    return eval


def lt(x, y):
    """
    x < y
    """
    return x < y


def contains(x, y):
    """
    x in y
    """
    return x in y


def eq(x, y):
    """
    x == y
    """
    return x == y
