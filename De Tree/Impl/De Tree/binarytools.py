"""
data tools for binary tree.
"""
import numpy as np
import datatools as dt
import itertools


def condition_eval(mat, i, split, efunc, eop):
    """
    for binary tree.
    get conditional information evalution of i-th col.

    :param:
    split: spliter of data
    efunc: evalution function of data
    eop: select opreator of data
    """
    condition_eval_r = 0.0
    for lbd in [lambda x, y: eop(x, y), lambda x, y: not eop(x, y)]:
        subset = dt.select(mat, i, split, lbd)
        prob = len(subset) / float(len(mat))
        condition_eval_r += prob * efunc(subset)
    return condition_eval_r


def splits_eval(mat, i, splits, efunc, eop):
    """
    for binary tree.
    get information evalution of i-th col with split.

    :param:
    splits: spliters of data
    efunc: evalution function of data
    eop: select opreator of data
    """
    evals = []
    for split in splits:
        map_col = [[eop(val, split)] for val in mat[:, i]]
        evals.append(efunc(map_col))
    return np.array(evals)


def choose(mat, efunc, spfunc, cmethod):
    """
    for binary tree.
    choose best col by information evalution gain.

    :param:
    efunc: evalution function of data
    spfunc: split function of data
    cmethod: choose method, 'gain' or 'gain_rate'
    """
    max_diff, max_index = float('-inf'), None
    max_split, operator = None, None
    if cmethod == 'gain' or cmethod == 'gain_rate':
        eval_m = efunc(mat)
        for i in range(len(mat[0]) - 1):
            vals = np.unique(mat[:, i])
            if len(vals) == 1:
                continue
            splits, op = spfunc(vals)
            diffs = [eval_m - condition_eval(mat, i, split, efunc, op)
                     for split in splits]
            if cmethod == 'gain_rate':
                diffs = diffs / splits_eval(mat, i, splits, efunc, op)
            index = np.argmax(diffs)
            if diffs[index] > max_diff:
                max_diff, max_index = diffs[index], i
                max_split, operator = splits[index], op
    return {cmethod: max_diff, 'index': max_index,
            'split': max_split, 'operator': operator}


def splits_continuous(vals):
    """
    get split thesholds of continuous values.

    return: subset spliter, operator to select subset
    """
    vals = sorted(vals)
    splits = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
    return splits, dt.lt


def splits_smart_discrete(vals):
    """
    get splits smartly.
    if len(vals) > 10, treat vals as continuous values.
    otherwise treat them as discrete values and
    get split set of vals.

    return: subset spliter, operator to select subset
    """
    if len(vals) == 2:
        return [(vals[0],)], dt.contains
    if len(vals) > 10:
        return splits_continuous(vals)
    splits = []
    for k in range(1, int(len(vals) / 2) + 1):
        splits.extend(itertools.combinations(vals, k))
    return splits, dt.contains


def subset(mat, args):
    """
    get binary subsets.
    we should reuse the column when there is continuous split.
    """
    index, split = args['index'], args['split']
    op, subsets = args['operator'], []
    fmt = (op.__name__, str(split))
    for lbd in [lambda x, y: op(x, y), lambda x, y: not op(x, y)]:
        if dt.is_continuous_split(args):
            subsets.append(dt.select(mat, index, split, lbd))
        else:
            subsets.append(dt.select_except_i(mat, index, split, lbd))
    return subsets, ['%s(x, %s)' % fmt, '~%s(x, %s)' % fmt]


def partition(val, args):
    """
    get binary partition label of val.
    """
    fmt = (args['operator'].__name__, str(args['split']))
    if args['operator'](val, args['split']):
        return '%s(x, %s)' % fmt
    else:
        return '~%s(x, %s)' % fmt
