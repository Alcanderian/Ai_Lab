"""
test module, C4.5
"""
import numpy as np

def select_except_i(mat, i, val, op):
    """
    sql:
    select *
    from mat - mat[*][i]
    where op(mat[*][i], val) is true.
    """
    subset = []
    for vec in mat:
        if op(vec[i], val):
            subset.append(np.append(vec[:i], vec[i + 1:]))
    return np.array(subset)

def entropy(mat):
    """
    get entropy.
    """
    sums = {}
    for vec in mat:
        if vec[-1] not in sums:
            sums[vec[-1]] = 0.0
        sums[vec[-1]] += 1.0
    entropy_r = 0.0
    for cnt in sums.values():
        prob = cnt / len(mat)
        entropy_r -= prob * np.log2(prob)
    return entropy_r


def condition_entropy(mat, i):
    """
    get conditional entropy of i-th col.
    """
    elems = np.unique(mat[:, i])
    condition_entropy_r = 0.0
    for elem in elems:
        subset = select_except_i(mat, i, elem, lambda x, y: x == y)
        prob = len(subset) / float(len(mat))
        condition_entropy_r += prob * entropy(subset)
    return condition_entropy_r


def choose_information_gain_rate(mat):
    """
    choose best col by information gain rate.
    """
    entropy_m = entropy(mat)
    max_gain_rate, max_index = float('-inf'), None
    for i in range(len(mat[0]) - 1):
        gain = entropy_m - condition_entropy(mat, i)
        if gain == 0:
            continue
        entropy_i = entropy(mat[:, i:i + 1])
        gain_rate = gain / entropy_i
        if gain_rate > max_gain_rate:
            max_gain_rate, max_index = gain_rate, i
    return {'information_gain_rate': max_gain_rate, 'index': max_index}


mat = np.loadtxt('../../Data/train.csv', delimiter=',')
print(choose_information_gain_rate(mat))
