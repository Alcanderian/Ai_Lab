"""
implement of de tree.
"""
import numpy as np
import data_op
from toolkit import id3


def max_class(mat):
    """
    class appear most.
    """
    class_sum = {}
    for vec in mat:
        if vec[-1] not in class_sum:
            class_sum[vec[-1]] = 0.0
        class_sum[vec[-1]] += 1.0
    keys, values = list(class_sum.keys()), list(class_sum.values())
    return keys[np.argmax(values)]

def is_leaf(node):
    """
    retrun True if node is leaf.
    """
    return not isinstance(node, dict)


def train(mat, heads=None, toolkit=id3):
    """
    struct of node:
    {'toolkit': toolkit, 'args': args, 'node': {label: child_node,...}} 
    or
    is not {}
    """
    if len(np.unique(mat[:, -1])) == 1:
        return mat[0][-1]
    if len(mat[0]) == 1:
        return max_class(mat)
    args = toolkit.choose(mat)
    if args['index'] is None:
        return max_class(mat)
    if heads is None:
        heads = list(range(0, len(mat[0]) - 1))
    subsets, partitions = toolkit.subset(mat, args)
    subheads = data_op.select_except_i(
        [heads], args['index'], heads[args['index']], lambda x, y: x == y)[0]
    # assign args['index'] to index of origin mat.
    args['index'] = int(heads[args['index']])
    node = {'toolkit': toolkit, 'args': args, 'node': {}}
    for subset, partition in zip(subsets, partitions):
        node['node'][partition] = train(subset, subheads, toolkit)
    return node


def test(tree, mat):
    return
