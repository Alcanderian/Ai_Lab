"""
implement of de tree.
"""
import numpy as np
import data_op
from toolkit import id3


def max_tag(mat):
    """
    tag appear most.
    """
    sums = {}
    for vec in mat:
        if vec[-1] not in sums:
            sums[vec[-1]] = 0.0
        sums[vec[-1]] += 1.0
    keys, values = list(sums.keys()), list(sums.values())
    return keys[np.argmax(values)]


def is_leaf(node):
    """
    retrun True if node is leaf.
    """
    return 'children' not in node


def train(mat, heads=None, toolkit=id3):
    """
    struct of non-leaf:
    {'toolkit': toolkit, 'args': args, 'children': {label: child_node,...}}
    struct of leaf:
    {'tag': tag}

    param:
    :heads: original column indices, do not pass it manually.
    """
    if len(np.unique(mat[:, -1])) == 1:
        return {'tag': mat[0][-1]}
    if len(mat[0]) == 1:
        return {'tag': max_tag(mat)}
    args = toolkit.choose(mat)
    if args['index'] is None:
        return {'tag': max_tag(mat)}
    if heads is None:
        heads = np.array(range(len(mat[0]) - 1))
    subsets, partitions = toolkit.subset(mat, args)
    subheads = data_op.select_except_i(
        [heads], args['index'], heads[args['index']], data_op.eq)[0]
    # assign args['index'] to index of origin mat.
    args['index'] = heads[args['index']]
    node = {'toolkit': toolkit, 'args': args, 'children': {}}
    for subset, partition in zip(subsets, partitions):
        node['children'][partition] = train(subset, subheads, toolkit)
    return node


def predict(tree, mat):
    """
    predict tags of vec in mat.
    """
    tags = []
    for vec in mat:
        node = tree
        while not is_leaf(node):
            args = node['args']
            val = vec[args['index']]
            partition = node['toolkit'].partition(val, args)
            node = node['children'][partition]
        tags.append(node['tag'])
    return tags
