"""
implement of de tree.
"""
import numpy as np
import datatools as dt
from toolkit import complex


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


def train(mat, heads=None, toolkit=complex.id3):
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
    subheads = dt.select_except_i(
        [heads], args['index'], heads[args['index']], dt.eq)[0]
    # assign args['index'] to index of origin mat.
    args['index'] = heads[args['index']]
    node = {'toolkit': toolkit, 'args': args, 'children': {}}
    for subset, partition in zip(subsets, partitions):
        node['children'][partition] = train(subset, subheads, toolkit)
    return node


def plotable(tree):
    """
    let the tree be plotable by treePlotter.py
    """
    if is_leaf(tree):
        return tree['tag']
    index = tree['args']['index']
    node = {index: tree['children']}
    for key in node[index].keys():
        node[index][key] = plotable(node[index][key])
    return node


def nearest_neighbor(label, neighbors):
    """
    get the nearest neighbor of the unknown label,
    only use in complex tree's prediction.
    """
    neighbors = list(neighbors)
    distances = np.abs(np.array(neighbors) - label)
    index = np.argmin(distances)
    return neighbors[index]


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
            if partition not in node['children']:
                partition = nearest_neighbor(
                    partition, node['children'].keys())
            node = node['children'][partition]
        tags.append(node['tag'])
    return tags


def validation(tree, mat):
    actual, result = mat[:, -1], predict(tree, mat)
    sums = {1: {1: 0.0, -1: 0.0}, -1: {1: 0.0, -1: 0.0}}
    for i, j in zip(actual, result):
        sums[int(i)][int(j)] += 1.0
    eval = {'accuary': (sums[1][1] + sums[-1][-1]) / len(actual),
            'precision': sums[1][1] / (sums[1][1] + sums[-1][1]),
            'recall': sums[1][1] / (sums[1][1] + sums[1][-1])}
    eval['f1'] = (2 * eval['precision'] * eval['recall'] 
                  / (eval['precision'] + eval['recall']))
    return eval
