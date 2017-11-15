"""
implement of de tree.
"""
import numpy as np
import datatools as dt
from toolkit import multiple


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


def train(mat, heads=None, toolkit=multiple.id3, 
          max_depth=float('inf'), depth=0):
    """
    :struct of non-leaf: {
        'toolkit': toolkit,
        'args': args,
        'children: {
            label: child_node,...
        }
        'tag': tag by voting
    }

    :struct of leaf:
    {'tag': tag}

    :param:
    heads: original column indices, do not pass it manually.
    depth: auto increment value, do not pass it manually.

    we should reuse the column when there is continuous split.
    """
    if max_depth < 0:
        return None
    if len(np.unique(mat[:, -1])) == 1:
        return {'tag': int(mat[0][-1])}
    if len(mat[0]) == 1:
        return {'tag': int(max_tag(mat))}
    args = toolkit.choose(mat)
    index = args['index']
    # [!]: or max_depth == depth
    if index is None or max_depth == depth:
        return {'tag': int(max_tag(mat))}
    if heads is None:
        heads = np.array(range(len(mat[0]) - 1))
    subsets, parts = toolkit.subset(mat, args)
    if dt.is_continuous_split(args):
        subheads = heads.copy()
    else:
        subheads = np.append(heads[:index], heads[index + 1:])
    # assign args['index'] to index of origin mat.
    args['index'] = heads[args['index']]
    node = {'toolkit': toolkit, 'args': args, 
            'children': {}, 'tag': int(max_tag(mat))}
    for i in range(len(subsets)):
        node['children'][parts[i]] = train(subsets[i], subheads,
                                           toolkit, max_depth, depth + 1)
    return node


def plotable(tree):
    """
    let the tree be plotable by treePlotter.py
    """
    if is_leaf(tree):
        return tree['tag']
    index = tree['args']['index']
    node = {index: tree['children'].copy()}
    for key in node[index].keys():
        node[index][key] = plotable(node[index][key])
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
            if partition not in node['children']:
                break
            node = node['children'][partition]
        tags.append(node['tag'])
    return np.array(tags, int)


def validation(tree, mat):
    """
    validate and evalute the tree.
    """
    actual, result = mat[:, -1], predict(tree, mat)
    return dt.evalution(actual, result)
