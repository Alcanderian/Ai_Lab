"""
gini module.
"""
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
