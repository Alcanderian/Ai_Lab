"""
entropy module.
"""
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
