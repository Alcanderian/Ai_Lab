"""
decision tree's toolkit
"""
import gini as gi
import entropy as en


class id3:
    """
    ID3 module.
    """
    choose = en.choose_information_gain
    subset = en.entropy_subset
    partition = en.entropy_partition


class c45:
    """
    C4.5 module.
    """
    choose = en.choose_information_gain_rate
    subset = en.entropy_subset
    partition = en.entropy_partition


class ccart:
    """
    continuous-CART module.
    """
    def choose(mat):
        """
        choose best col and best split by continuous gini slipt.
        """
        return gi.choose_gini(mat, gi.gini_get_splits_continuous)


    subset = gi.gini_subset
    partition = gi.gini_partition


class sdcart:
    """
    smart-discrete-CART module.
    """
    def choose(mat):
        """
        choose best col and best split by discrete gini slipt.
        """
        return gi.choose_gini(mat, gi.gini_get_splits_smart_discrete)


    subset = gi.gini_subset
    partition = gi.gini_partition
