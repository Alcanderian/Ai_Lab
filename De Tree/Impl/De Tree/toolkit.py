"""
decision tree's toolkit
"""
import gini as gi
import entropy as en


class id3:
    def choose(mat):
        return en.choose_information_gain(mat)


    def subset(mat, args):
        return en.entropy_subset(mat, args)


    def partition(val, args):
        return en.entropy_partition(val, args)


class c45:
    def choose(mat):
        return en.choose_information_gain_rate(mat)


    def subset(mat, args):
        return en.entropy_subset(mat, args)


    def partition(val, args):
        return en.entropy_partition(val, args)


class ccart:
    def choose(mat):
        return gi.choose_gini_continuous(mat)


    def subset(mat, args):
        return gi.gini_continuous_subset(mat, args)


    def partition(val, args):
        return gi.gini_continuous_partition(val, args)