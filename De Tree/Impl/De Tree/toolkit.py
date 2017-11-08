"""
decision tree's toolkit
"""
import gini as gi
import entropy as en
import binarytools as bt
import multipletools as mt


class multiple:
    """
    multiple tree warpper.
    """
    class id3:
        """
        warpper, multiple-ID3 module.
        """
        def choose(mat):
            return mt.choose(mat, en.entropy, 'gain')

        subset = mt.subset
        partition = mt.partition

    class c45:
        """
        warpper, multiple-C4.5 module.
        """
        def choose(mat):
            return mt.choose(mat, en.entropy, 'gain_rate')

        subset = mt.subset
        partition = mt.partition

    class cart:
        """
        warpper, multiple-CART module.
        """
        def choose(mat):
            return mt.choose(mat, gi.gini, 'gain')

        subset = mt.subset
        partition = mt.partition


class binary:
    """
    binary tree warpper.
    """
    class continuous:
        """
        split binary continuous.
        """
        class id3:
            """
            warpper, binary-continuous-ID3 module.
            """
            def choose(mat):
                return bt.choose(mat, en.entropy, bt.splits_continuous, 'gain')

            subset = bt.subset
            partition = bt.partition

        class c45:
            """
            warpper, binary-continuous-C4.5 module.
            """
            def choose(mat):
                return bt.choose(mat, en.entropy,
                                 bt.splits_continuous, 'gain_rate')

            subset = bt.subset
            partition = bt.partition

        class cart:
            """
            warpper, binary-continuous-CART module.
            """
            def choose(mat):
                return bt.choose(mat, gi.gini, bt.splits_continuous, 'gain')

            subset = bt.subset
            partition = bt.partition

    class discrete:
        """
        split binary discrete smartly.
        """
        class id3:
            """
            warpper, binary-smart-discrete-ID3 module.
            """
            def choose(mat):
                return bt.choose(mat, en.entropy,
                                 bt.splits_smart_discrete, 'gain')

            subset = bt.subset
            partition = bt.partition

        class c45:
            """
            warpper, binary-smart-discrete-C4.5 module.
            """
            def choose(mat):
                return bt.choose(mat, en.entropy,
                                 bt.splits_smart_discrete, 'gain_rate')

            subset = bt.subset
            partition = bt.partition

        class cart:
            """
            warpper, binary-smart-discrete-CART module.
            """
            def choose(mat):
                return bt.choose(mat, gi.gini,
                                 bt.splits_smart_discrete, 'gain')

            subset = bt.subset
            partition = bt.partition
