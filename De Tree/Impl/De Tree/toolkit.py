"""
decision tree's toolkit
"""
import gini as gi
import entropy as en
import binarytools as bt
import complextools as ct


class complex:
    """
    complex tree warpper.
    """
    class id3:
        """
        warpper, complex-ID3 module.
        """
        def choose(mat):
            return ct.choose(mat, en.entropy, 'gain')


        subset = ct.subset
        partition = ct.partition


    class c45:
        """
        warpper, complex-C4.5 module.
        """
        def choose(mat):
            return ct.choose(mat, en.entropy, 'gain_rate')


        subset = ct.subset
        partition = ct.partition


    class cart:
        """
        warpper, complex-CART module.
        """
        def choose(mat):
            return ct.choose(mat, gi.gini, 'gain')


        subset = ct.subset
        partition = ct.partition


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
                return bt.choose(mat, en.entropy, bt.splits_continuous, 'gain_rate')


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
                return bt.choose(mat, en.entropy, bt.splits_smart_discrete, 'gain')


            subset = bt.subset
            partition = bt.partition


        class c45:
            """
            warpper, binary-smart-discrete-C4.5 module.
            """
            def choose(mat):
                return bt.choose(mat, en.entropy, bt.splits_smart_discrete, 'gain_rate')


            subset = bt.subset
            partition = bt.partition


        class cart:
            """
            warpper, binary-smart-discrete-CART module.
            """
            def choose(mat):
                return bt.choose(mat, gi.gini, bt.splits_smart_discrete, 'gain')


            subset = bt.subset
            partition = bt.partition
