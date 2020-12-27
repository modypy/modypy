from simtree.blocks import LeafBlock
import numpy as np


class Constant(LeafBlock):
    """
    A constant source block.

    Provides the given constant value as output.
    """

    def __init__(self, value, **kwargs):
        LeafBlock.__init__(self, **kwargs)
        self.value = np.asarray(value).flatten()
        self.num_outputs = self.value.size

    def output_function(self, t):
        return self.value


class SourceFromCallable(LeafBlock):
    """A source block providing the output from a callable."""

    def __init__(self, fun, **kwargs):
        LeafBlock.__init__(self, **kwargs)
        self.fun = fun

    def output_function(self, *args):
        return self.fun(*args)
