"""Provides some simple source blocks"""
import numpy as np

from simtree.model import Block, Signal


class Constant(Block):
    """
    A constant source block.

    Provides the given constant value as output.
    """

    def __init__(self, parent, value):
        Block.__init__(self, parent)
        self.value = np.asarray(value).flatten()

        self.output = Signal(self,
                             shape=self.value.shape,
                             function=self.output_function)

    def output_function(self, data):
        """Calculates the output of this block"""
        return self.value


class SourceFromCallable(Block):
    """A source block providing the output from a callable."""

    def __init__(self, fun, **kwargs):
        LeafBlock.__init__(self, **kwargs)
        self.fun = fun

    def output_function(self, *args):
        return self.fun(*args)
