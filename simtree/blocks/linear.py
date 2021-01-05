"""Blocks for linear, time-invariant systems"""
import numpy as np
from simtree.model import Block, Port, State, Signal


class LTISystem(Block):
    """
    Implementation of a linear, time-invariant block of the following format::

      dx/dt = system_matrix * x +        input_matrix * u
      y     = output_matrix * x + feed_through_matrix * u

    The matrices ``system_matrix``, ``input_matrix``, ``output_matrix`` and
    ``feed_through_matrix`` define the state and output behaviour of the block.
    """

    def __init__(self,
                 parent,
                 system_matrix,
                 input_matrix,
                 output_matrix,
                 feed_through_matrix):
        Block.__init__(self, parent)

        system_matrix = np.atleast_2d(system_matrix)
        input_matrix = np.atleast_2d(input_matrix)
        output_matrix = np.atleast_2d(output_matrix)
        feed_through_matrix = np.atleast_2d(feed_through_matrix)

        if system_matrix.shape[0] != system_matrix.shape[1]:
            raise ValueError("The system matrix must be square")
        if system_matrix.shape[0] != input_matrix.shape[0]:
            raise ValueError(
                "The height of the system matrix and the input matrix "
                "must be the same")
        if output_matrix.shape[0] != feed_through_matrix.shape[0]:
            raise ValueError(
                "The height of the output matrix and the "
                "feed-through matrix feed_through_matrix must be the same")
        if system_matrix.shape[1] != output_matrix.shape[1]:
            raise ValueError(
                "The width of the system matrix and the output matrix"
                "must be the same")
        if input_matrix.shape[1] != feed_through_matrix.shape[1]:
            raise ValueError(
                "The width of the output matrix and the feed-through "
                "matrix must be the same")

        self.system_matrix = system_matrix
        self.input_matrix = input_matrix
        self.output_matrix = output_matrix
        self.feed_through_matrix = feed_through_matrix

        self.input = Port(self, shape=self.input_matrix.shape[1])
        self.state = State(self,
                           state=self.system_matrix.shape[0],
                           derivative_function=self.state_derivative)
        self.output = Signal(self,
                             shape=self.output_matrix.shape[0],
                             function=self.output_function)

    def state_derivative(self, data):
        """Calculates the state derivative for the system"""
        state = data.states[self.state]
        inputs = data.states[self.input]
        return (self.system_matrix @ state) + (self.input_matrix @ inputs)

    def output_function(self, data):
        """Calculates the output for the system"""
        state = data.states[self.state]
        inputs = data.states[self.input]
        return (self.output_matrix @ state) + (self.feed_through_matrix @ inputs)


class Gain(Block):
    """
    A simple linear gain block.

    Provides the input scaled by the constant gain as output.
    """

    def __init__(self, parent, k):
        Block.__init__(self, parent)
        self.k = np.asarray(k)

        self.input = Port(self, shape=k.shape[0])
        self.output = Signal(self,
                             shape=k.shape[1],
                             function=self.output_function)

    def output_function(self, data):
        """Calculates the output for the system"""
        return self.k @ data.inputs[self.input]


class Sum(Block):
    """
    A linear weighted sum block.

    This block may have a number of inputs which are interpreted as vectors of
    common dimension. The output of the block is calculated as the weighted
    sum of the inputs.

    The ``channel_weights`` give the factors by which the individual channels are
    weighted in the sum.
    """

    def __init__(self,
                 parent,
                 channel_weights,
                 output_size=1):
        Block.__init__(self, parent)

        self.channel_weights = np.asarray(channel_weights)
        self.output_size = output_size

        self.inputs = [Port(self, shape=self.output_size)
                       for port_idx in range(self.channel_weights.shape[0])]
        self.output = Signal(self,
                             shape=self.output_size,
                             function=self.output_function)

    def output_function(self, data):
        """Calculates the output for the system"""
        inputs = np.array(shape=(self.output_size, len(self.inputs)))
        for port_idx in range(len(self.inputs)):
            inputs[:, port_idx] = data.inputs[self.inputs[port_idx]]
        return self.channel_weights @ inputs
