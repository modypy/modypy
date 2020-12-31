import numpy as np
from simtree.blocks import LeafBlock


class LTISystem(LeafBlock):
    """
    Implementation of a linear, time-invariant block of the format

      dx/dt = A*x+B*u
      y = C*x+D*u

    The matrices A,B,C and D define the state and output behaviour of the block.
    """

    def __init__(self, A, B, C, D, **kwargs):
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)
        C = np.atleast_2d(C)
        D = np.atleast_2d(D)
        LeafBlock.__init__(self,
                           num_states=A.shape[1],
                           num_inputs=B.shape[1],
                           num_outputs=C.shape[0],
                           **kwargs)
        if A.shape[0] != A.shape[1]:
            raise ValueError("The state update matrix A must be square")
        if A.shape[0] != B.shape[0]:
            raise ValueError(
                "The height of the state update matrix A and the input matrix B "
                "must be the same")
        if C.shape[0] != D.shape[0]:
            raise ValueError(
                "The height of the state output matrix C and the "
                "feed-through matrix D must be the same")
        if A.shape[1] != C.shape[1]:
            raise ValueError(
                "The width of the state update matrix A and the state output "
                "matrix C must be the same")
        if B.shape[1] != D.shape[1]:
            raise ValueError(
                "The width of the state output matrix C and the feed-through "
                "matrix D must be the same")
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def state_update_function(self, t, *args):
        if self.num_states > 0:
            dxdt = np.matmul(self.A, args[0])
            if self.num_inputs > 0:
                dxdt += np.matmul(self.B, args[1])
            return dxdt
        return np.empty(0)

    def output_function(self, t, *args):
        if self.num_states > 0 and self.num_inputs > 0:
            return np.matmul(self.C, args[0]) + np.matmul(self.D, args[1])
        if self.num_states > 0:
            return np.matmul(self.C, args[0])
        if self.num_inputs > 0:
            return np.matmul(self.D, args[0])
        return np.zeros(self.num_outputs)


class Gain(LeafBlock):
    """
    A simple linear gain block.

    Provides the input scaled by the constant gain as output.
    """

    def __init__(self, k, **kwargs):
        k = np.asarray(k)
        LeafBlock.__init__(
            self, num_inputs=k.shape[1], num_outputs=k.shape[0], **kwargs)
        self.k = k

    def output_function(self, t, inputs):
        return np.matmul(self.k, inputs)


class Sum(LeafBlock):
    """
    A linear weighted sum block.

    This block may have a number of inputs which are interpreted a vectors of
    common dimension. The output of the block is calculated as the weighted
    sum of the inputs.

    The block has `channel_dim` outputs, which represent the elements of the
    weighted vector sum of the inputs.
    In that case, inputs `0:channel_dim` represent the first vector, inputs
    `channel_dim:2*channel_dim` represent the second vector, etc.

    The `channel_weights` give the factors by which the individual channels are
    weighted in the sum.
    """
    def __init__(self,
                 channel_weights=[1, 1],
                 channel_dim=1,
                 **kwargs):
        channel_weights = np.asarray(channel_weights)
        LeafBlock.__init__(self,
                           num_inputs=channel_weights.size * channel_dim,
                           num_outputs=channel_dim,
                           **kwargs)
        self.channel_weights = channel_weights
        self.channel_dim = channel_dim

    def output_function(self, time, inputs):
        inputs = np.asarray(inputs).reshape(-1, self.channel_dim)
        return np.matmul(self.channel_weights, inputs)
