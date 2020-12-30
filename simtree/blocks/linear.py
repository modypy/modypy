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
