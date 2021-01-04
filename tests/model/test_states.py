"""Tests for the simtree.model.states module"""
import numpy as np

from simtree.model import System, Block, Port
from simtree.model.states import State


class SimpleBlock(Block):
    """A simple block with a few states"""
    def __init__(self, parent):
        Block.__init__(self, parent)
        self.omega_dot = Port(self, shape=3)
        self.omega = State(self, shape=3, derivative_function=self.omega_dot)
        self.dcm = State(self, shape=(3, 3), derivative_function=self.dcm_dot, initial_condition=np.eye(3))

    def omega_dot(self):
        pass

    def dcm_dot(self):
        pass


def test_state_basics():
    """Test some basic functionality of states and state instances"""
    system = System()
    block = SimpleBlock(system)

    assert block.omega.size == 3
    assert block.dcm.size == 9
