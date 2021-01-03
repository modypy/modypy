"""Tests for the simtree.model.states module"""
import numpy as np

from simtree.model import ModelContext, Block
from simtree.model.states import State


class SimpleBlock(Block):
    """A simple block with a few states"""
    omega = State(shape=3)
    dcm_matrix = State(shape=(3, 3), initial_value=np.eye(3))

    @omega.derivative
    def omega_dot(self, time, state, inputs):
        return inputs[0]


def test_state_basics():
    """Test some basic functionality of states and state instances"""
    context = ModelContext()

    assert SimpleBlock.omega.size == 3
    assert SimpleBlock.dcm_matrix.size == 9
    assert SimpleBlock.omega.derivative_function == SimpleBlock.omega_dot

    block = SimpleBlock(context)
    assert block.omega.state_index is not None
    assert block.omega.state == SimpleBlock.omega
