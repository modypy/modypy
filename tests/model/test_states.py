"""Tests for the `simtree.model.states`` package"""
import numpy as np
import numpy.testing as npt

from simtree.model import System
from simtree.model.evaluation import Evaluator
from simtree.model.states import State, SignalState


def test_state():
    """Test the ``State`` class"""

    system = System()
    state_a = State(system,
                    derivative_function=(lambda data: 0))
    state_b = State(system, shape=3,
                    derivative_function=(lambda data: np.zeros(3)))
    state_c = State(system, shape=(3, 3),
                    derivative_function=(lambda data: np.zeros(3, 3)))
    state_d = State(system,
                    derivative_function=(lambda data: 0),
                    initial_condition=1)

    # Check the sizes
    assert state_a.size == 1
    assert state_b.size == 3
    assert state_c.size == 9
    assert state_d.size == 1

    # Test the slice property
    assert state_c.state_slice == slice(state_c.state_index,
                                        state_c.state_index + state_c.size)


def test_signal_state():
    """Test the ``SignalState`` class"""

    system = System()
    state_a = SignalState(system,
                          derivative_function=(lambda data: 0))

    # Test the output
    evaluator = Evaluator(time=0, system=system)
    npt.assert_almost_equal(evaluator.get_port_value(state_a),
                            np.zeros(1))
