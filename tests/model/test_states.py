"""Tests for the `modypy.model.states`` package"""
from unittest.mock import Mock

import numpy as np

from modypy.model import System, State


def test_state():
    """Test the ``State`` class"""

    system = System()
    state_a = State(system, derivative_function=(lambda data: 0))
    state_b = State(
        system, shape=3, derivative_function=(lambda data: np.zeros(3))
    )
    state_c = State(
        system, shape=(3, 3), derivative_function=(lambda data: np.zeros(3, 3))
    )
    state_d = State(
        system, derivative_function=(lambda data: 0), initial_condition=1
    )

    # Check the sizes
    assert state_a.size == 1
    assert state_b.size == 3
    assert state_c.size == 9
    assert state_d.size == 1

    # Test the slice property
    assert state_c.state_slice == slice(
        state_c.state_index, state_c.state_index + state_c.size
    )


def test_state_access():
    """Test whether calling a ``State`` object calls the ``get_state_value`` and
    ``set_state_value`` methods of the provider object"""

    system = System()
    state = State(system)

    provider = Mock()

    # Check read access
    state(provider)
    provider.get_state_value.assert_called_with(state)

    # Check write access
    provider.reset_mock()
    state.set_value(provider, 10)
    provider.set_state_value.assert_called_with(state, 10)
