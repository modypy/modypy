import pytest

import numpy as np
import numpy.testing as npt

from simtree.blocks import LeafBlock
from simtree.linearization import find_steady_state, system_jacobian
from fixtures.models import first_order_lag, first_order_lag_no_input, damped_oscillator, lti_gain


@pytest.fixture(params=[3, 5, 7, 9, 11])
def interpolation_order(request):
    return request.param

@pytest.mark.parametrize(
    "param,x0,u0",
    [
        (lti_gain(1), None, None),                    # 1 input, 1 output, no x_ref

        # 1 input, 1 output, 1 x_ref
        (first_order_lag(T=1, x0=10), None, None),
        (first_order_lag(T=1, x0=10), 10, None),
        (first_order_lag(T=1, x0=10), None, 10),

        # no input, with x_ref
        (first_order_lag_no_input(T=1, x0=10), None, None),

        # 1 input, 2 states, 1 output
        (damped_oscillator(m=100, k=1., d=20), None, None),  # critically damped
        (damped_oscillator(m=100, k=0.5, d=20), None, None),  # overdamped
        (damped_oscillator(m=100, k=2., d=20), None, None),  # underdamped
    ])
def test_steady_state_linearisation(param, x0, u0, interpolation_order):
    system, sim_time = param

    # Find the steady x_ref of the system
    sol, x0, u0 = find_steady_state(system, time=0, x_start=x0, u_start=u0)

    assert sol.success
    assert x0.size == system.num_states
    assert u0.size == system.num_inputs

    if system.num_states > 0 and system.num_inputs > 0:
        dxdt = system.state_update_function(0, x0, u0)
        y = system.output_function(0, x0, u0)
    elif system.num_states > 0:
        dxdt = system.state_update_function(0, x0)
        y = system.output_function(0, x0)
    elif system.num_inputs > 0:
        dxdt = np.empty(0)
        y = system.output_function(0, u0)
    else:
        dxdt = np.empty(0)
        y = np.empty(0)

    npt.assert_almost_equal(dxdt, np.zeros(system.num_states))
    npt.assert_almost_equal(y, np.zeros(system.num_outputs))

    # Linearize the system
    A, B, C, D = system_jacobian(system, 0, x0, u0, order=interpolation_order)
    npt.assert_almost_equal(A, system.A)
    npt.assert_almost_equal(B, system.B)
    npt.assert_almost_equal(C, system.C)
    npt.assert_almost_equal(D, system.D)


def test_output_only():
    system = LeafBlock(num_outputs=1)

    # Try to find the steady x_ref of the system
    with pytest.raises(ValueError):
        sol, x0, u0 = find_steady_state(system, time=0)

    # Try to linearize the system
    with pytest.raises(ValueError):
        A, B, C, D = system_jacobian(system, time=0, x_ref=[], u_ref= [])


def test_steady_state_underdefined():
    system = LeafBlock(num_inputs=2, num_states=1)
    with pytest.raises(ValueError):
        sol, x0, u0 = find_steady_state(system, time=0)
