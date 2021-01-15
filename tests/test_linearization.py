import pytest

import numpy as np
import numpy.testing as npt

from modypy.linearization import find_steady_state, system_jacobian
from modypy.model import Evaluator, System, OutputPort, State, InputSignal
from fixtures.models import first_order_lag, first_order_lag_no_input, damped_oscillator, lti_gain


@pytest.fixture(params=[3, 5, 7, 9, 11])
def interpolation_order(request):
    return request.param


@pytest.mark.parametrize(
    "param",
    [
        lti_gain(1),                    # 1 input, 1 output, no state

        # 1 input, 1 output, 1 state
        first_order_lag(time_constant=1, initial_value=10),

        # no input, with state
        first_order_lag_no_input(time_constant=1, initial_value=10),

        # 1 input, 2 states, 1 output
        damped_oscillator(mass=100, spring_coefficient=1., damping_coefficient=20),   # critically damped
        damped_oscillator(mass=100, spring_coefficient=0.5, damping_coefficient=20),  # overdamped
        damped_oscillator(mass=100, spring_coefficient=2., damping_coefficient=20),   # underdamped
    ])
def test_steady_state_linearisation(param, interpolation_order):
    system, lti, sim_time = param

    # Find the steady state of the system
    sol, x0, u0 = find_steady_state(system,
                                    time=0,
                                    method="lm")

    assert sol.success
    assert x0.size == system.num_states
    assert u0.size == system.num_inputs

    evaluator = Evaluator(time=0, system=system, state=x0, inputs=u0)
    npt.assert_almost_equal(evaluator.state_derivative,
                            np.zeros(system.num_states))
    npt.assert_almost_equal(evaluator.outputs,
                            np.zeros(system.num_outputs))

    # Linearize the system
    A, B, C, D = system_jacobian(system, 0, x0, u0, order=interpolation_order)

    # Check the matrices
    npt.assert_almost_equal(A, lti.system_matrix)
    npt.assert_almost_equal(B, lti.input_matrix)
    npt.assert_almost_equal(C, lti.output_matrix)
    npt.assert_almost_equal(D, lti.feed_through_matrix)

    # Get the full jacobian
    jac = system_jacobian(system,
                          0,
                          x0,
                          u0,
                          single_matrix=True,
                          order=interpolation_order)
    A = jac[:system.num_states, :system.num_states]
    B = jac[:system.num_states, system.num_states:]
    C = jac[system.num_states:, :system.num_states]
    D = jac[system.num_states:, system.num_states:]
    npt.assert_almost_equal(A, lti.system_matrix)
    npt.assert_almost_equal(B, lti.input_matrix)
    npt.assert_almost_equal(C, lti.output_matrix)
    npt.assert_almost_equal(D, lti.feed_through_matrix)


def test_output_only():
    system = System()
    OutputPort(system)

    # Try to find the steady x_ref of the system
    with pytest.raises(ValueError):
        find_steady_state(system, time=0)

    # Try to linearize the system
    with pytest.raises(ValueError):
        system_jacobian(system, time=0, x_ref=[], u_ref=[])


def test_steady_state_under_defined():
    system = System()
    InputSignal(system)
    State(owner=system, shape=2, derivative_function=None)

    with pytest.raises(ValueError):
        find_steady_state(system, time=0)
