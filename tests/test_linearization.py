# pylint: disable=redefined-outer-name,missing-module-docstring
import numpy as np
import pytest
from fixtures.models import (
    damped_oscillator,
    first_order_lag,
    first_order_lag_no_input,
)
from modypy.linearization import (
    LinearizationConfiguration,
    OutputDescriptor,
    system_jacobian,
)
from modypy.model import Signal, System
from modypy.steady_state import SteadyStateConfiguration, find_steady_state
from numpy import testing as npt


@pytest.fixture(params=[3, 5, 7, 9, 11])
def interpolation_order(request):
    return request.param


@pytest.mark.parametrize(
    "param",
    [
        # 1 input, 1 output, 1 state
        first_order_lag(time_constant=1, initial_value=10),
        # no input, with state
        first_order_lag_no_input(time_constant=1, initial_value=10),
        # 1 input, 2 states, 1 output
        damped_oscillator(
            mass=100, spring_coefficient=1.0, damping_coefficient=20
        ),  # critically damped
        damped_oscillator(
            mass=100, spring_coefficient=0.5, damping_coefficient=20
        ),  # overdamped
    ],
)
def test_lti_linearization(param, interpolation_order):
    system, lti, _, outputs = param

    # Find the steady state of the system
    steady_state_config = SteadyStateConfiguration(system)
    # Constrain all outputs to zero
    for output in outputs:
        steady_state_config.ports[output].lower_limit = 0
        steady_state_config.ports[output].upper_limit = 0
    # Find the steady state
    sol = find_steady_state(steady_state_config)

    assert sol.success
    assert sol.state.size == system.num_states
    assert sol.inputs.size == system.num_inputs

    npt.assert_allclose(
        system.state_derivative(sol.system_state),
        np.zeros(system.num_states),
        rtol=0,
        atol=1e-5,
    )

    # Set up the configuration for linearization at the steady state
    jacobian_config = LinearizationConfiguration(
        system, time=0, state=sol.state, inputs=sol.inputs
    )
    # Set up the single LTI output
    output = OutputDescriptor(jacobian_config, lti.output)
    jacobian_config.interpolation_order = interpolation_order

    # Linearize the system
    A, B, C, D = system_jacobian(jacobian_config)

    # Check the matrices
    npt.assert_almost_equal(A, np.atleast_2d(lti.system_matrix))
    npt.assert_almost_equal(B, np.atleast_2d(lti.input_matrix))
    npt.assert_almost_equal(C, np.atleast_2d(lti.output_matrix))
    npt.assert_almost_equal(D, np.atleast_2d(lti.feed_through_matrix))
    npt.assert_almost_equal(
        C[output.output_slice], np.atleast_2d(lti.output_matrix)
    )
    npt.assert_almost_equal(
        D[output.output_slice], np.atleast_2d(lti.feed_through_matrix)
    )

    # Get the full jacobian
    jac = system_jacobian(jacobian_config, single_matrix=True)
    A = jac[: system.num_states, : system.num_states]
    B = jac[: system.num_states, system.num_states :]
    C = jac[system.num_states :, : system.num_states]
    D = jac[system.num_states :, system.num_states :]
    npt.assert_almost_equal(A, np.atleast_2d(lti.system_matrix))
    npt.assert_almost_equal(B, np.atleast_2d(lti.input_matrix))
    npt.assert_almost_equal(C, np.atleast_2d(lti.output_matrix))
    npt.assert_almost_equal(D, np.atleast_2d(lti.feed_through_matrix))
    npt.assert_almost_equal(
        C[output.output_slice], np.atleast_2d(lti.output_matrix)
    )
    npt.assert_almost_equal(
        D[output.output_slice], np.atleast_2d(lti.feed_through_matrix)
    )

    # Set up the configuration for linearization at default input and state
    jacobian_config_def = LinearizationConfiguration(system, time=0)
    # Set up the single LTI output
    output = OutputDescriptor(jacobian_config_def, lti.output)
    jacobian_config_def.interpolation_order = interpolation_order

    # Linearize the system, but get the individual matrices
    A, B, C, D = system_jacobian(jacobian_config_def)

    # Check the matrices
    npt.assert_almost_equal(A, np.atleast_2d(lti.system_matrix))
    npt.assert_almost_equal(B, np.atleast_2d(lti.input_matrix))
    npt.assert_almost_equal(C, np.atleast_2d(lti.output_matrix))
    npt.assert_almost_equal(D, np.atleast_2d(lti.feed_through_matrix))
    npt.assert_almost_equal(
        C[output.output_slice], np.atleast_2d(lti.output_matrix)
    )
    npt.assert_almost_equal(
        D[output.output_slice], np.atleast_2d(lti.feed_through_matrix)
    )

    # Linearize the system, but get the structure
    jacobian = system_jacobian(jacobian_config_def, single_matrix="struct")

    # Check the matrices
    npt.assert_almost_equal(
        jacobian.system_matrix, np.atleast_2d(lti.system_matrix)
    )
    npt.assert_almost_equal(
        jacobian.input_matrix, np.atleast_2d(lti.input_matrix)
    )
    npt.assert_almost_equal(
        jacobian.output_matrix, np.atleast_2d(lti.output_matrix)
    )
    npt.assert_almost_equal(
        jacobian.feed_through_matrix, np.atleast_2d(lti.feed_through_matrix)
    )
    npt.assert_almost_equal(
        jacobian.output_matrix[output.output_slice],
        np.atleast_2d(lti.output_matrix),
    )
    npt.assert_almost_equal(
        jacobian.feed_through_matrix[output.output_slice],
        np.atleast_2d(lti.feed_through_matrix),
    )


def test_empty_system():
    """Test whether the linearization algorithm properly flags a system without
    states and inputs"""

    # Create a system and add a signal to it, so we can check whether the
    # algorithm is actually checking states and inputs instead of signals or
    # outputs
    system = System()
    output_signal = Signal()
    config = LinearizationConfiguration(system)
    OutputDescriptor(config, output_signal)

    with pytest.raises(ValueError):
        system_jacobian(config, single_matrix=False)
