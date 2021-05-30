"""
Test the steady-state determination algorithm.
"""
import numpy as np
import pytest
from modypy.blocks.aerodyn import Propeller
from modypy.blocks.linear import sum_signal
from modypy.blocks.sources import constant
from modypy.model import InputSignal, SignalState, State, System
from modypy.steady_state import SteadyStateConfiguration, find_steady_state
from numpy import testing as npt


def water_tank_model(inflow_area=0.01,
                     outflow_area=0.02,
                     tank_area=0.2,
                     target_height=5,
                     initial_condition=None,
                     initial_guess=None,
                     max_inflow_velocity=None):
    """Create a steady-state configuration for a water tank"""

    g = 9.81  # Gravity

    # Create a new system
    system = System()

    # Model the inflow
    inflow_velocity = InputSignal(system)

    # Model the height state
    def height_derivative(data):
        """Calculate the time derivative of the height"""

        return (inflow_area * inflow_velocity(data)
                - outflow_area * np.sqrt(2 * g * height_state(data))
                ) / tank_area

    height_state = SignalState(system, derivative_function=height_derivative)

    # Set up the steady-state configuration
    config = SteadyStateConfiguration(system)
    # Enforce the inflow to be non-negative
    config.inputs[inflow_velocity].lower_bounds = 0
    if max_inflow_velocity is not None:
        config.inputs[inflow_velocity].upper_bounds = max_inflow_velocity
    # Define the target height
    config.states[height_state].lower_bounds = target_height
    config.states[height_state].upper_bounds = target_height

    if initial_condition is not None:
        config.states[height_state].initial_condition = initial_condition
    if initial_guess is not None:
        config.inputs[inflow_velocity].initial_guess = initial_guess

    return config


def propeller_model(thrust_coefficient=0.09,
                    power_coefficient=0.04,
                    density=1.29,
                    diameter=8 * 25.4E-3,
                    moment_of_inertia=5.284E-6,
                    target_thrust=1.5 * 9.81 / 4,
                    maximum_torque=None):
    """Create a steady-state configuration for two propellers"""

    system = System()

    torque_1 = InputSignal(system)
    torque_2 = InputSignal(system)

    density_signal = constant(density)

    propeller_1 = Propeller(system,
                            thrust_coefficient=thrust_coefficient,
                            power_coefficient=power_coefficient,
                            diameter=diameter)
    propeller_2 = Propeller(system,
                            thrust_coefficient=thrust_coefficient,
                            power_coefficient=power_coefficient,
                            diameter=diameter)

    def speed_1_dt(data):
        """Derivative of the speed of the first propeller"""
        tot_torque = torque_1(data) - propeller_1.torque(data)
        return tot_torque / (2 * np.pi * moment_of_inertia)

    def speed_2_dt(data):
        """Derivative of the speed of the second propeller"""
        tot_torque = torque_2(data) - propeller_2.torque(data)
        return tot_torque / (2 * np.pi * moment_of_inertia)

    speed_1 = SignalState(system,
                          derivative_function=speed_1_dt)
    speed_2 = SignalState(system,
                          derivative_function=speed_2_dt)

    propeller_1.density.connect(density_signal)
    propeller_2.density.connect(density_signal)
    propeller_1.speed_rps.connect(speed_1)
    propeller_2.speed_rps.connect(speed_2)

    total_thrust = sum_signal((propeller_1.thrust,
                               propeller_2.thrust))
    total_power = sum_signal((propeller_1.power,
                              propeller_2.power))

    # Set up steady-state configuration
    config = SteadyStateConfiguration(system)
    # Minimize the total power
    config.objective = total_power
    # Constrain the speed to be positive
    config.states[speed_1].lower_bounds = 0
    config.states[speed_2].lower_bounds = 0
    # Constrain the torques to be positive
    config.inputs[torque_1].lower_bounds = 0
    config.inputs[torque_2].lower_bounds = 0
    npt.assert_equal(config.inputs[torque_1].lower_bounds, 0)
    npt.assert_equal(config.inputs[torque_2].lower_bounds, 0)
    if maximum_torque is not None:
        config.inputs[torque_1].upper_bounds = maximum_torque
        config.inputs[torque_2].upper_bounds = maximum_torque

    # Constrain the total thrust
    config.ports[total_thrust].lower_bounds = target_thrust
    config.ports[total_thrust].upper_bounds = target_thrust

    return config


def pendulum(length=1):
    """Create a steady-state model for a pendulum"""

    system = System()

    g = 9.81  # Gravity

    def omega_dt(data):
        """Calculate the acceleration of the pendulum"""

        return np.sin(phi(data)) * g / length

    def phi_dt(data):
        """Calculate the velocity of the pendulum"""

        return omega(data)

    omega = State(system,
                  derivative_function=omega_dt)
    phi = State(system,
                derivative_function=phi_dt)

    # Set up a steady-state configuration
    config = SteadyStateConfiguration(system)
    # Minimize the total energy
    config.objective = (lambda data: (g * length * (1 - np.cos(phi(data))) +
                                      (length * omega(data)) ** 2))
    # Constrain the angular velocity (no steady state)
    config.states[omega].steady_state = False
    # Constrain the angle
    config.states[phi].steady_state = False
    config.states[phi].lower_bounds = -np.pi
    config.states[phi].upper_bounds = np.pi

    return config


def test_state_constraint_access():
    """Test access to state constraint properties"""

    system = System()
    state_1 = State(system)
    state_2 = State(system)
    config = SteadyStateConfiguration(system)

    config.states[state_1].lower_bounds = 10
    config.states[state_2].lower_bounds = 20
    config.states[state_1].upper_bounds = 15
    config.states[state_2].upper_bounds = 25
    config.states[state_2].steady_state = False
    config.states[state_1].initial_condition = 100
    npt.assert_equal(config.states[state_1].lower_bounds, 10)
    npt.assert_equal(config.states[state_2].lower_bounds, 20)
    npt.assert_equal(config.states[state_1].upper_bounds, 15)
    npt.assert_equal(config.states[state_2].upper_bounds, 25)
    assert not config.states[state_2].steady_state
    npt.assert_equal(config.states[state_1].initial_condition, 100)


def test_input_constraint_access():
    """Test access to input constraint properties"""

    system = System()
    input_1 = InputSignal(system)
    input_2 = InputSignal(system)
    config = SteadyStateConfiguration(system)

    config.inputs[input_1].lower_bounds = 10
    config.inputs[input_2].lower_bounds = 20
    config.inputs[input_1].upper_bounds = 15
    config.inputs[input_2].upper_bounds = 25
    config.inputs[input_1].initial_guess = 100
    npt.assert_equal(config.inputs[input_1].lower_bounds, 10)
    npt.assert_equal(config.inputs[input_2].lower_bounds, 20)
    npt.assert_equal(config.inputs[input_1].upper_bounds, 15)
    npt.assert_equal(config.inputs[input_2].upper_bounds, 25)
    npt.assert_equal(config.inputs[input_1].initial_guess, 100)


@pytest.mark.parametrize(
    'config',
    [
        water_tank_model(),
        water_tank_model(initial_guess=3, initial_condition=10),
        propeller_model(),
        propeller_model(maximum_torque=0.2),
        pendulum()
    ]
)
def test_steady_state(config):
    """Test the find_steady_state function"""

    # Adjust solver options
    config.solver_options['gtol'] = 1E-10

    sol = find_steady_state(config)
    assert sol.success

    # Structural checks
    assert sol.state.size == config.system.num_states
    assert sol.inputs.size == config.system.num_inputs

    # Check state bounds
    assert (-config.solver_options['gtol'] <=
            (sol.state - config.state_bounds[:, 0])).all()
    assert (-config.solver_options['gtol'] <=
            (config.state_bounds[:, 1] - sol.state)).all()

    # Check input bounds
    assert (-config.solver_options['gtol'] <=
            (sol.inputs - config.input_bounds[:, 0])).all()
    assert (-config.solver_options['gtol'] <=
            (config.input_bounds[:, 1] - sol.inputs)).all()

    # Check port constraints
    for signal_constraint in config.ports.values():
        value = signal_constraint.port(sol.system_state)
        assert (-config.solver_options['gtol'] <=
                (value - signal_constraint.lb)).all()
        assert (-config.solver_options['gtol'] <=
                (signal_constraint.ub - value)).all()

    # Check for steady states
    num_steady_states = np.count_nonzero(config.steady_states)
    diff_tol = np.sqrt(config.solver_options['gtol']*num_steady_states)
    steady_state_derivatives = \
        config.system.state_derivative(sol.system_state)[config.steady_states]
    npt.assert_allclose(steady_state_derivatives.ravel(),
                        0,
                        atol=diff_tol)


def test_invalid_objective():
    """Test the steady-state finding algorithm with an invalid objective"""

    system = System()

    config = SteadyStateConfiguration(system)
    config.objective = True

    with pytest.raises(ValueError):
        find_steady_state(config)


def test_without_goal():
    """Test the steady-state finding algorithm without objective function
    and with no steady states"""

    system = System()

    config = SteadyStateConfiguration(system)

    with pytest.raises(ValueError):
        find_steady_state(config)
