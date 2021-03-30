"""
Test the steady-state determination algorithm.
"""
import numpy as np
import numpy.testing as npt
import pytest

from modypy.blocks.aerodyn import Propeller
from modypy.blocks.linear import sum_signal
from modypy.blocks.sources import constant
from modypy.model import System, InputSignal, SignalState, State, PortNotConnectedError, Port, Signal
from modypy.steady_state import SteadyStateConfiguration, find_steady_state, DuplicateSignalConstraintError


def water_tank_model(inflow_area=0.01,
                     outflow_area=0.02,
                     tank_area=0.2,
                     target_height=5):
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
    config.input_bounds[inflow_velocity.input_slice, 0] = 0
    # Define the target height
    config.state_bounds[height_state.state_slice] = target_height

    return config


def propeller_model(port_constraints=True,
                    thrust_coefficient=0.09,
                    power_coefficient=0.04,
                    density=1.29,
                    diameter=8 * 25.4E-3,
                    moment_of_inertia=5.284E-6,
                    target_thrust=1.5 * 9.81 / 4):
    """Create a steady-state configuration for two propellers"""

    system = System()

    torque_1 = InputSignal(system)
    torque_2 = InputSignal(system)

    density_signal = constant(system, density)

    def speed_1_dt(data):
        """Derivative of the speed of the first propeller"""
        return torque_1(data) / (2 * np.pi * moment_of_inertia)

    def speed_2_dt(data):
        """Derivative of the speed of the second propeller"""
        return torque_2(data) / (2 * np.pi * moment_of_inertia)

    speed_1 = SignalState(system,
                          derivative_function=speed_1_dt)
    speed_2 = SignalState(system,
                          derivative_function=speed_2_dt)

    propeller_1 = Propeller(system,
                            thrust_coefficient=thrust_coefficient,
                            power_coefficient=power_coefficient,
                            diameter=diameter)
    propeller_2 = Propeller(system,
                            thrust_coefficient=thrust_coefficient,
                            power_coefficient=power_coefficient,
                            diameter=diameter)

    propeller_1.density.connect(density_signal)
    propeller_2.density.connect(density_signal)
    propeller_1.speed_rps.connect(speed_1)
    propeller_2.speed_rps.connect(speed_2)

    total_thrust = sum_signal(system, (propeller_1.thrust,
                                       propeller_2.thrust))
    total_power = sum_signal(system, (propeller_1.power,
                                      propeller_2.power))

    # Set up steady-state configuration
    config = SteadyStateConfiguration(system)
    # Minimize the total power
    config.objective = total_power
    # Constrain the total thrust
    if port_constraints:
        config.add_port_constraint(total_thrust, target_thrust)
    else:
        config.signal_bounds[total_thrust.signal_slice] = target_thrust

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
    # No steady states
    config.steady_states = [False, ] * system.num_states
    # Bounds for the angle
    config.state_bounds[phi.state_slice, 0] = -np.pi
    config.state_bounds[phi.state_slice, 1] = np.pi

    return config


@pytest.mark.parametrize(
    "config",
    [
        water_tank_model(),
        propeller_model(port_constraints=True),
        propeller_model(port_constraints=False),
        pendulum()
    ]
)
def test_steady_state(config):
    """Test the find_steady_state function"""

    # Adjust solver options
    config.solver_options["gtol"] = 1E-17

    sol = find_steady_state(config)
    assert sol.success is True

    # Structural checks
    assert sol.state.size == config.system.num_states
    assert sol.inputs.size == config.system.num_inputs

    # Check state bounds
    assert (config.state_bounds[:, 0] <= sol.state).all()
    assert (sol.state <= config.state_bounds[:, 1]).all()

    # Check input bounds
    assert (config.input_bounds[:, 0] <= sol.inputs).all()
    assert (sol.inputs <= config.input_bounds[:, 1]).all()

    for signal in config.system.signals:
        idxs = signal.signal_slice
        value = signal(sol.evaluator).flatten()
        # Check lower bounds
        assert (np.isnan(config.signal_bounds[idxs, 0]) |
                (config.signal_bounds[idxs, 0] <= value)).all()
        # Check upper bounds
        assert (np.isnan(config.signal_bounds[idxs, 1]) |
                (value <= config.signal_bounds[idxs, 1])).all()

    for signal_constraint in config.signal_constraints:
        value = signal_constraint.signal(sol.evaluator)
        assert signal_constraint.lb <= value
        assert value <= signal_constraint.ub

    # Check for steady states
    steady_state_count = np.count_nonzero(config.steady_states)
    steady_state_derivatives = \
        config.system.state_derivative(sol.evaluator)[config.steady_states]
    npt.assert_allclose(steady_state_derivatives.flatten(),
                        np.zeros(steady_state_count),
                        atol=1E-8)


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


def test_unconnected_port():
    """Test whether adding a port constraint for an unconnected port
    leads to an exception"""

    system = System()
    port = Port(system)

    config = SteadyStateConfiguration(system)

    with pytest.raises(PortNotConnectedError):
        config.add_port_constraint(port)


def test_duplicate_signal_constraint():
    """Test whether adding multiple constraints for the same signal leads
    to an exception."""

    system = System()
    port_a = Port(system)
    port_b = Port(system)
    signal = Signal(system)
    port_a.connect(signal)
    port_b.connect(signal)

    config = SteadyStateConfiguration(system)
    config.add_port_constraint(port_a)
    with pytest.raises(DuplicateSignalConstraintError):
        config.add_port_constraint(port_b)
