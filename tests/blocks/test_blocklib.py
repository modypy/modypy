# pylint: disable=missing-module-docstring
import math

import numpy as np
import pytest
from modypy.blocks.aerodyn import Propeller, Thruster
from modypy.blocks.elmech import DCMotor
from modypy.blocks.linear import sum_signal
from modypy.blocks.rigid import DirectCosineToEuler, RigidBody6DOFFlatEarth
from modypy.blocks.sources import constant, time, FunctionSignal
from modypy.model import InputSignal, System, Clock
from modypy.simulation import Simulator, SimulationResult
from modypy.steady_state import SteadyStateConfiguration, find_steady_state
from numpy import testing as npt


def test_time_signal():
    system = System()
    Clock(owner=system, period=1 / 10.0)
    simulator = Simulator(system=system, start_time=0.0)
    result = SimulationResult(
        system=system, source=simulator.run_until(time_boundary=10.0)
    )
    npt.assert_equal(result.time, time(result))


def atan2_function_signal():
    sin_signal = FunctionSignal(np.sin, time)
    cos_signal = FunctionSignal(np.cos, time)
    atan2_signal = FunctionSignal(np.arctan2, signals=(sin_signal, cos_signal))
    return atan2_signal


@pytest.mark.parametrize(
    "func_signal, ref_signal",
    [
        (FunctionSignal(np.sin, time), lambda s: np.sin(s.time)),
        (
            FunctionSignal(
                np.arctan2,
                signals=(
                    FunctionSignal(np.sin, time),
                    FunctionSignal(np.cos, time),
                ),
            ),
            time,
        ),
    ],
)
def test_function_signal(func_signal, ref_signal):
    system = System()
    Clock(owner=system, period=1 / 10.0)
    simulator = Simulator(system=system, start_time=0.0)
    result = SimulationResult(
        system=system, source=simulator.run_until(time_boundary=1.0)
    )
    npt.assert_almost_equal(func_signal(result), ref_signal(result))


@pytest.mark.parametrize(
    "thrust_coefficient, power_coefficient",
    [
        [0.09, 0.04],
        [(lambda n: 0.09), (lambda n: 0.04)],
    ],
)
def test_aerodyn_blocks(thrust_coefficient, power_coefficient):
    system = System()
    propeller = Propeller(
        system,
        thrust_coefficient=thrust_coefficient,
        power_coefficient=power_coefficient,
        diameter=8 * 25.4e-3,
    )
    dcmotor = DCMotor(
        system,
        motor_constant=789.0e-6,
        resistance=43.3e-3,
        inductance=1.9e-3,
        moment_of_inertia=5.284e-6,
        initial_omega=1,
    )
    thruster = Thruster(
        system, vector=np.c_[0, 0, -1], arm=np.c_[1, 1, 0], direction=1
    )
    density = constant(value=1.29)
    gravity = constant(value=[0, 0, 1.5 / 4 * 9.81])
    voltage = InputSignal(system)

    voltage.connect(dcmotor.voltage)
    density.connect(propeller.density)
    dcmotor.speed_rps.connect(propeller.speed_rps)
    propeller.torque.connect(dcmotor.external_torque)
    propeller.thrust.connect(thruster.scalar_thrust)
    dcmotor.torque.connect(thruster.scalar_torque)

    force_sum = sum_signal((thruster.thrust_vector, gravity))

    # Determine the steady state
    steady_state_config = SteadyStateConfiguration(system)
    # Enforce the sum of forces to be zero along the z-axis
    steady_state_config.ports[force_sum].lower_bounds[2] = 0
    steady_state_config.ports[force_sum].upper_bounds[2] = 0
    # Ensure that the input voltage is non-negative
    steady_state_config.inputs[voltage].lower_bounds = 0
    sol = find_steady_state(steady_state_config)
    assert sol.success

    npt.assert_almost_equal(sol.state, [856.5771575, 67.0169392])
    npt.assert_almost_equal(sol.inputs, [3.5776728])

    npt.assert_almost_equal(propeller.power(sol.system_state), 45.2926865)
    npt.assert_almost_equal(
        np.ravel(thruster.torque_vector(sol.system_state)),
        [-3.67875, 3.67875, -0.0528764],
    )


def test_rigidbody_movement():
    mass = 1.5
    omega = 2 * math.pi / 120
    r = 10.0
    vx = r * omega
    moment_of_inertia = np.diag([13.2e-3, 12.8e-3, 24.6e-3])

    system = System()
    rb_6dof = RigidBody6DOFFlatEarth(
        owner=system,
        mass=mass,
        initial_velocity_earth=[vx, 0, 0],
        initial_angular_rates_earth=[0, 0, omega],
        initial_position_earth=[0, 0, 0],
        initial_transformation=np.eye(3, 3),
        moment_of_inertia=moment_of_inertia,
    )
    dcm_to_euler = DirectCosineToEuler(system)
    thrust = constant(value=np.r_[0, mass * omega * vx, 0])
    moment = constant(value=np.r_[0, 0, 0])

    thrust.connect(rb_6dof.forces_body)
    moment.connect(rb_6dof.moments_body)

    rb_6dof.dcm.connect(dcm_to_euler.dcm)

    rtol = 1e-12
    atol = 1e-12
    sim = Simulator(system, start_time=0, rtol=rtol, atol=atol)
    *_, last_item = sim.run_until(time_boundary=30.0, include_last=True)

    npt.assert_allclose(
        dcm_to_euler.yaw(last_item), math.pi / 2, rtol=rtol, atol=atol
    )
    npt.assert_allclose(dcm_to_euler.pitch(last_item), 0, rtol=rtol, atol=atol)
    npt.assert_allclose(dcm_to_euler.roll(last_item), 0, rtol=rtol, atol=atol)
    npt.assert_allclose(
        rb_6dof.position_earth(last_item), [r, r, 0], rtol=rtol, atol=atol
    )
    npt.assert_allclose(
        rb_6dof.velocity_body(last_item), [vx, 0, 0], rtol=rtol, atol=atol
    )
    npt.assert_allclose(
        rb_6dof.omega_body(last_item), [0, 0, omega], rtol=rtol, atol=atol
    )


def test_rigidbody_defaults():
    mass = 1.5
    moment_of_inertia = np.diag([13.2e-3, 12.8e-3, 24.6e-3])

    system = System()
    rb_6dof = RigidBody6DOFFlatEarth(
        owner=system, mass=mass, moment_of_inertia=moment_of_inertia
    )
    dcm_to_euler = DirectCosineToEuler(system)
    thrust = constant(value=np.r_[0, 0, 0])
    moment = constant(value=moment_of_inertia @ np.r_[0, 0, math.pi / 30 ** 2])

    thrust.connect(rb_6dof.forces_body)
    moment.connect(rb_6dof.moments_body)

    rb_6dof.dcm.connect(dcm_to_euler.dcm)

    rtol = 1e-12
    atol = 1e-12
    sim = Simulator(system, start_time=0, rtol=rtol, atol=atol)
    *_, last_item = sim.run_until(time_boundary=30.0, include_last=True)

    npt.assert_allclose(
        dcm_to_euler.yaw(last_item), math.pi / 2, rtol=rtol, atol=atol
    )
    npt.assert_allclose(dcm_to_euler.pitch(last_item), 0, rtol=rtol, atol=atol)
    npt.assert_allclose(dcm_to_euler.roll(last_item), 0, rtol=rtol, atol=atol)
    npt.assert_allclose(
        rb_6dof.position_earth(last_item), [0, 0, 0], rtol=rtol, atol=atol
    )
