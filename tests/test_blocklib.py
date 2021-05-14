import math

import pytest
import numpy as np
import numpy.testing as npt

from modypy.blocks.aerodyn import Propeller, Thruster
from modypy.blocks.elmech import DCMotor
from modypy.blocks.linear import sum_signal
from modypy.blocks.sources import constant
from modypy.blocks.rigid import RigidBody6DOFFlatEarth, DirectCosineToEuler
from modypy.model import System, InputSignal
from modypy.simulation import Simulator
from modypy.steady_state import SteadyStateConfiguration, find_steady_state


@pytest.mark.parametrize(
    "thrust_coefficient, power_coefficient",
    [
        [0.09, 0.04],
        [(lambda n: 0.09), (lambda n: 0.04)],
    ]
)
def test_aerodyn_blocks(thrust_coefficient, power_coefficient):
    system = System()
    propeller = Propeller(system,
                          thrust_coefficient=thrust_coefficient,
                          power_coefficient=power_coefficient,
                          diameter=8 * 25.4E-3)
    dcmotor = DCMotor(system,
                      motor_constant=789.E-6,
                      resistance=43.3E-3,
                      inductance=1.9E-3,
                      moment_of_inertia=5.284E-6,
                      initial_omega=1)
    thruster = Thruster(system,
                        vector=np.c_[0, 0, -1],
                        arm=np.c_[1, 1, 0],
                        direction=1)
    density = constant(value=1.29)
    gravity = constant(value=[0, 0, 1.5/4*9.81])
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
    # Enforce the sum of forces to be zero
    steady_state_config.add_port_constraint(force_sum,
                                            lower_limit=[-np.inf, -np.inf, 0],
                                            upper_limit=[np.inf, np.inf, 0])
    # Ensure that the input voltage is non-negative
    sol = find_steady_state(steady_state_config)
    assert sol.success

    npt.assert_almost_equal(sol.state, [856.57715753, 67.01693871])
    npt.assert_almost_equal(sol.inputs, [3.5776728])

    npt.assert_almost_equal(
        propeller.power(sol.system_state),
        45.2926865)
    npt.assert_almost_equal(
        np.ravel(thruster.torque_vector(sol.system_state)),
        [-3.67875, 3.67875, -0.0528764])


def test_rigidbody_movement():
    mass = 1.5
    omega = 2 * math.pi / 120
    r = 10.0
    vx = r * omega
    moment_of_inertia = np.diag([13.2E-3, 12.8E-3, 24.6E-3])

    system = System()
    rb_6dof = RigidBody6DOFFlatEarth(owner=system,
                                     mass=mass,
                                     initial_velocity_earth=[vx, 0, 0],
                                     initial_angular_rates_earth=[0, 0, omega],
                                     initial_position_earth=[0, 0, 0],
                                     initial_transformation=np.eye(3, 3),
                                     moment_of_inertia=moment_of_inertia)
    dcm_to_euler = DirectCosineToEuler(system)
    thrust = constant(value=np.r_[0, mass * omega * vx, 0])
    moment = constant(value=np.r_[0, 0, 0])

    thrust.connect(rb_6dof.forces_body)
    moment.connect(rb_6dof.moments_body)

    rb_6dof.dcm.connect(dcm_to_euler.dcm)

    sim = Simulator(system, start_time=0)
    *_, last_item = sim.run_until(time_boundary=30.0)

    npt.assert_almost_equal(
        dcm_to_euler.yaw(last_item),
        math.pi/2)
    npt.assert_almost_equal(
        dcm_to_euler.pitch(last_item),
        0)
    npt.assert_almost_equal(
        dcm_to_euler.roll(last_item),
        0)
    npt.assert_almost_equal(
        rb_6dof.position_earth(last_item),
        [r, r, 0])
    npt.assert_almost_equal(
        rb_6dof.velocity_body(last_item),
        [vx, 0, 0]
    )
    npt.assert_almost_equal(
        rb_6dof.omega_body(last_item),
        [0, 0, omega]
    )


def test_rigidbody_defaults():
    mass = 1.5
    moment_of_inertia = np.diag([13.2E-3, 12.8E-3, 24.6E-3])

    system = System()
    rb_6dof = RigidBody6DOFFlatEarth(owner=system,
                                     mass=mass,
                                     moment_of_inertia=moment_of_inertia)
    dcm_to_euler = DirectCosineToEuler(system)
    thrust = constant(value=np.r_[0, 0, 0])
    moment = constant(value=moment_of_inertia @ np.r_[0, 0, math.pi/30**2])

    thrust.connect(rb_6dof.forces_body)
    moment.connect(rb_6dof.moments_body)

    rb_6dof.dcm.connect(dcm_to_euler.dcm)

    sim = Simulator(system, start_time=0)
    *_, last_item = sim.run_until(time_boundary=30.0)

    npt.assert_almost_equal(
        dcm_to_euler.yaw(last_item),
        math.pi/2)
    npt.assert_almost_equal(
        dcm_to_euler.pitch(last_item),
        0)
    npt.assert_almost_equal(
        dcm_to_euler.roll(last_item),
        0)
    npt.assert_almost_equal(
        rb_6dof.position_earth(last_item),
        [0, 0, 0])
