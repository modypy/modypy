import math

import pytest
import numpy as np
import numpy.testing as npt

from modypy.blocks.aerodyn import Propeller, Thruster
from modypy.blocks.elmech import DCMotor
from modypy.blocks.linear import Sum
from modypy.blocks.sources import constant
from modypy.blocks.rigid import RigidBody6DOFFlatEarth, DirectCosineToEuler
from modypy.linearization import find_steady_state
from modypy.model import System, InputSignal, Evaluator, OutputPort
from modypy.simulation import Simulator


@pytest.mark.parametrize(
    "channel_weights, output_size, inputs, expected_output",
    [
        ([1, 1], 1, [1, -1], [0]),
        ([1, 2], 2, [[1, 2], [3, 4]], [7, 10]),
        ([1, 2, 3], 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [30, 36, 42]),
    ]
)
def test_sum_block(channel_weights, output_size, inputs, expected_output):
    system = System()
    sum_block = Sum(system,
                    channel_weights=channel_weights,
                    output_size=output_size)
    for idx in range(len(inputs)):
        input_signal = InputSignal(system, shape=output_size, value=inputs[idx])
        sum_block.inputs[idx].connect(input_signal)

    evaluator = Evaluator(time=0, system=system)
    actual_output = evaluator.signals[sum_block.output.signal_slice]
    npt.assert_almost_equal(actual_output, expected_output)


@pytest.mark.parametrize(
    "thrust_coeff, power_coeff",
    [
        [0.09, 0.04],
        [(lambda n: 0.09), (lambda n: 0.04)],
    ]
)
def test_aerodyn_blocks(thrust_coeff, power_coeff):
    system = System()
    propeller = Propeller(system,
                          thrust_coeff=thrust_coeff,
                          power_coeff=power_coeff,
                          diameter=8 * 25.4E-3)
    dcmotor = DCMotor(system,
                      Kv=789.E-6,
                      R=43.3E-3,
                      L=1.9E-3,
                      J=5.284E-6,
                      initial_omega=1)
    thruster = Thruster(system,
                        vector=np.c_[0, 0, -1],
                        arm=np.c_[1, 1, 0],
                        direction=1)
    density = constant(system, value=1.29)
    gravity = constant(system, value=[0, 0, 1.5/4*9.81])
    force_sum = Sum(system, channel_weights=[1, 1], output_size=3)
    voltage = InputSignal(system)

    voltage.connect(dcmotor.voltage)
    density.connect(propeller.density)
    dcmotor.speed_rps.connect(propeller.speed_rps)
    propeller.torque.connect(dcmotor.external_torque)
    propeller.thrust.connect(thruster.scalar_thrust)
    dcmotor.torque.connect(thruster.scalar_torque)

    thruster.thrust_vector.connect(force_sum.inputs[0])
    gravity.connect(force_sum.inputs[1])

    force_target = OutputPort(system, shape=3)
    torque_target = OutputPort(system, shape=3)

    force_sum.output.connect(force_target)
    thruster.torque_vector.connect(torque_target)

    # Determine the steady state
    sol, states, inputs = find_steady_state(system=system,
                                            time=0,
                                            solver_options={
                                                'maxiter': 2000
                                            })
    assert sol.success

    npt.assert_almost_equal(states, [494.5280238,  22.3374414])
    npt.assert_almost_equal(inputs, [1.3573938])

    evaluator = Evaluator(system=system, time=0, state=states, inputs=inputs)
    npt.assert_almost_equal(evaluator.get_port_value(propeller.power), 8.7156812)


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
                                     initial_position_earth=[0,0,0],
                                     initial_transformation=np.eye(3, 3),
                                     moment_of_inertia=moment_of_inertia)
    dcm_to_euler = DirectCosineToEuler(system)
    thrust = constant(system, value=np.r_[0, mass * omega * vx, 0])
    moment = constant(system, value=np.r_[0, 0, 0])

    thrust.connect(rb_6dof.forces_body)
    moment.connect(rb_6dof.moments_body)

    rb_6dof.dcm.connect(dcm_to_euler.dcm)

    yaw_output = OutputPort(system)
    pitch_output = OutputPort(system)
    roll_output = OutputPort(system)

    pos_output = OutputPort(system, shape=3)

    dcm_to_euler.yaw.connect(yaw_output)
    dcm_to_euler.pitch.connect(pitch_output)
    dcm_to_euler.roll.connect(roll_output)
    rb_6dof.position_earth.connect(pos_output)

    sim = Simulator(system, start_time=0)
    message = sim.run_until(time_boundary=30.0)

    assert message is None

    npt.assert_almost_equal(sim.result.outputs[-1][yaw_output.output_slice],
                            math.pi/2)
    npt.assert_almost_equal(sim.result.outputs[-1][pitch_output.output_slice],
                            0)
    npt.assert_almost_equal(sim.result.outputs[-1][roll_output.output_slice],
                            0)
    npt.assert_almost_equal(sim.result.outputs[-1][pos_output.output_slice],
                            [r, r, 0])


def test_rigidbody_defaults():
    mass = 1.5
    omega = 2 * math.pi / 120
    r = 10.0
    moment_of_inertia = np.diag([13.2E-3, 12.8E-3, 24.6E-3])

    system = System()
    rb_6dof = RigidBody6DOFFlatEarth(owner=system,
                                     mass=mass,
                                     moment_of_inertia=moment_of_inertia)
    dcm_to_euler = DirectCosineToEuler(system)
    thrust = constant(system, value=np.r_[0, 0, 0])
    moment = constant(system, value=moment_of_inertia @ np.r_[0, 0, math.pi/30**2])

    thrust.connect(rb_6dof.forces_body)
    moment.connect(rb_6dof.moments_body)

    rb_6dof.dcm.connect(dcm_to_euler.dcm)

    yaw_output = OutputPort(system)
    pitch_output = OutputPort(system)
    roll_output = OutputPort(system)

    pos_output = OutputPort(system, shape=3)

    dcm_to_euler.yaw.connect(yaw_output)
    dcm_to_euler.pitch.connect(pitch_output)
    dcm_to_euler.roll.connect(roll_output)
    rb_6dof.position_earth.connect(pos_output)

    sim = Simulator(system, start_time=0)
    message = sim.run_until(time_boundary=30.0)

    assert message is None

    npt.assert_almost_equal(sim.result.outputs[-1][yaw_output.output_slice],
                            math.pi/2)
    npt.assert_almost_equal(sim.result.outputs[-1][pitch_output.output_slice],
                            0)
    npt.assert_almost_equal(sim.result.outputs[-1][roll_output.output_slice],
                            0)
    npt.assert_almost_equal(sim.result.outputs[-1][pos_output.output_slice],
                            [0, 0, 0])
