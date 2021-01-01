import math

import pytest
import numpy as np
import numpy.testing as npt

from simtree.blocks import NonLeafBlock
from simtree.blocks.aerodyn import Propeller, Thruster
from simtree.blocks.elmech import DCMotor
from simtree.blocks.linear import Sum
from simtree.blocks.sources import Constant
from simtree.blocks.rigid import RigidBody6DOFFlatEarth, DirectCosineToEuler
from simtree.compiler import compile
from simtree.linearization import find_steady_state
from simtree.simulator import Simulator

@pytest.mark.parametrize(
    "block, inputs, expected_output",
    [
        (Sum(channel_weights=[1, 1], channel_dim=1),
         [1, -1],
         [0]),
        (Sum(channel_weights=[1, 2], channel_dim=2),
         [1, 2, 3, 4],
         [7, 10]),
        (Sum(channel_weights=[1, 2, 3], channel_dim=3),
         np.r_[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
         [30, 36, 42])
    ]
)
def test_sum_block(block, inputs, expected_output):
    actual_output = block.output_function(0, inputs)
    npt.assert_almost_equal(actual_output, expected_output)

@pytest.mark.parametrize(
    "propeller",
    [
        Propeller(thrust_coeff=0.09,
                  power_coeff=0.04,
                  diameter=8 * 25.4E-3),
        Propeller(thrust_coeff=(lambda n: 0.09),
                  power_coeff=(lambda n: 0.04),
                  diameter=8 * 25.4E-3),
    ]
)
def test_aerodyn_blocks(propeller):
    dcmotor = DCMotor(Kv=789.E-6,
                      R=43.3E-3,
                      L=1.9E-3,
                      J=5.284E-6)
    thruster = Thruster(vector=np.c_[0, 0, -1],
                        arm=np.c_[1, 1, 0],
                        direction=1)
    density = Constant(value=1.29, name="density")
    gravity = Constant(value=[0, 0, 1.5/4*9.81], name="gravity")
    thrust_sum = Sum(channel_weights=[1, 1], channel_dim=3)
    model = NonLeafBlock(children=[dcmotor,
                                   propeller,
                                   thruster,
                                   density,
                                   gravity,
                                   thrust_sum],
                         num_inputs=1,
                         num_outputs=3)
    # Connect the voltage
    model.connect_input(0, dcmotor, 0)
    # Connect the propeller braking torque
    model.connect(propeller, 1, dcmotor, 1)
    # Connect the shaft speed to the propeller
    model.connect(dcmotor, 0, propeller, 0)
    # Connect the density to the propeller
    model.connect(density, 0, propeller, 1)
    # Connect the propeller thrust to the thruster
    model.connect(propeller, 0, thruster, 0)
    # Connect the motor torque to the thruster
    model.connect(dcmotor, 1, thruster, 1)
    # Connect the thrust vector to the thrust sum
    model.connect(thruster, range(3), thrust_sum, range(3))
    # Connect the gravity to the thrust sum
    model.connect(gravity, range(3), thrust_sum, range(3,6))
    # Connect the thrust output to the block output
    model.connect_output(thrust_sum, range(3), range(3))

    # Compile the model
    compiled_model = compile(model)

    # Define the initial state and input
    x_initial = np.r_[1, 0]
    u_initial = np.r_[0]

    # Determine the steady state
    sol, x0, u0 = find_steady_state(system=compiled_model,
                                    time=0,
                                    x_start=x_initial,
                                    u_start=u_initial,
                                    solver_options={
                                        'maxiter': 2000
                                    })
    assert sol.success == True
    npt.assert_almost_equal(x0, [856.57715753, 67.01693923])
    npt.assert_almost_equal(u0, [3.57767285])


def test_rigidbody_movement():
    mass = 1.5
    omega = 2 * math.pi / 120
    r = 10.0
    vx = r * omega
    gravity = np.c_[0, 0, 9.81]
    moment_of_inertia = np.diag([13.2E-3, 12.8E-3, 24.6E-3])
    rb_6dof = RigidBody6DOFFlatEarth(mass=mass,
                                     gravity=gravity,
                                     initial_velocity_earth=[vx, 0, 0],
                                     initial_angular_rates_earth=[0, 0, omega],
                                     initial_position_earth=[0,0,0],
                                     initial_transformation=np.eye(3, 3),
                                     moment_of_inertia=moment_of_inertia)
    dcm_to_euler = DirectCosineToEuler()
    thrust = Constant(value=np.r_[0, mass * omega * vx, -1.5 * 9.81])
    moment = Constant(value=np.r_[0, 0, 0])

    sys = NonLeafBlock(children=[thrust, moment, rb_6dof, dcm_to_euler],
                       num_outputs=6)
    sys.connect(thrust, range(3), rb_6dof, range(0, 3))
    sys.connect(moment, range(3), rb_6dof, range(3, 6))
    sys.connect(rb_6dof, range(6, 15), dcm_to_euler, range(9))
    sys.connect_output(dcm_to_euler, range(3), range(3))
    sys.connect_output(rb_6dof, range(3, 6), range(3, 6))

    sys_compiled = compile(sys)

    sim = Simulator(sys_compiled, 0, 30.0)
    message = sim.run()

    assert message is None
    npt.assert_almost_equal(sim.result.output[-1],
                            [math.pi/2, 0, 0, r, r, 0])


def test_rigidbody_defaults():
    mass = 1.5
    moment_of_inertia = np.diag([13.2E-3, 12.8E-3, 24.6E-3])

    rb_6dof = RigidBody6DOFFlatEarth(mass=mass,
                                     moment_of_inertia=moment_of_inertia)
    dcm_to_euler = DirectCosineToEuler()
    thrust = Constant(value=np.r_[0, 0, -mass*9.81])
    moment = Constant(value=moment_of_inertia @ np.r_[0, 0, math.pi/30**2])

    sys = NonLeafBlock(children=[thrust, moment, rb_6dof, dcm_to_euler],
                       num_outputs=6)
    sys.connect(thrust, range(3), rb_6dof, range(0, 3))
    sys.connect(moment, range(3), rb_6dof, range(3, 6))
    sys.connect(rb_6dof, range(6, 15), dcm_to_euler, range(9))
    sys.connect_output(dcm_to_euler, range(3), range(3))
    sys.connect_output(rb_6dof, range(3, 6), range(3, 6))

    sys_compiled = compile(sys)

    sim = Simulator(sys_compiled, 0, 30.0)
    message = sim.run()

    assert message is None
    npt.assert_almost_equal(sim.result.output[-1],
                            [math.pi/2, 0, 0, 0, 0, 0])
