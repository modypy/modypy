"""
Some rigid-body simulation using moments and forces showing an object
moving in a circle with constant velocity and turn-rate.
"""
import math

import numpy as np
import matplotlib.pyplot as plt

from modypy.model import System
from modypy.blocks.sources import constant
from modypy.blocks.rigid import RigidBody6DOFFlatEarth, DirectCosineToEuler
from modypy.simulation import Simulator

# Set up the basic parameters of the system
# We want the body to rotate at 3 deg/s (leading to a full circle in 2min)
# and to have a forward velocity such that the body moves through a circle of
# radius 10.0.
MASS = 1.5
OMEGA = 2 * math.pi / 120
RADIUS = 10.0
VELOCITY_X = RADIUS * OMEGA
moment_of_inertia = np.diag([13.2E-3, 12.8E-3, 24.6E-3])

# Create a system
system = System()

# Add a block for 6-DoF (linear and angular motion) of a rigid body
rb_6dof = RigidBody6DOFFlatEarth(system,
                                 mass=MASS,
                                 initial_velocity_earth=[VELOCITY_X, 0, 0],
                                 initial_angular_rates_earth=[0, 0, OMEGA],
                                 moment_of_inertia=moment_of_inertia)

# Use the DirectCosineToEuler block to convert the Direct Cosine Matrix
# provided by the rigid-body block into Euler angles (pitch, roll and yaw)
dcm_to_euler = DirectCosineToEuler(system)

# We provide some thrust as centripetal force
thrust = constant(system, value=np.r_[0, MASS * OMEGA * VELOCITY_X, 0])
# No torques
torque = constant(system, value=np.r_[0, 0, 0])

# Connect thrust and torque to the rigid-body block
thrust.connect(rb_6dof.forces_body)
torque.connect(rb_6dof.moments_body)

# Connect the Direct Cosine Matrix output to the Euler conversion block
rb_6dof.dcm.connect(dcm_to_euler.dcm)

# Simulate the system for 2 minutes
sim = Simulator(system, start_time=0)
message = sim.run_until(time_boundary=120.0)

# Assume that simulation was successful
assert message is None

# Plot the Euler angles
fig_euler, (ax_yaw, ax_pitch, ax_roll) = plt.subplots(nrows=3, sharex="row")
ax_yaw.plot(sim.result.time,
            np.rad2deg(
                dcm_to_euler.yaw(sim.result)))
ax_pitch.plot(sim.result.time,
              np.rad2deg(
                  dcm_to_euler.pitch(sim.result)))
ax_roll.plot(sim.result.time,
             np.rad2deg(
                 dcm_to_euler.roll(sim.result)))
ax_yaw.set_title("Yaw")
ax_pitch.set_title("Pitch")
ax_roll.set_title("Roll")

# Plot the trajectory in top view
fig_top_view, ax = plt.subplots()
position = rb_6dof.position_earth(sim.result)
ax.plot(position[0], position[1])
plt.show()
