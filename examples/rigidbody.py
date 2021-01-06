import math

import numpy as np
import matplotlib.pyplot as plt

from modypy.model import System
from modypy.blocks.sources import constant
from modypy.blocks.rigid import RigidBody6DOFFlatEarth, DirectCosineToEuler
from modypy.simulation import Simulator

MASS = 1.5
OMEGA = 2 * math.pi / 120
RADIUS = 10.0
VELOCITY_X = RADIUS * OMEGA
moment_of_inertia = np.diag([13.2E-3, 12.8E-3, 24.6E-3])

system = System()

rb_6dof = RigidBody6DOFFlatEarth(system,
                                 mass=MASS,
                                 initial_velocity_earth=[VELOCITY_X, 0, 0],
                                 initial_angular_rates_earth=[0, 0, OMEGA],
                                 moment_of_inertia=moment_of_inertia)
dcm_to_euler = DirectCosineToEuler(system)
thrust = constant(system, value=np.r_[0, MASS * OMEGA * VELOCITY_X, 0])
moment = constant(system, value=np.r_[0, 0, 0])

thrust.connect(rb_6dof.forces_body)
moment.connect(rb_6dof.moments_body)
rb_6dof.dcm.connect(dcm_to_euler.dcm)

sim = Simulator(system, start_time=0)
message = sim.run_until(t_bound=120.0)

assert message is None

fig_euler, (ax_yaw, ax_pitch, ax_roll) = plt.subplots(nrows=3, sharex='row')
ax_yaw.plot(sim.result.time, np.rad2deg(sim.result.signals[:, dcm_to_euler.yaw.signal_slice]))
ax_pitch.plot(sim.result.time, np.rad2deg(sim.result.signals[:, dcm_to_euler.pitch.signal_slice]))
ax_roll.plot(sim.result.time, np.rad2deg(sim.result.signals[:, dcm_to_euler.roll.signal_slice]))
ax_yaw.set_title("Yaw")
ax_pitch.set_title("Pitch")
ax_roll.set_title("Roll")

fig_top_view, ax = plt.subplots()
position = sim.result.signals[:, rb_6dof.position_earth.signal_slice]
ax.plot(position[:, 0], position[:, 1])
plt.show()
