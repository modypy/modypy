import math

import numpy as np
import matplotlib.pyplot as plt

from simtree.compiler import Compiler
from simtree.blocks.sources import Constant
from simtree.blocks.rigid import RigidBody6DOFFlatEarth, DirectCosineToEuler
from simtree.blocks import NonLeafBlock
from simtree.simulator import Simulator

mass = 1.5
omega = 2*math.pi/120
r = 10.0
vx = r*omega
gravity = np.c_[0, 0, 9.81]
moment_of_inertia = np.diag([13.2E-3, 12.8E-3, 24.6E-3])

rb_6dof = RigidBody6DOFFlatEarth(mass=mass,
                                 gravity=gravity,
                                 initial_velocity_earth=[vx, 0, 0],
                                 initial_angular_rates_earth=[0, 0, omega],
                                 moment_of_inertia=moment_of_inertia)
dcm_to_euler = DirectCosineToEuler()
thrust = Constant(value=np.r_[0, mass*omega*vx, -1.5*9.81])
moment = Constant(value=np.r_[0, 0, 0])

sys = NonLeafBlock(children=[thrust, moment, rb_6dof, dcm_to_euler],
                   num_outputs=6)
sys.connect(thrust, range(3), rb_6dof, range(0,3))
sys.connect(moment, range(3), rb_6dof, range(3,6))
sys.connect(rb_6dof, range(6,15), dcm_to_euler, range(9))
sys.connect_output(dcm_to_euler, range(3), range(3))
sys.connect_output(rb_6dof, range(3,6), range(3,6))

compiler = Compiler(sys)
sys_compiled = compiler.compile()

sim = Simulator(sys_compiled, 0, 120.0)
message = sim.run()

assert message is None

fig, (ax_yaw, ax_pitch, ax_roll) = plt.subplots(nrows=3, sharex=True)
ax_yaw.plot(sim.result.time, np.rad2deg(sim.result.output[:, 0]))
ax_pitch.plot(sim.result.time, np.rad2deg(sim.result.output[:, 1]))
ax_roll.plot(sim.result.time, np.rad2deg(sim.result.output[:, 2]))
ax_yaw.set_title("Yaw")
ax_pitch.set_title("Pitch")
ax_roll.set_title("Roll")

fig, ax = plt.subplots()
ax.plot(sim.result.output[:, 3], sim.result.output[:, 4])
plt.show()
