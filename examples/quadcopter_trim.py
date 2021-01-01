import itertools

import numpy as np
import scipy.linalg as la

from simtree.blocks import NonLeafBlock
from simtree.blocks.aerodyn import Propeller, Thruster
from simtree.blocks.elmech import DCMotor
from simtree.blocks.sources import Constant
from simtree.blocks.linear import Sum
from simtree.compiler import Compiler
from simtree.linearization import find_steady_state, system_jacobian
from simtree.utils.uiuc_db import load_static_propeller


class Engine(NonLeafBlock):
    def __init__(self,
                 ct, cp, diameter,
                 Kv, R, L, J,
                 direction, vector, arm,
                 **kwargs):
        NonLeafBlock.__init__(self,
                              num_inputs=2,
                              num_outputs=6,
                              **kwargs)
        self.dcmotor = DCMotor(Kv, R, L, J, name="dcmotor")
        self.propeller = Propeller(thrust_coeff=ct,
                                   power_coeff=cp,
                                   diameter=diameter,
                                   name="propeller")
        self.thruster = Thruster(direction=direction,
                                 vector=vector,
                                 arm=arm)
        # Add the children
        self.add_block(self.dcmotor)
        self.add_block(self.propeller)
        self.add_block(self.thruster)
        # Connect the input voltage to the motor
        self.connect_input(0, self.dcmotor, 0)
        # Connect the torque of the propeller to the motor
        self.connect(self.propeller, 1, self.dcmotor, 1)
        # Connect the speed of the motor to the speed of the propeller
        self.connect(self.dcmotor, 0, self.propeller, 0)
        # Connect the input density to the density for the propeller
        self.connect_input(1, self.propeller, 1)
        # Connect the thrust of the propeller to the thruster
        self.connect(self.propeller, 0, self.thruster, 0)
        # Connect the torque of the DC motor to the thruster
        self.connect(self.dcmotor, 2, self.thruster, 1)
        # Connect the thrust and torque of the thruster to outputs 0:6
        self.connect_output(self.thruster, range(6), range(6))


thrust_coeff, torque_coeff = \
    load_static_propeller('volume-1/data/apcsf_8x3.8_static_2777rd.txt',
                          interp_options={"bounds_error": False,
                                          "fill_value": "extrapolate"})

parameters = {
    'Kv': 789.E-6,
    'R': 43.3E-3,
    'L': 1.9E-3,
    'J': 5.284E-6,
    'ct': thrust_coeff,
    'cp': torque_coeff,
    'D': 8*25.4E-3
}

RADIUS_X = 0.25
RADIUS_Y = 0.25

engines = [
    Engine(name="engine1",
           vector=np.c_[0, 0, -1], arm=np.c_[+RADIUS_X, +RADIUS_Y, 0], direction=1,
           **parameters),
    Engine(name="engine2",
           vector=np.c_[0, 0, -1], arm=np.c_[+RADIUS_X, -RADIUS_Y, 0], direction=-1,
           **parameters),
    Engine(name="engine3",
           vector=np.c_[0, 0, -1], arm=np.c_[-RADIUS_X, -RADIUS_Y, 0], direction=1,
           **parameters),
    Engine(name="engine4",
           vector=np.c_[0, 0, -1], arm=np.c_[-RADIUS_X, +RADIUS_Y, 0], direction=-1,
           **parameters),
]

gravity_source = Constant(value=np.c_[0, 0, 1.5*9.81], name="gravity")
density = Constant(value=1.29, name="rho")
thrust_sum = Sum(channel_weights=[1, 1, 1, 1, 1], channel_dim=3, name="thrust")
torque_sum = Sum(channel_weights=[1, 1, 1, 1], channel_dim=3, name="torque")

frame = NonLeafBlock(name="frame",
                     num_inputs=4,
                     num_outputs=6)
frame.add_block(thrust_sum)
frame.add_block(torque_sum)
frame.add_block(gravity_source)
frame.add_block(density)
for idx, engine in zip(itertools.count(), engines):
    # Add each of the engines and connect it
    frame.add_block(engine)
    # Connect the voltage input
    frame.connect_input(idx, engine, 0)
    # Connect the density
    frame.connect(density, 0, engine, 1)
    # Connect the thrust output to the thrust sum
    frame.connect(engine, range(3), thrust_sum, range(3*idx, 3*idx+3))
    # Connect the torque output to the thrust sum
    frame.connect(engine, range(3, 6), torque_sum, range(3*idx, 3*idx+3))
# Connect the gravity to the thrust sum
frame.connect(gravity_source, range(3), thrust_sum,
              range(3*len(engines), 3*len(engines)+3))
# Connect the thrust and torque sums to the outputs
frame.connect_output(thrust_sum, range(3), range(3))
frame.connect_output(torque_sum, range(3), range(3, 6))

# Compile the frame
compiler = Compiler(frame)
compiled_system = compiler.compile()

# solution: omega=855, i=66.77, v=3.565

omega0, i0, v0 = 1, 0, 0

x_start = 4 * [omega0, i0]
u_start = 4 * [v0]

# Find the steady state of the system
sol, x0, u0 = find_steady_state(compiled_system,
                                time=0,
                                x_start=x_start,
                                u_start=u_start,
                                solver_options={
                                    'maxiter': 500*(12+1)
                                })
dxdt0 = compiled_system.state_update_function(0, x0, u0)
y0 = compiled_system.output_function(0, x0, u0)
A, B, C, D = system_jacobian(compiled_system, 0, x0, u0)

print("success=%s nfev=%d" % (sol.success, sol.nfev))
if sol.success:
    print("\tx0  =%s" % x0)
    print("\tu0  =%s" % u0)
    print("\tdxdt=%s" % dxdt0)
    print("\ty0  =%s" % y0)
    print("A:")
    print(A)
    print("B:")
    print(B)
    print("C:")
    print(C)
    print("D:")
    print(D)

    w, vr = la.eig(A)
    print("Eigenvalues:%s" % w)
else:
    print("message=%s" % sol.message)
