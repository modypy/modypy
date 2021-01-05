import itertools

import numpy as np
import scipy.linalg as la

from simtree.model import Block, Port, System, InputSignal, OutputPort
from simtree.model.evaluator import Evaluator
from simtree.blocks.aerodyn import Propeller, Thruster
from simtree.blocks.elmech import DCMotor
from simtree.blocks.sources import constant
from simtree.blocks.linear import Sum
from simtree.linearization import find_steady_state, system_jacobian, InputSignal
from simtree.utils.uiuc_db import load_static_propeller


class Engine(Block):
    def __init__(self,
                 parent,
                 ct, cp, diameter,
                 Kv, R, L, J,
                 direction, vector, arm):
        Block.__init__(self, parent)

        self.voltage = Port(self, shape=1)
        self.density = Port(self, shape=1)

        self.dcmotor = DCMotor(self, Kv, R, L, J, initial_omega=1)
        self.propeller = Propeller(self,
                                   thrust_coeff=ct,
                                   power_coeff=cp,
                                   diameter=diameter)
        self.thruster = Thruster(self,
                                 direction=direction,
                                 vector=vector,
                                 arm=arm)

        self.thrust_vector = Port(self, shape=3)
        self.torque_vector = Port(self, shape=3)

        self.dcmotor.voltage.connect(self.voltage)
        self.propeller.density.connect(self.density)

        self.dcmotor.external_torque.connect(self.propeller.torque)
        self.propeller.speed_rps.connect(self.dcmotor.speed_rps)
        self.thruster.scalar_thrust.connect(self.propeller.thrust)
        self.thruster.scalar_torque.connect(self.dcmotor.torque)

        self.thrust_vector.connect(self.thruster.thrust_vector)
        self.torque_vector.connect(self.thruster.torque_vector)


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
    'diameter': 8*25.4E-3
}

RADIUS_X = 0.25
RADIUS_Y = 0.25

system = System()

engines = [
    Engine(system,
           vector=np.c_[0, 0, -1], arm=np.c_[+RADIUS_X, +RADIUS_Y, 0], direction=1,
           **parameters),
    Engine(system,
           vector=np.c_[0, 0, -1], arm=np.c_[-RADIUS_X, +RADIUS_Y, 0], direction=-1,
           **parameters),
    Engine(system,
           vector=np.c_[0, 0, -1], arm=np.c_[-RADIUS_X, -RADIUS_Y, 0], direction=1,
           **parameters),
    Engine(system,
           vector=np.c_[0, 0, -1], arm=np.c_[+RADIUS_X, -RADIUS_Y, 0], direction=-1,
           **parameters),
]

voltages = [
    InputSignal(system, value=0),
    InputSignal(system, value=0),
    InputSignal(system, value=0),
    InputSignal(system, value=0),
]

gravity_source = constant(system, value=np.r_[0, 0, 1.5*9.81])
counter_torque = constant(system, value=np.r_[0, 0, 0])
density = constant(system, value=1.29)
force_sum = Sum(system, channel_weights=np.ones(5), output_size=3)
torque_sum = Sum(system, channel_weights=np.ones(5), output_size=3)
current_sum = Sum(system, channel_weights=np.ones(4))

for idx, engine in zip(itertools.count(), engines):
    voltages[idx].connect(engine.voltage)
    engine.density.connect(density)
    engine.thrust_vector.connect(force_sum.inputs[idx])
    engine.torque_vector.connect(torque_sum.inputs[idx])
    engine.dcmotor.current.connect(current_sum.inputs[idx])

gravity_source.connect(force_sum.inputs[-1])
counter_torque.connect(torque_sum.inputs[-1])

force_output = OutputPort(system, shape=3)
torque_output = OutputPort(system, shape=3)

force_output.connect(force_sum.output)
torque_output.connect(torque_sum.output)

# Find the steady state of the system
sol, x0, u0 = find_steady_state(system,
                                time=0,
                                solver_options={
                                    'maxiter': 500 * (12 + 1),
                                    'xtol': 1E-15
                                })

if sol.success:
    evaluator = Evaluator(system=system, time=0, state=x0, inputs=u0)
    print("\tx0     =%s" % x0)
    print("\tu0     =%s" % x0)
    print("\tdxdt   =%s" % evaluator.state_derivative)
    print("\tforce  =%s" % evaluator.get_port_value(force_sum.output))
    print("\ttorque =%s" % evaluator.get_port_value(torque_sum.output))
    A, B, C, D = system_jacobian(system, 0, x0, u0)
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
    print("Eigenvectors:%s" % vr)
else:
    print("message=%s" % sol.message)
