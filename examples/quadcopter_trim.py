"""
A larger example showcasing the steady-state-determination and linearisation
of complex systems, in this case for a quadrocopter frame with four
DC-motors with propellers.
"""
import itertools

import numpy as np

from modypy.model import Block, Port, System, InputSignal
from modypy.blocks.aerodyn import Propeller, Thruster
from modypy.blocks.elmech import DCMotor
from modypy.blocks.sources import constant
from modypy.blocks.linear import sum_signal
from modypy.steady_state import SteadyStateConfiguration, find_steady_state
from modypy.linearization import system_jacobian, LinearizationConfiguration, OutputDescriptor
from modypy.utils.uiuc_db import load_static_propeller


class Engine(Block):
    """
    An engine block consisting of a propeller and a DC-motor.
    """

    def __init__(self,
                 parent,
                 thrust_coefficient, power_coefficient, diameter,
                 motor_constant, resistance, inductance, moment_of_inertia,
                 direction, vector, arm):
        Block.__init__(self, parent)

        # Create ports for the motor voltage and the air density
        self.voltage = Port()
        self.density = Port()

        # Create the motor
        self.dcmotor = DCMotor(self,
                               motor_constant,
                               resistance,
                               inductance,
                               moment_of_inertia,
                               initial_omega=1)

        # Create the propeller
        self.propeller = Propeller(self,
                                   thrust_coefficient=thrust_coefficient,
                                   power_coefficient=power_coefficient,
                                   diameter=diameter)

        # Create a thruster that converts the scalar thrust and torque into
        # thrust and torque vectors, considering the arm of the engine
        # relative to the center of gravity
        self.thruster = Thruster(self,
                                 direction=direction,
                                 vector=vector,
                                 arm=arm)

        # Provide output ports for the thrust and torque vectors
        self.thrust_vector = Port(shape=3)
        self.torque_vector = Port(shape=3)

        # Connect the voltage to the motor
        self.dcmotor.voltage.connect(self.voltage)
        # Connect the density to the propeller
        self.propeller.density.connect(self.density)

        # Connect the motor and propeller to each other
        self.dcmotor.external_torque.connect(self.propeller.torque)
        self.propeller.speed_rps.connect(self.dcmotor.speed_rps)

        # Provide the scalar thrust and torque to the thruster
        self.thruster.scalar_thrust.connect(self.propeller.thrust)
        self.thruster.scalar_torque.connect(self.dcmotor.torque)

        # Connect the thrust and torque vectors to the outputs
        self.thrust_vector.connect(self.thruster.thrust_vector)
        self.torque_vector.connect(self.thruster.torque_vector)


# Import thrust and torque coefficients from the UIUC propeller database
thrust_coeff, torque_coeff = \
    load_static_propeller("volume-1/data/apcsf_8x3.8_static_2777rd.txt",
                          interp_options={"bounds_error": False,
                                          "fill_value": "extrapolate"})

# Set up a general parameters vector for all four engines
parameters = {
    "motor_constant": 789.E-6,
    "resistance": 43.3E-3,
    "inductance": 1.9E-3,
    "moment_of_inertia": 5.284E-6,
    "thrust_coefficient": thrust_coeff,
    "power_coefficient": torque_coeff,
    "diameter": 8 * 25.4E-3
}

# Set up the positions of the engines
POSITION_X = 0.25
POSITION_Y = 0.25

# Create a system
system = System()

# Add the engines to the system.
# Note how the positions and the directions alternate between the engines.
engines = [
    Engine(system,
           vector=np.c_[0, 0, -1],
           arm=np.c_[+POSITION_X, +POSITION_Y, 0],
           direction=1,
           **parameters),
    Engine(system,
           vector=np.c_[0, 0, -1],
           arm=np.c_[-POSITION_X, +POSITION_Y, 0],
           direction=-1,
           **parameters),
    Engine(system,
           vector=np.c_[0, 0, -1],
           arm=np.c_[-POSITION_X, -POSITION_Y, 0],
           direction=1,
           **parameters),
    Engine(system,
           vector=np.c_[0, 0, -1],
           arm=np.c_[+POSITION_X, -POSITION_Y, 0],
           direction=-1,
           **parameters),
]

# Create input and output signals.
# The steady-state determination will try to vary the input signals and states
# such that the state derivatives and the output signals are zero.

# Create individual input signals for the voltages of the four engines
voltages = [
    InputSignal(system, value=0),
    InputSignal(system, value=0),
    InputSignal(system, value=0),
    InputSignal(system, value=0),
]

# Provide the air density as input
density = constant(value=1.29)

# Connect the engines
for idx, engine in zip(itertools.count(), engines):
    # Provide voltage and density to the engine
    voltages[idx].connect(engine.voltage)
    density.connect(engine.density)

# We consider gravity and a possible counter torque that need to be compensated
# by thrust and torque
gravity_source = constant(value=np.r_[0, 0, 1.5 * 9.81])
counter_torque = constant(value=np.r_[0, 0, 0])

# Determine the sum of forces and torques
forces_sum = sum_signal([engine.thrust_vector for engine in engines] +
                        [gravity_source])
torques_sum = sum_signal([engine.torque_vector for engine in engines] +
                         [counter_torque])

# Determine the sum of all currents
total_current = sum_signal([engine.dcmotor.current for engine in engines])

# Configure the steady-state finder
steady_state_config = SteadyStateConfiguration(system)
# Minimize total current
steady_state_config.objective = total_current
# Force that the input voltages are non-negative
for voltage in voltages:
    steady_state_config.inputs[voltage].lower_bounds = 0
# Force that currents and speeds are non-negative
for engine in engines:
    steady_state_config.states[engine.dcmotor.current].lower_bounds = 0
    steady_state_config.states[engine.dcmotor.omega].lower_bounds = 0

# Enforce the total torque to be zero
steady_state_config.ports[torques_sum].lower_bounds = 0
steady_state_config.ports[torques_sum].upper_bounds = 0
# Enforce the total vertical force to be zero
# Note that we only force the vertical force to zero, as otherwise we'd have
# too many equality constraints.
# There are the following free variables
# - The four voltage inputs
# - The eight state variables
# There are the following equality constraints
# - The eight state derivatives (which are constrained to zero)
# - The three components of the torque vector
# - The vertical component of the thrust vector
steady_state_config.ports[forces_sum].lower_bounds[2] = 0
steady_state_config.ports[forces_sum].upper_bounds[2] = 0

# Find the steady state of the system
sol = find_steady_state(steady_state_config)

if sol.success:
    # The function was successful in finding the steady state
    print("Steady state determination was successful")
    print("\tstates =%s" % sol.state)
    print("\tinputs =%s" % sol.inputs)

    # Determine the jacobian of the whole system at the steady state
    jacobi_config = LinearizationConfiguration(system,
                                               state=sol.state,
                                               inputs=sol.inputs)
    # We want to get the total force and torque as outputs
    force_output = OutputDescriptor(jacobi_config, forces_sum)
    torque_output = OutputDescriptor(jacobi_config, torques_sum)

    A, B, C, D = system_jacobian(jacobi_config)
    print("A:")
    print(A)
    print("B:")
    print(B)
    print("C:")
    print(C)
    print("D:")
    print(D)
else:
    print("Steady state determination failed with message=%s" % sol.message)
    print(sol)
