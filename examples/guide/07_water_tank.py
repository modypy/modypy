"""
Steady-state determination for a water tank.
"""
import numpy as np

from modypy.model import System, SignalState, InputSignal, OutputPort
from modypy.blocks.sources import constant
from modypy.blocks.linear import sum_signal
from modypy.linearization import find_steady_state

# Constants
G = 9.81    # Gravity
A1 = 0.01   # Inflow cross section
A2 = 0.02   # Outflow cross section
At = 0.2    # Tank cross section
TARGET_HEIGHT = 5

# Create a new system
system = System()

# Model the inflow
inflow_velocity = InputSignal(system)


# Model the height state
def height_derivative(data):
    """Calculate the time derivative of the height"""

    return (A1*data.signals[inflow_velocity]
            - A2*np.sqrt(2*G*data.states[height_state]))/At


height_state = SignalState(system, derivative_function=height_derivative)

# Define the target height
target_height = constant(system, TARGET_HEIGHT)

# Express the output constraint
height_delta = sum_signal(system,
                          input_signals=(height_state, target_height),
                          gains=(1, -1))
height_delta_target = OutputPort(system)
height_delta_target.connect(height_delta)

# Find the steady state
result, steady_state, steady_inputs = find_steady_state(system, time=0)
print("Target height: %f" % TARGET_HEIGHT)
print("Steady state height: %f" % steady_state[height_state.state_slice])
print("Steady state inflow: %f" % steady_inputs[inflow_velocity.input_slice])
print("Theoretical state state inflow: %f" % (
    np.sqrt(2*G*TARGET_HEIGHT)*A2/A1
))
