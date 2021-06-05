"""
Steady-state determination for a water tank.
"""
import numpy as np

from modypy.model import System, InputSignal, State
from modypy.steady_state import SteadyStateConfiguration, find_steady_state

# Constants
G = 9.81  # Gravity
A1 = 0.01  # Inflow cross section
A2 = 0.02  # Outflow cross section
At = 0.2  # Tank cross section
TARGET_HEIGHT = 5

# Create a new system
system = System()

# Model the inflow
inflow_velocity = InputSignal(system)


# Model the height state
def height_derivative(system_state):
    """Calculate the time derivative of the height"""

    return (
        A1 * inflow_velocity(system_state)
        - A2 * np.sqrt(2 * G * height_state(system_state))
    ) / At


height_state = State(system, derivative_function=height_derivative)

# Configure for steady-state determination
steady_state_config = SteadyStateConfiguration(system)
# Enforce the height to equal the target height
steady_state_config.states[height_state].lower_bounds = TARGET_HEIGHT
steady_state_config.states[height_state].upper_bounds = TARGET_HEIGHT
# Enforce the inflow to be non-negative
steady_state_config.inputs[inflow_velocity].lower_bounds = 0

# Find the steady state
result = find_steady_state(steady_state_config)
print("Target height: %f" % TARGET_HEIGHT)
print("Steady state height: %f" % height_state(result.system_state))
print("Steady state inflow: %f" % inflow_velocity(result.system_state))
print(
    "Steady state height derivative: %f"
    % height_derivative(result.system_state)
)
print(
    "Theoretical steady state inflow: %f"
    % (np.sqrt(2 * G * TARGET_HEIGHT) * A2 / A1)
)
