"""
Steady-state determination for a water tank.
"""
import numpy as np

from modypy.model import System, SignalState, InputSignal
from modypy.linearization import system_jacobian,\
    LinearizationConfiguration,\
    OutputDescriptor
from modypy.steady_state import SteadyStateConfiguration, find_steady_state

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

    return (A1*inflow_velocity(data)
            - A2*np.sqrt(2*G*height_state(data)))/At


height_state = SignalState(system, derivative_function=height_derivative)

# Configure for steady-state determination
steady_state_config = SteadyStateConfiguration(system)
# Enforce the inflow to be non-negative
steady_state_config.input_bounds[inflow_velocity.input_slice, 0] = 0
# Enforce the height to equal the target height
steady_state_config.state_bounds[height_state.state_slice] = TARGET_HEIGHT

# Find the steady state
result = find_steady_state(steady_state_config)
print("Target height: %f" % TARGET_HEIGHT)
print("Steady state height: %f" % height_state(result.system_state))
print("Steady state inflow: %f" % inflow_velocity(result.system_state))
print("Steady state height derivative: %f" %
      height_derivative(result.system_state))
print("Theoretical steady state inflow: %f" % (
    np.sqrt(2*G*TARGET_HEIGHT)*A2/A1
))

# Set up the configuration for finding the system jacobian
jacobian_config = LinearizationConfiguration(system,
                                             state=result.state,
                                             inputs=result.inputs)
# We want to have the height as output
output_1 = OutputDescriptor(jacobian_config, height_state)

# Find the system jacobian at the steady state
jac_A, jac_B, jac_C, jac_D = system_jacobian(jacobian_config,
                                             single_matrix=False)
print("Linearization at steady-state point:")
print("A=%s" % jac_A)
print("B=%s" % jac_B)
print("C=%s" % jac_C)
print("D=%s" % jac_D)
