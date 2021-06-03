"""
A planet orbiting a sun.
"""
import numpy.linalg as linalg
import matplotlib.pyplot as plt

from modypy.model import System, State
from modypy.simulation import Simulator, SimulationResult

# Define the system parameters
G = 6.67e-11  # m^3/s^2
SUN_MASS = 1.989e30  # kg

# Parameter of the Earth Orbit
PERIHEL = 147.1e9  # m
VELOCITY_PERIHEL = 30.29e3  # m/s
ROTATION_TIME = 365.256 * 24 * 60 * 60  # s

# Define the initial state
X_0 = (PERIHEL, 0)
V_0 = (0, -VELOCITY_PERIHEL)

# Create the system
system = System()


# Define the derivatives
def velocity_dt(system_state):
    """Calculate the derivative of the velocity"""
    pos = position(system_state)
    distance = linalg.norm(pos, axis=0)
    return -G * SUN_MASS / (distance ** 3) * pos


# Create the states
velocity = State(
    owner=system,
    shape=2,
    derivative_function=velocity_dt,
    initial_condition=V_0,
)
position = State(
    owner=system, shape=2, derivative_function=velocity, initial_condition=X_0
)

# Run a simulation
simulator = Simulator(system, start_time=0.0, max_step=24 * 60 * 60)
result = SimulationResult(
    system, simulator.run_until(time_boundary=ROTATION_TIME)
)

# Plot the result
trajectory = position(result)
plt.plot(trajectory[0], trajectory[1])
plt.gca().set_aspect("equal", "box")
plt.title("Planet Orbit")
plt.savefig("03_planet_orbit_trajectory.png")

plt.figure()
plt.plot(result.time, linalg.norm(trajectory, axis=0))
plt.xlabel("Time (s)")
plt.ylabel("Distance from the Sun (m)")
plt.title("Distance Earth-Sun over Time")
plt.savefig("03_planet_orbit_distance.png")
plt.show()
