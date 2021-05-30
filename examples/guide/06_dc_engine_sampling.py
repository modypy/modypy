"""
An engine consisting of a DC motor and a static propeller, sampling the
generated thrust at regular intervals.
"""
import matplotlib.pyplot as plt

from modypy.model import System, SignalState, Block, Clock
from modypy.blocks.aerodyn import Propeller
from modypy.blocks.elmech import DCMotor
from modypy.blocks.sources import constant
from modypy.simulation import Simulator, SimulationResult


# Define the engine
class Engine(Block):
    """A block defining an engine consisting of a DC motor and a propeller"""

    def __init__(self,
                 parent,
                 thrust_coefficient,
                 power_coefficient,
                 diameter,
                 motor_constant,
                 resistance,
                 inductance,
                 moment_of_inertia):
        Block.__init__(self, parent)

        # Create the DC motor and the propeller
        self.dc_motor = DCMotor(self,
                                motor_constant=motor_constant,
                                resistance=resistance,
                                inductance=inductance,
                                moment_of_inertia=moment_of_inertia)
        self.propeller = Propeller(self,
                                   thrust_coefficient=thrust_coefficient,
                                   power_coefficient=power_coefficient,
                                   diameter=diameter)

        # We will simply pass through the voltage and density ports of the
        # motor and the propeller
        self.voltage = self.dc_motor.voltage
        self.density = self.propeller.density

        # We also pass on the thrust and the torque of the whole engine
        self.thrust = self.propeller.thrust
        self.torque = self.dc_motor.torque

        # The propeller needs to know the speed of the motor axle
        self.dc_motor.speed_rps.connect(self.propeller.speed_rps)

        # The DC-motor needs to know the torque required by the propeller
        self.propeller.torque.connect(self.dc_motor.external_torque)


# Create the system and the engine
system = System()
engine = Engine(system,
                motor_constant=789.E-6,
                resistance=43.3E-3,
                inductance=1.9E-3,
                moment_of_inertia=5.284E-6,
                thrust_coefficient=0.09,
                power_coefficient=0.04,
                diameter=8*25.4E-3)

# Provide constant signals for the voltage and the air density
voltage = constant(value=3.5)
density = constant(value=1.29)

# Connect them to the corresponding inputs of the engine
engine.voltage.connect(voltage)
engine.density.connect(density)

# Set up the state for keeping the sampled value.
sample_state = SignalState(system)

# Create a clock for sampling at 100Hz
sample_clock = Clock(system, period=0.01)


# Define the function for updating the state
def update_sample(system_state):
    """Update the state of the sampler"""
    sample_state.set_value(system_state, engine.thrust(system_state))


# Register it as event handler on the clock
sample_clock.register_listener(update_sample)

# Create the simulator and run it
simulator = Simulator(system, start_time=0.0)
result = SimulationResult(system, simulator.run_until(time_boundary=0.5))

# Plot the result
plt.plot(result.time, engine.thrust(result), "r", label="Continuous-Time")
plt.step(result.time, sample_state(result), "g", label="Sampled", where="post")
plt.title("Engine with DC-Motor and Static Propeller")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Thrust")
plt.savefig("06_dc_engine_sampling.png")
plt.show()
