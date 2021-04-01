"""
An engine consisting of a DC motor and a static propeller.
"""
import numpy as np
import matplotlib.pyplot as plt

from modypy.model import System, Signal, SignalState, Block, Port
from modypy.blocks.sources import constant
from modypy.simulation import Simulator


# Define the DC-motor block
class DCMotor(Block):
    """A block describing a DC-motor"""

    def __init__(self,
                 parent,
                 motor_constant,
                 resistance,
                 inductance,
                 moment_of_inertia,
                 initial_speed=None,
                 initial_current=None):
        Block.__init__(self, parent)
        self.motor_constant = motor_constant
        self.resistance = resistance
        self.inductance = inductance
        self.moment_of_inertia = moment_of_inertia

        # Create the velocity and current state
        # These can also be used as signals which export the exact value of
        # the respective state.
        self.omega = SignalState(self,
                                 derivative_function=self.omega_dt,
                                 initial_condition=initial_speed)
        self.current = SignalState(self,
                                   derivative_function=self.current_dt,
                                   initial_condition=initial_current)

        # Create the output for the speed in revolutions per second
        self.speed_rps = Signal(value=self.speed_rps_value)

        # Create the output for the generated torque
        self.torque = Signal(value=self.torque_value)

        # Create (input) ports for voltage and external torque load
        self.voltage = Port()
        self.external_torque = Port()

    def omega_dt(self, data):
        """Calculate the derivative of the angular velocity"""

        return ((self.motor_constant * self.current(data)
                 - self.external_torque(data)) /
                self.moment_of_inertia)

    def current_dt(self, data):
        """Calculate the derivative of the coil current"""

        return ((self.voltage(data)
                 - self.resistance * self.current(data)
                 - self.motor_constant * self.omega(data)) /
                self.inductance)

    def speed_rps_value(self, data):
        """Calculate the rotational velocity in rotations per second"""

        return self.omega(data) / (2 * np.pi)

    def torque_value(self, data):
        """Calculate the total torque generated by the motor"""

        return self.motor_constant * self.current(data)


# Define the static propeller block
class Propeller(Block):
    """A block representing a static propeller"""

    def __init__(self,
                 parent,
                 thrust_coefficient,
                 power_coefficient,
                 diameter):
        Block.__init__(self, parent)
        self.thrust_coefficient = thrust_coefficient
        self.power_coefficient = power_coefficient
        self.diameter = diameter

        # Define the thrust and torque output signals
        self.thrust = Signal(value=self.thrust_output)
        self.torque = Signal(value=self.torque_output)

        # Define the input ports for propeller speed and air density
        self.speed_rps = Port()
        self.density = Port()

    def thrust_output(self, data):
        """Calculate the thrust force of the propeller"""

        rho = self.density(data)
        n = self.speed_rps(data)
        return self.thrust_coefficient * rho * self.diameter ** 4 * n ** 2

    def torque_output(self, data):
        """Calculate the drag torque of the propeller"""

        rho = self.density(data)
        n = self.speed_rps(data)
        return self.power_coefficient / (2 * np.pi) * \
            rho * self.diameter ** 5 * n ** 2


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
voltage = constant(system, value=3.5)
density = constant(system, value=1.29)

# Connect them to the corresponding inputs of the engine
engine.voltage.connect(voltage)
engine.density.connect(density)

# Create the simulator and run it
simulator = Simulator(system, start_time=0.0)
msg = simulator.run_until(time_boundary=0.5)

if msg is not None:
    print("Simulation failed with message '%s'" % msg)
else:
    # Plot the result
    plt.plot(simulator.result.time,
             engine.thrust(simulator.result)[0])
    plt.title("Engine with DC-Motor and Static Propeller")
    plt.xlabel("Time")
    plt.ylabel("Thrust")
    plt.savefig("05_dc_engine_simulation.png")
    plt.show()
