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
                                   initial_condition=initial_speed)

        # Create the output for the speed in revolutions per second
        self.speed_rps = Signal(self,
                                value=self.speed_rps_value)

        # Create the output for the generated torque
        self.torque = Signal(self,
                             value=self.torque_value)

        # Create (input) ports for voltage and external torque load
        self.voltage = Port(self)
        self.external_load = Port(self)

    def omega_dt(self, data):
        return ((self.motor_constant * data.states[self.current]
                 - data.signals[self.external_load]) /
                self.moment_of_inertia)

    def current_dt(self, data):
        return ((data.signals[self.voltage]
                 - self.resistance * data.states[self.current]
                 - self.motor_constant * data.states[self.omega]) /
                self.inductance)

    def speed_rps_value(self, data):
        return data.signals[self.omega] / (2 * np.pi)

    def torque_value(self, data):
        return self.motor_constant * data.states[self.current]


# Define the static propeller block
class StaticPropeller(Block):
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
        self.thrust = Signal(self,
                             value=self.thrust_output)
        self.torque = Signal(self,
                             value=self.torque_output)

        # Define the input ports for propeller speed and air density
        self.speed_rps = Port(self)
        self.density = Port(self)

    def thrust_output(self, data):
        rho = data.signals[self.density]
        n = data.signals[self.speed_rps]
        return self.thrust_coefficient * rho * self.diameter ** 4 * n ** 2

    def torque_output(self, data):
        rho = data.signals[self.density]
        n = data.signals[self.speed_rps]
        return self.power_coefficient / (2 * np.pi) * \
            rho * self.diameter ** 5 * n ** 2


# Define the engine
class Engine(Block):
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
        self.propeller = StaticPropeller(self,
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
        self.propeller.torque.connect(self.dc_motor.external_load)


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
             simulator.result.signals[:, engine.thrust.signal_slice])
    plt.title("Engine with DC-Motor and Static Propeller")
    plt.xlabel("Time")
    plt.ylabel("Thrust")
    plt.savefig("05_dc_engine_simulation.png")
    plt.show()