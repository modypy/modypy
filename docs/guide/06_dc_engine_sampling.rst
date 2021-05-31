Discrete-Time: Sampling a DC-Engine
===================================

In the previous examples in this guide we only used continuous-time simulation.
However, MoDyPy is also capable of modelling systems that mix continuous-
and discrete-time elements.
The discrete-time elements are modelled using clocks and states that do not have
a derivative function, but which are instead updated as a reaction to events ---
either from a clock or from a zero-crossing event source.

In the preceding example we modelled a DC-engine with a propeller.
In this example, we will extend that system by sampling the generated thrust at
regular time intervals.
At the end of this exercise you will know

- how to define clocks,
- how to introduce states for discrete-time systems, and
- how to update states when a clock tick occurs.

For this, we will slightly modify the DC-engine example from the previous
exercise.
The DC-motor block and the propeller block are also provided as standard blocks,
which we will use in this example.

Defining our System
-------------------

We again start by importing our required elements:

.. code-block:: python

    import matplotlib.pyplot as plt

    from modypy.model import System, SignalState, Block, Clock
    from modypy.blocks.aerodyn import Propeller
    from modypy.blocks.elmech import DCMotor
    from modypy.blocks.sources import constant
    from modypy.simulation import Simulator

We will re-use the ``Engine`` class as well as the system definition from the
previous exercise:

.. code-block:: python

    class Engine(Block):
        # ...

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

Now, we define a state for keeping the last sampled value of the thrust signal:

.. code-block:: python

    sample_state = SignalState(system)

Note how we did not specify a derivative function for this state.
Internally, this is modelled as the derivative function being a constant zero,
i.e., the state does not change over time.

However, we want our state to change, but we want it to change at specific,
periodic events.
For this, we first declare a clock, that will deliver such a stream of periodic
events:

.. code-block:: python

    sample_clock = Clock(system, period=0.01)

Finally, we define a function that updates our sampling state from the
sampled signal, and we register that function as an event handler on the clock:

.. code-block:: python

    def update_sample(system_state):
        """Update the state of the sampler"""
        sample_state.set_value(system_state, engine.thrust(system_state))


    sample_clock.register_listener(update_sample)

Running the Simulation
----------------------

Our system is now fully defined.
Now we want to run a simulation of it and plot the results:

.. code-block:: python

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

The result is shown in :numref:`dc_engine_sampling`.

.. _dc_engine_sampling:
.. figure:: 06_dc_engine_sampling.png
    :align: center
    :alt: DC-Engine simulation with discrete-time sampling

    DC-Engine simulation with discrete-time sampling

Note that this time we did not specify the `max_step` parameter.
The simulator takes care that intermediate samples are available on every tick
of every clock in our system.

Of course, we could add the `max_step` parameter anyway, for example, if we were
not only interested in the behaviour of the system at the clock ticks, but also
in between.
However, for simulation of a discrete-time control system, we might be satisfied
with simulating the system accurately at the sampling points implied by the
sampling clock.

Working with Clocks
-------------------

There are many possibilities for defining clocks.
Multiple clocks may have different periods, or they may have the same period but
be offset against each other, they may only run until a specific point in time
and then stop.
Have a look at :class:`modypy.model.events.Clock` to find out about all the
possibilities.

Also, the quicker way of introducing a so-called `zero-order hold
<https://en.wikipedia.org/wiki/Zero-order_hold>`_ element as we did here is
using the :func:`modypy.blocks.discrete.zero_order_hold` function.
