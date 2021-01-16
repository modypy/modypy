A Simple Integrator
===================

As our first model, we will describe a simple integrator with a sine wave input.
As output we should also get a sine wave, just shifted in phase. After this
exercise, you will know

- how to create a system,
- how to add signals and states to it,
- how to run a simulation of the system, and
- how to access simulation results.

An integrator is a very simple dynamic element with a single state
:math:`x\left(t\right)`. The derivative :math:`\frac{d}{dt} x\left(t\right)` of
that state is just the input into the system. In addition to the derivative, we
need to specify the initial value of the state :math:`x\left(t_0\right)`:

.. math::
    x\left(t_0\right) &= x_0 \\
    \frac{d}{dt} x\left(t\right) &= u\left(t\right)

To view our results, we will use ``matplotlib``, so we install that first:

.. code-block:: bash

    $ pip install matplotlib

Now let us code our model. We start by importing the relevant declarations:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    from modypy.model import System, State, Signal
    from modypy.simulation import Simulator

All models in ``modypy`` are contained in a
:class:`System <modypy.model.system.System>`, so we need to create an instance:

.. code-block:: python

    system = System()

Now we can add states, signals and other elements to that system. Let us first
define a function that will calculate the value of our input and create a signal
out of it:

.. code-block:: python

    def sine_input(data):
        return np.sin(data.time)

    input_signal = Signal(system,
                          shape=1,
                          value=sine_input)

We add the :class:`signal <modypy.model.ports.Signal>` to ``system`` by
specifying that as the *owner* of the signal. Our signal is a scalar, which is
why we specify a ``shape`` of ``1``. The value is determined by the
``sine_input`` function.

The ``sine_input`` function is passed a
:class:`structure <modypy.model.evaluation.DataProvider>` with a property ``time``
that is set to the current time. We use that to make our input signal change
with time.

Now we need to create the integrator. For that we first provide a function that
calculates the value of the derivative of our state. That function simply
returns the value of our input signal:

.. code-block:: python

    def integrator_dt(data):
        return data.signals[input_signal]

    integrator_state = State(system,
                             shape=1,
                             derivative_function=integrator_dt,
                             initial_condition=-1)

Here, we see another feature of the
:class:`structure <modypy.model.evaluation.DataProvider>` passed to evaluation
functions in ``modypy``: The ``signals`` property holds a dictionary that can
be accessed using the signal objects and will provide the current value of that
signal.

The state itself also is a scalar, so it has the same shape as our signal. Note
that signals and states by default are scalar, so you could as well remove the
``shape`` parameter.

The ``initial_condition`` specifies the initial value of the state. This is the
value the state has when the simulation starts. We set it to ``-1`` so that we
get a nice cosine-wave.

The ``derivative_function`` is the function that gives our time derivative of
our state. In our case, this is simply the current value of our input signal.

Now, our system is already complete. We have our signal source and our integrator
state. Let's have a look at the motion of our system. For that, we create a
:class:`Simulator <modypy.simulation.Simulator>`:

.. code-block:: python

    simulator = Simulator(system,
                          start_time=0.0)

We set the start time for the simulation to ``0``. To run the simulation, we
have to call ``run_until``:

.. code-block:: python

    msg = simulator.run_until(time_boundary=10.0)

The ``time_boundary`` parameter gives the time until that the simulation should
be run. In our case, we want the simulation to run for ten time-units. You can
think of this as minutes, but if your system is expressed in the proper units,
these can also be hours, days, years, or whatever you need to use.

The result value of the ``run_until`` method is ``None`` when the simulation was
successful and any other value if it failed. In that case, the result value gives
some indication as to the reason for the failure.

We check it and in case of failure print the reason. Otherwise, we want to plot
the input and the integrator state:

.. code-block:: python

    if msg is not None:
        print("Simulation failed with message '%s'" % msg)
    else:
        # Plot the result
        input_line, integrator_line = \
            plt.plot(simulator.result.time,
                     simulator.result.signals[:, input_signal.signal_slice],
                     'r',
                     simulator.result.time,
                     simulator.result.state[:, integrator_state.state_slice],
                     'g')
        plt.legend((input_line, integrator_line), ('Input', 'Integrator State'))
        plt.xlabel("Time")
        plt.savefig("01_integrator_simulation.png")
        plt.show()

The result of that simulation can be seen in :numref:`integrator_simulation`.

.. _integrator_simulation:
.. figure:: 01_integrator_simulation.png
    :align: center
    :alt: Results of integrator simulation

    Results of integrator simulation: Input and integrator state

In red, we see the input signal, while the value of our integrator state is
plotted in green. Looks quite correct.

But what happened here? We accessed the ``result`` property of our simulator.
This is an instance of :class:`SimulatorResult <modypy.simulation.SimulatorResult>`
and provides access to the values of our signals and states in the ``signals``
and ``state`` property.

These are represented as two-dimensional vectors, with the first dimension
representing the sample index, and the second dimension representing the state
or signal index. The sampling timestamp for each of the samples can be found in
the ``time`` property, which is a one-dimensional array with the index being the
sample-index.

Upon creation, each signal and state is assigned a range of consecutive state or
signal indices. The number of these indices for each state or signal depend on
the shape of the signal or state. A scalar signal/state will be assigned a single
index, but a state that is a 3x3 matrix will be assigned 9 indices.

The slice of indices assigned to a signal can be retrieved by using the
``signal_slice`` method. Similarly, we can get the slice of state indices for a
state by using ``state_slice``.

In the example above, we plot both the input signal and the integrator state
against time. If we wanted, we could do other things with these results, such
as checking the performance of a controller we built against control performance
constraints and many other things.
