Steady-State: A Water Tank
==========================

Besides simulation, another interesting application is steady state
determination. A steady state is a condition for a dynamical system where the
state of the system does not change with time. Steady states are important for
designing linear controllers for non-linear systems, as the knowledge of a
steady state allows to create a linear approximation of the system around that
steady state and based on that to create a linear controller.

In this example we will model the outflow from a water tank and determine the
amount of inflow necessary to keep the height at a specific level. After this
example you will know

- how to explicitly model inputs to a system,
- how to express constraints on specific signals in a system using output ports,
  and
- how to use the steady-state determination functionality of `MoDyPy` to find a
  steady state that fulfills the constraints.

A Water Tank
------------

We will model a water tank with an inflow and an outflow, as shown in
:numref:`tank_flow`. The tank has a cross section area of :math:`A_t`.
An inflow with a cross section of :math:`A_1` provides incoming water at a
velocity of :math:`v_1\left(t\right)` and the current fill height of the tank is
:math:`h\left(t\right)`. As a consequence of the pressure of the water, the
water flows out of the tank through an outflow with cross section :math:`A_2`
at a velocity of :math:`v_2\left(t\right)`.

.. _tank_flow:
.. figure:: 07_tank_flow.svg
    :align: center
    :alt: Tank with inflow and outflow

    Tank with inflow and outflow

According to Toricelli's Law, we know that

.. math::
    v_2\left(t\right) = \sqrt{2 g h\left(t\right)}

Thus, we can describe the change in height :math:`\dot{h}\left(t\right)` as
follows:

.. math::
    \dot{h}\left(t\right) =
    \frac{A_1 v_1\left(t\right) - A_2 \sqrt{2 g h\left(t\right)}}{A_t}

It is clear that for any given height :math:`h_0` an inflow of

.. math::
    v_1\left(t\right) = \sqrt{2 g h_0} \frac{A_2}{A_1}

is required to keep a steady height. However, we will now determine this
numerically.

First, we will import all the necessary declarations, define some constants and
create a new system:

.. code-block:: python

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

In our problem, the inflow velocity is an input that may have to be determined
as part of the steady-state determination. In order for the steady-state
determination algorithm to recognize it as an input it can modify, we declare it
as an :class:`InputSignal <modypy.model.ports.InputSignal>`.

.. code-block:: python

    # Model the inflow
    inflow_velocity = InputSignal(system)


Now we can define our fill height as a state:

.. code-block:: python

    def height_derivative(data):
        """Calculate the time derivative of the height"""

        return (A1*data.signals[inflow_velocity]
                - A2*np.sqrt(2*G*data.states[height_state]))/At


    height_state = SignalState(system, derivative_function=height_derivative)

We have one constraint, which is that the height shall be at the value given by
`TARGET_HEIGHT`. We model this by determining the difference between the current
and the target height. To tell the steady-state determination algorithm that we
want this difference to be zero, we declare an
:class:`OutputPort <modypy.model.port.OutputPort>` and connect it to the signal
showing the difference:

.. code-block:: python

    # Define the target height
    target_height = constant(system, TARGET_HEIGHT)

    # Express the output constraint
    height_delta = sum_signal(system,
                              input_signals=(height_state, target_height),
                              gains=(1, -1))
    height_delta_target = OutputPort(system)
    height_delta_target.connect(height_delta)

Now our system including its constraints and inputs is defined and we can run
the steady-state algorithm. The algorithm returns a tuple consisting of

- a :class:`scipy.optimize.OptimizeResult` object showing whether the search
  converged,
- an array giving the state at which the steady state situation occurs, and
- an array giving the values of the input ports for which the steady state
  situation occurs.

We will print these together with the theoretical steady state of our system:

.. code-block:: python

    print("Target height: %f" % TARGET_HEIGHT)
    print("Steady state height: %f" % steady_state[height_state.state_slice])
    print("Steady state inflow: %f" % steady_inputs[inflow_velocity.input_slice])
    print("Theoretical state state inflow: %f" % (
        np.sqrt(2*G*TARGET_HEIGHT)*A2/A1
    ))

Running this code should give us the following output:

.. code-block::

    Target height: 5.000000
    Steady state height: 5.000000
    Steady state inflow: 19.809089
    Theoretical state state inflow: 19.809089

We see that the determined and the theoretical inflow coincide and that the
height is at the target that we want it to be. Playing around with the target
height we get different values:

.. code-block::

    Target height: 7.000000
    Steady state height: 7.000000
    Steady state inflow: 23.438430
    Theoretical state state inflow: 23.438430
