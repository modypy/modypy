A Pendulum
==========

In this exercise we will model a simple pendulum: A mass suspended by a stiff
connector from a fixed support. We will model the angle and angular velocity of
the pendulum as states and let the pendulum move from some initial position.
At the end of this exercise, you will know how to model such a pendulum and by
that will learn how to work with multiple states.

In :numref:`pendulum` you can see a schematic drawing of a pendulum just like
the one we are going to model. We will not consider drag.

.. _pendulum:
.. figure:: 02_pendulum.svg
    :align: center
    :alt: Pendulum with quantities

    Pendulum with mass, length, angle and gravitational force marked

Our pendulum has two states:

- the current angle :math:`\alpha\left(t\right)` and
- the current angular velocity :math:`\omega\left(t\right)`.

Newtonian mechanics tells us that

.. math::
    \frac{d}{dt} \omega\left(t\right) =
    - \frac{g \sin\left(\alpha\left(t\right)\right)}{l}

Our additional equation for the state :math:`\alpha\left(t\right)` comes from
the relationship between position and velocity:

.. math::
    \frac{d}{dt} \alpha\left(t\right) = \omega\left(t\right)

In addition, we have our initial conditions for both of the states:

.. math::
    \alpha\left(t_0\right) & = \alpha_0 \\
    \omega\left(t_0\right) &= \omega_0

We will now implement the system described by these equations. Again we start
by importing all necessary definitions. In addition, we will define some
constants for our system:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    from modypy.model import System, State
    from modypy.simulation import Simulator

    # Define the system parameters
    LENGTH = 1.0
    GRAVITY = 9.81

    # Define the initial conditions
    ALPHA_0 = np.deg2rad(10)
    OMEGA_0 = 0

Now let us define our system and our two states:

.. code-block:: python

    # Create the system
    system = System()

    # Define the derivatives of the states
    def alpha_dt(data):
        return data.states[omega]


    def omega_dt(data):
        return -GRAVITY/LENGTH * np.sin(data.states[alpha])
        pass


    # Create the alpha state
    alpha = State(system,
                  derivative_function=alpha_dt,
                  initial_condition=ALPHA_0)

    # Create the omega state
    omega = State(system,
                  derivative_function=omega_dt,
                  initial_condition=OMEGA_0)

In the previous example, our state derivative function for the integrator only
depended on the input signal. Here, each of the derivative functions depend on
the value of the respective other state and use the ``data.states`` dictionary
to retrieve that value.

Again, we set up a simulator and run the system for 10 seconds:

.. code-block:: python

    # Run a simulation
    simulator = Simulator(system, start_time=0.0)
    msg = simulator.run_until(time_boundary=10.0)

    if msg is not None:
        print("Simulation failed with message '%s'" % msg)
    else:
        # Plot the result
        alpha_line, omega_line = \
            plt.plot(simulator.result.time,
                     simulator.result.state[:, alpha.state_slice],
                     'r',
                     simulator.result.time,
                     simulator.result.state[:, omega.state_slice],
                     'g')
        plt.legend((alpha_line, omega_line), ('Alpha', 'Omega'))
        plt.savefig("02_pendulum.png")
        plt.show()

The result of that simulation can be seen in :numref:`pendulum_simulation`.

.. _pendulum_simulation:
.. figure:: 02_pendulum_simulation.png
    :align: center
    :alt: Results of pendulum simulation

    Results of pendulum simulation: Angle and angular velocity

If you want, you can now play around with the parameters gravity and length or
the initial states. For example, you can give the pendulum some initial impulse
by setting ``OMEGA_0`` to some value other than 0.

As an additional exercise, try to integrate drag into the system, specified by
the drag coefficient :math:`\gamma` with :math:`\gamma>0`:

.. math::
    \frac{d}{dt} \omega\left(t\right) =
    - \frac{g \sin\left(\alpha\left(t\right)\right)}{l}
    - \gamma \omega\left(t\right)
