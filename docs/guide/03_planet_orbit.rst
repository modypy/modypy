Multi-Dimensional States: A Planet Orbit
========================================

In this exercise we will model the system of a sun and a planet orbiting around
it.
We will model the two-dimensional position and velocity of the planet.
After completing this exercise, you will know

- how to model systems with multi-dimensional states and
- how to specify integrator options.

Deriving the Equations
----------------------

In :numref:`planet_orbit` you can see a schematic drawing of the sun and the
orbiting planet together with the position and the velocity of the planet.

.. _planet_orbit:
.. figure:: 03_planet_orbit.svg
    :align: center
    :alt: Sun and Planet

    Sun and Planet with governing quantities

Thanks to Isaac Newton, we know that the gravitational force acting on the
planet has the magnitude

.. math::
    F = G \frac{m M}{r^2}

where

- :math:`M` is the mass of the sun,
- :math:`m` is the mass of the planet,
- :math:`r` is the distance of the planet from the sun, and
- :math:`G` is the gravitational constant, with a value of approximately
  :math:`6.67\times 10^{-11} \frac{\text{m}^3}{\text{kg}\text{s}^2}`.

Thus, the acceleration of the planet is given by:

.. math::
    \frac{d}{dt} \vec{v}\left(t\right) =
    - G \frac{M}{\left|\vec{x}\left(t\right)\right|^2}
    \frac{\vec{x}\left(t\right)}{\left|\vec{x}\left(t\right)\right|}

And again, we have the time-derivative of the position equal the velocity:

.. math::
    \frac{d}{dt} \vec{x}\left(t\right) = \vec{v}\left(t\right)

With the initial conditions :math:`\vec{v}\left(t_0\right)=\vec{v}_0` and
:math:`\vec{x}\left(t_0\right)=\vec{x}_0` our system is defined.
So now let us implement it.

Defining our System
-------------------

Again, we first import the required modules:

.. code-block:: python

    import numpy as np
    import numpy.linalg as linalg
    import matplotlib.pyplot as plt

    from modypy.blocks.linear import integrator
    from modypy.model import System, State
    from modypy.simulation import Simulator, SimulationResult

We will need ``numpy.linalg`` to determine the norm of the position vector.

Following that we will define the system parameters and the initial states:

.. code-block:: python

    # Define the system parameters
    G = 6.67E-11*(24*60*60)**2
    SUN_MASS = 1.989E30
    PLANET_ORBIT = 149.6E09
    PLANET_ORBIT_TIME = 365.256

    # Define the initial state
    PLANET_VELOCITY = 2 * np.pi * PLANET_ORBIT / PLANET_ORBIT_TIME
    X_0 = PLANET_ORBIT * np.c_[np.cos(np.deg2rad(20)),
                               np.sin(np.deg2rad(20))]
    V_0 = 0.5*PLANET_VELOCITY * np.c_[-np.sin(np.deg2rad(20)),
                                      +np.cos(np.deg2rad(20))]

Note that we have scaled up the times to days to make them a bit more manageable.

Now let us define the system, its states and state derivatives:

.. code-block:: python

    # Create the system
    system = System()


    # Define the derivatives
    def velocity_dt(system_state):
        """Calculate the derivative of the velocity"""
        pos = position(system_state)
        distance = linalg.norm(pos)
        return -G * SUN_MASS/(distance**3) * pos


    # Create the states
    velocity = State(system,
                     shape=2,
                     derivative_function=velocity_dt,
                     initial_condition=V_0)
    position = integrator(system, input_signal=velocity, initial_condition=X_0)

The main thing that changed from the previous examples is that now our states
are two-dimensional.
In that case, ``modypy`` will provide their values as actual ``numpy`` arrays or
vectors in this case.

Running the Simulation
----------------------

Finally, let us set up a simulation, run it and plot the results:

.. code-block:: python

    # Run a simulation
    simulator = Simulator(system,
                          start_time=0.0,
                          max_step=1)
    result = SimulationResult(system,
                              simulator.run_until(time_boundary=PLANET_ORBIT_TIME))

    # Plot the result
    trajectory = position(result)
    plt.plot(trajectory[0], trajectory[1])
    plt.title("Planet Orbit")
    plt.savefig("03_planet_orbit_simulation.png")
    plt.show()

This time, we do not plot the values of the states over time, but instead we
plot the trajectory.
The result can be seen in :numref:`planet_orbit_simulation`.

.. _planet_orbit_simulation:
.. figure:: 03_planet_orbit_simulation.png
    :align: center
    :alt: Results of planet orbit simulation

    Results of planet orbit simulation

If you want, you can now play around a bit with the initial state or any of the
other parameters.
