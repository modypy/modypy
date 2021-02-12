API Documentation
=================

The API consists of five main parts:

:doc:`The Model API <model>`
    is used for building system models from states, signals, ports, events and
    blocks.

:doc:`The Blocks Library <blocks>`
    contains a set of often-used blocks, including sources and constants as
    well as blocks for aerodynamics, electro-mechanics and rigid-body motion.

:doc:`The Simulation API <packages/simulation>`
    provides the :class:`modypy.simulation.Simulator` class to run simulations
    of systems and accessing the trajectory data.

:doc:`The Steady-State API <packages/steady_state>`
    providing functions to identify steady-state configurations.

:doc:`The Linearization API <packages/linearization>`
    providing functions to identify steady-state configurations and linearizing
    the system in such configurations.

:doc:`The Utilities Module <utils>`
    providing utilities, e.g., for accessing measurement databases like the
    UIUC propeller database.

.. toctree::
    :hidden:
    :glob:
    :maxdepth: 4

    model
    blocks
    packages/simulation
    packages/steady_state
    packages/linearization
    utils
