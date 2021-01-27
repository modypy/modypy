MoDyPy
======

MoDyPy (rhymes with "modify") is a Python framework for *Mo*\ delling *dy*\ namic
systems in *Py*\ thon. The framework provides methods for describing
continuous and discrete-time linear and non-linear systems in
`state-space representation <https://en.wikipedia.org/wiki/State-space_representation>`_.
It was originally inspired by `simupy <https://github.com/simupy/simupy>`_
developed by Ben Margolis, but has a completely different philosophy and
architecture than simupy.

The basic components of a dynamic system in MoDyPy are
:doc:`states <api/packages/model/states>` and
:doc:`signals <api/packages/model/ports>`.
States represent the internal state of the system, and signals represent the
values calculated based on the state. Ports can be connected to signals, so that
reusable blocks with input and output ports can be easily built. For more details,
refer to the :doc:`Programmer's Guide <guide>` and the
:doc:`API Documentation <api/api>`.

Main Features
-------------

- Simple architecture based on states, signals and connectible ports
- Enables hierarchical modelling
- Allows the establishment of reusable building blocks
- Simulator for linear and non-linear continuous- and discrete-time systems
- Clock system to model periodic events and discrete-time components
- Steady state determination and linearization
- Library of standard blocks, including 6-degree-of-freedom rigid body motion
- Tested for 100% statement and branch coverage

Installation
------------

MoDyPy is available via the *pip* installer:

.. code-block:: bash

  $ pip install modypy

To install the development version,

.. code-block:: bash

  $ git clone https://github.com/ralfgerlich/modypy.git
  $ pip install -e modypy

Examples
--------

.. figure:: guide/06_dc_engine_sampling.png
    :align: center
    :alt: Simulation of a DC-motor with propeller

    Simulation of a DC-motor with propeller

Check out the examples in the ``examples`` directory and the
:doc:`Programmer's Guide <guide>`, which include:

``dcmotor.py``
    A simple example using a DC-motor driving a propeller and sampling the
    thrust using a zero-order hold.
``rigidbody.py``
    Some rigid-body simulation using moments and forces showing an object
    moving in a circle with constant velocity and turn-rate.
``bouncing_ball.py``
    An example modelling a bouncing ball, demonstrating the use of events and
    event-handler functions.
``quadcopter_trim.py``
    A larger example showcasing the steady-state-determination and linearisation
    of complex systems, in this case for a quadrocopter frame with four
    DC-motors with propellers.

They can be run from the sources using, e.g.,

.. code-block:: bash

  $ pip install matplotlib
  $ python examples/bouncing_ball.py

Note that some of the examples require ``matplotlib`` to run and display the
results.

.. toctree::
    :glob:
    :hidden:
    :maxdepth: 2

    math
    guide
    api/api
