The Model API
=============

The Model API provides the basic elements required for building models:

Systems and Blocks (:mod:`modypy.model.system`)
    are the basic building blocks of models in MoDyPy. Each system exists in the
    context of a :class:`modypy.model.system.System` object. The
    :class:`modypy.model.system.Block` class supports the creation of re-usable
    blocks in hierarchical models.

Ports and Signals (:mod:`modypy.model.ports`)
    provide the means for parts of the system to communicate with each other.
    Signals (:class:`modypy.model.ports.Signal`) are the sources of values in the
    system, and are associated with a value or a function that is used to
    dynamically calculate the value of the signal for any given time and state
    of the system. Ports (:class:`modypy.model.ports.Port`) can be connected to
    signals and transport their values. Any signal is also a port that is
    connected to itself.

States (:mod:`modypy.model.states`)
    are used to describe the evolution of the behaviour of a system over
    time. Each state (:class:`modypy.model.states.State`) is associated with a
    shape, an initial value and a derivative function that describes its
    evolution over time. States can also serve as the source of signals,
    (:class:'modypy.model.states.SignalState`) providing direct access to
    the state contents for other elements.

Events (:mod:`modypy.model.events`)
    provide the means to handle the occurrence occurrence of special events
    during simulation and to react to them by modifying the state of the system.
    An event may either be defined by a clock tick
    (:class:`modypy.model.events.Clock`) or by the sign-change of a special
    event function which may depend on the time, the state and the values of the
    signals in the system (:class:`modypy.model.events.ZeroCrossEventSource`).
    Events and specifically clocks are the basis of the simulation of
    discrete-time elements in a system.

Evaluation (:mod:`modypy.model.evaluation`)
    is done by :class:`modypy.model.evaluation.Evaluator` objects, which lazily
    determine the values of signals, ports, events and state derivatives at one
    specific point in time. The dependencies between signals are considered and
    algebraic loops are automatically detected and reported. Calculation results
    are cached for better performance.

.. toctree::
    :hidden:
    :glob:
    :maxdepth: 4

    packages/model/*
