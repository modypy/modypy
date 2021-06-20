Mathematical Basics
===================

.. contents::

`Dynamical systems <https://en.wikipedia.org/wiki/Dynamical_system>`_ are systems
which evolve over time. In general, such a system has a set of inputs and outputs,
both of which may change over time. In contrast, for a static system its output
:math:`\vec{y}\left(t\right)` only depends on the value of the input
:math:`\vec{u}\left(t\right)` at that exact point in time:

.. math::
    \vec{y}\left(t\right) = \vec{h}\left(\vec{u}\left(t\right)\right)

For a dynamic system, the output at a given point in time may also depend on the
inputs at other points in time. In the real world, all systems are *causal*, i.e.
the effect always comes after the cause. Thus, for a causal dynamic system, the
output at a point in time :math:`t` may only depend on the inputs at times
before :math:`t`.

That means that dynamic systems must have some kind of memory to represent the
effects of earlier inputs. We say that a dynamical system has a *state* and we
represent that state as a vector. For most practical applications that state
has finite dimension.

Continuous-time Systems
-----------------------

MoDyPy supports modelling continuous-time systems in state-space formulation:

.. math::
    \vec{x}\left(t_0\right) &= \vec{x}_0 \\
    \frac{d}{dt} \vec{x}\left(t\right) &=
        \vec{g}\left(t, \vec{x}\left(t\right), \vec{u}\left(t\right)\right) \\
    \vec{y}\left(t\right) &=
        \vec{h}\left(t, \vec{x}\left(t\right), \vec{u}\left(t\right)\right)

Here,

- :math:`\vec{x}\left(t\right)` is a vector that represents the *state* of
    the system at time :math:`t`, and
- :math:`\vec{x}_0` is the *initial state* or *initial condition* of the system
    at time :math:`t_0`.

The function :math:`\vec{g}` describes the derivative of the state
with respect to time. That is, it describes how the state evolves with time.
Again, it may also depend on the time, the state and the input vector.

Finding the value of :math:`\vec{x}\left(t\right)` for some value of :math:`t`
based on the equations given above is called an
`initial value problem <https://en.wikipedia.org/wiki/Initial_value_problem>`_,
and closed solutions are only known for specific forms of :math:`\vec{g}`, such
as those that are linear. For most other cases, numerical approximations of the
solutions are available and also sufficient.

Discrete-Time Systems
---------------------

While continuous-time systems allow us to easily describe typical physical
processes, there are some applications which require us to describe the progression
of a system in discrete time steps. In that case, the state and output only
change at specific points in time. Usually, these points in time are
regularly-spaced, and the time between two consecutive of such points in time is
called the *period* of the system.

A typical formulation you will find is the following:

.. math::
    \vec{x}_{k+1} &= \vec{g}\left(k, \vec{x}_k, \vec{u}_k\right) \\
    \vec{y}_k &= \vec{h}\left(k, \vec{x}_k, \vec{u}_k\right)

This kind of system can be easily simulated by a computer, as the state at the
respective next time step is explicitly given, while in the continuous case, an
initial value problem with differential equations needs to be solved first.
This is also why digital controllers or filters implemented in software are
typically modelled as discrete-time systems.

System Properties
-----------------

Systems may be classified according to a set of attributes, such as
*time-dependency*, *linearity* or *autonomy*. In the following, a few important
attributes for systems are given:

time-invariance
    A system is said to be *time-invariant* if both :math:`\vec{g}` and
    :math:`\vec{h}` are independent of the parameter :math:`t`. A system that
    is not time-invariant is said to be *time-varying*.

linearity
    A system is said to be *linear* if both :math:`\vec{g}` and
    :math:`\vec{h}` are linear in their parameters :math:`t`, :math:`\vec{x}` and
    :math:`\vec{u}`. In that case, both functions can take a specific form and
    closed solutions are available. For more information, see
    :ref:`linear-systems`. Systems that are not linear are called *non-linear*.

autonomy
    A system is said to be *autonomous* if both :math:`\vec{g}` and
    :math:`\vec{h}` are independent of the input :math:`\vec{u}`.

.. _linear-systems:

Linear Time-Invariant Systems
-----------------------------

A very interesting special class of systems are *linear and time-invariant*
systems, or *LTI* for short. For continuous-time systems, these can be expressed
in the following form:

.. math::
    \frac{d}{dt} \vec{x} &= A \vec{x} + B \vec{u} \\
    \vec{y} &= C \vec{x} + D \vec{u}

with matrices :math:`A`, :math:`B`, :math:`C`, :math:`D`. Linear discrete-time
systems can be expressed in a similar form:

.. math::
    \vec{x}_{k+1} &= A \vec{x}_k + B \vec{u}_k \\
    \vec{y}_k &= C \vec{x}_k + D \vec{u}_k

These systems are particularly interesting, because there are generic methods
for solving the differential and difference equations for the state in closed
form, for analysing the
`controllability <https://en.wikipedia.org/wiki/Controllability>`_,
`observability <https://en.wikipedia.org/wiki/Observability>`_ and
`stability <https://en.wikipedia.org/wiki/Stability_theory>`_ of such a system
and for designing `controllers <https://en.wikipedia.org/wiki/Control_system>`_
and so-called `observers <https://en.wikipedia.org/wiki/State_observer>`_.

However, MoDyPy is able to handle non-linear and linear systems of both the
discrete- and the continuous-time variant. Specifically, MoDyPy is designed to
handle mixed systems consisting of continuous-time and discrete-time parts,
also with mixed periods. It can also
:doc:`derive linear approximations </api/packages/linearization>` of the dynamics
of a continuous-time non-linear system.

Event Functions
---------------

Besides regular clocks, so-called mixed systems can also contain discrete
state transitions defined by the occurrence of events that are not bound to
time.
Examples for such events can be a ball hitting a surface (see the
:doc:`Bouncing Ball Tutorial </guide/04_bouncing_ball>`) or the velocity of a
car falling below the minimum at which adaptive cruise control is possible.

Such events can be described by so-called event functions, which are functions
mapping the state of the system to a scalar.
The event is said to occur when the event function changes its sign.
The sign change can also be constrained to a specific direction, so that the
event is said to occur only if the sign changes, for example, from positive to
negative.

State-changes can be modelled to be a consequence of such events.
For example, a model of a car with adaptive cruise control can actively
deactivate the controller when the velocity drops below the pre-defined
threshold.

However, such state changes can also lead to other state changes.
In the example of the bouncing ball, bouncing off the ground leads to loss of
energy in the ball due to internal friction.
As a consequence, the maximum height to which the ball will ascend after
bouncing will diminish with each bounce, and the time between bounces will
become smaller and smaller.

For the bouncing ball and other similar systems the maximum number of events
occurring in any given positive length of time may be unbounded.
That means that for any natural number :math:`n` and any positive length of time
:math:`\Delta t>0` we would be able to find a time interval
:math:`\left[t, t+\Delta t)` such that there are more than :math:`n` events in
that time interval.

This kind of behaviour is called
`Zeno-behaviour <https://en.wikipedia.org/wiki/Hybrid_system#Bouncing_ball>`_
after the Greek philosopher Zeno of Elea.
It brings a lot of problems for simulation and analysis of such hybrid systems,
and a lot of ways were invented trying to deal with them.
