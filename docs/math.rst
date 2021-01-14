Mathematical Basics
===================

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
based on the equations given above is called an *initial value problem*, and
closed solutions are only known for specific forms of :math:`\vec{g}`, such as
those that are linear. However, for most, only approximate numerical solutions
are required.

Systems may be classified according to a set of attributes, such as
*time-dependency*, *linearity* or *autonomy*. In the following, a few important
attributes for systems are given:

time-dependency
    A system is said to be *time-independent* if both :math:`\vec{g}` and
    :math:`\vec{h}` are independent of the parameter :math:`t`. A system that
    is not time-independent is said to be *time-dependent*.

linearity
    A system is said to be *linear* if both :math:`\vec{g}` and
    :math:`\vec{h}` are linear in their parameters :math:`\vec{x}` and
    :math:`\vec{u}`. In that case, both functions can take a specific form
    which will be discussed further down. Systems that are not linear are called
    *non-linear*.

autonomy
    A system is said to be *autonomous* if both :math:`\vec{g}` and
    :math:`\vec{h}` are independent of the input :math:`\vec{u}`.

A very interesting special class of systems are *linear and time-independent*
systems, or *LTI* for short. These can be expressed in the following form:

.. math::
    \frac{d}{dt} \vec{x} &= A \vec{x} + B \vec{u} \\
    \vec{y} &= C \vec{x} + D \vec{u}

with matrices :math:`A`, :math:`B`, :math:`C`, :math:`D`. These systems are
particularly interesting, because there are generic methods for solving the
differential equation for the state in closed form, for analysing the
`controllability <https://en.wikipedia.org/wiki/Controllability>`_,
`observability <https://en.wikipedia.org/wiki/Observability>`_ and
`stability <https://en.wikipedia.org/wiki/Stability_theory>`_ of such a system
and for designing `controllers <https://en.wikipedia.org/wiki/Control_system>`_
and so-called `observers <https://en.wikipedia.org/wiki/State_observer>`_.

However, MoDyPy is able to handle non-linear and linear continuous-time systems.
It can also :doc:`derive linear approximations <api/packages/linearization>` of
the dynamics of a non-linear system.

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

Again, linear and time-invariant discrete-time systems are special in that they
can be brought into a simpler form:

.. math::
    \vec{x}_{k+1} &= A \vec{x}_k + B \vec{u}_k \\
    \vec{y}_k &= C \vec{x}_k + D \vec{u}_k

MoDyPy can handle both linear and non-linear discrete-time systems. Specifically,
MoDyPy is designed to handle mixed systems consisting of continuous-time and
discrete-time parts, also with mixed periods.
