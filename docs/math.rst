Mathematical Basics
===================

`Dynamical systems <https://en.wikipedia.org/wiki/Dynamical_system>`_ are systems
which evolve over time. The evolution is usually described in terms of the
internal state of said system as well as its input and output signals.
These relationships are described by a mathematical formulation.

MoDyPy deals with such dynamical systems in state-space representation.
Specifically, dynamical systems in MoDyPy are modelled as continuous-time
systems, using the following principal formulation:

.. math::
    \vec{x}\left(t_0\right) &= \vec{x}_0 \\
    \frac{d}{dt} \vec{x}\left(t\right) &=
        \vec{g}\left(t, \vec{x}\left(t\right), \vec{u}\left(t\right)\right) \\
    \vec{y}\left(t\right) &=
        \vec{h}\left(t, \vec{x}\left(t\right), \vec{u}\left(t\right)\right)

Here,
- :math:`\vec{x}\left(t\right)` is a vector that represents the *state* of
the system at time :math:`t`,
- :math:`\vec{u}\left(t\right)` is a vector that represents the *inputs* to the
system at that time, and
- :math:`\vec{x}_0` is the *initial state* or *initial condition* of the system
at time :math:`t_0`.

The *output* of the system is described by the function :math:`\vec{h}`, which
may depend on the time, the state and the input vector.
Similarly, the function :math:`\vec{g}` describes the derivative of the state
with respect to time. That is, it describes how the state evolves with time.
Again, it may also depend on the time, the state and the input vector.

Finding the value of :math:`\vec{x}\left(t\right)` for some value of :math:`t`
based on the equations given above is called an *initial value problem*, and
closed solutions are only known for specific forms of :math:`\vec{g}`, such as
linear forms. However, for most, approximate numerical solutions are required.

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

However, MoDyPy is able to handle non-linear and linear systems. It can also
:doc:`derive linear approximations <api/packages/linearization>` of the dynamics
of a non-linear system.
