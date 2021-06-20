Simulation
==========

.. contents::

One major part of the functionality of MoDyPy is its simulation capacity,
implemented in the :doc:`Simulation API </api/packages/simulation>`.
The main part of that is implemented in the
:class:`Simulator <modypy.simulation.Simulator>` class.

The simulator is mainly a logical wrapper around a solver for ordinary
differential equations (ODEs).
Such solvers are, for example, implemented in the :mod:`integrate
<scipy.integrate>` module provided by SciPy, namely the subclasses of
:class:`ODESolver <scipy.integrate.ODESolver>`.
In addition to integration of the differential equations, the simulator handles
zero-crossing- and clock events.

A Simple Simulation Loop
------------------------

For running a simulation of a simple system from time :math:`t_0` to time
:math:`t_f` the algorithm could look like this:

1. Initialize the current time :math:`t` and the current system state
   :math:`\vec{x}\left(t\right)` to their initial values.
2. Provide the current system state to the caller
3. Repeat until :math:`t=t_f`:

  1. Instruct the ODE-solver to determine the next sample of the system state
     :math:`\vec{x}\left(t'\right)` at some time
     :math:`t' \in \left(t, t_f\right]`.
  2. Advance the time :math:`t` to :math:`t'` and set the system state to the
     state provided by the ODE-solver.
  3. Provide the current system state to the caller.

However, that does not consider any events occurring during simulation, such as
zero-crossing events or clock events.

Simulation with Clocks
----------------------

So instead, to cover the handling of clocks, the loop-part is modified as
follows:

1. Determine :math:`t_s:=min\left(t_n,t_f\right)`, where :math:`t_n` is the next
   time at which a clock will expire.
   If there is no further clock to expire, then :math:`t_n=t_f`.
2. Instruct the ODE-solver to determine the next sample of the system state
   :math:`\vec{x}\left(t'\right)` at some time
   :math:`t' \in \left(t, t_f\right]`,
3. Advance the time :math:`t` to :math:`t'` and set the system state to the
   state provided by the ODE-solver.
4. If there are any clocks expiring at the current time, execute their event
   handlers, if any.
5. Provide the current system state to the caller.

This ensures that the actions performed in the clock event handlers actually
work on the system state as found when the clock ticks.
Also note that the discontinuities introduced by the clock event handlers need
to be considered consistently when reporting the system state.
In this case, the system state after the modifications is reported --
corresponding to :math:`lim_{\tau \to t'^{-}} \vec{x}\left(\tau\right)`, the
right-hand-side limit of the system state function over time.

That does consider clocks, for which we do know the time at which they will
expire.
But what about zero-crossing events?

Simulation with Events
----------------------

To handle zero-crossing events, the main loop is modified as follows:


1. Determine :math:`t_s:=min\left(t_n,t_f\right)`, where :math:`t_n` is the next
   time at which a clock will expire.
2. Instruct the ODE-solver to determine the next sample of the system state
   :math:`\vec{x}\left(t'\right)` at some time
   :math:`t' \in \left(t, t_f\right]`,
3. If any zero-crossing events have occurred during the time interval
   :math:`\left[t,t'\right]`, handle them as follows:

    1. Determine the time :math:`t_e \in \left[t,t'\right]` where the
       first zero-crossing event occurred.
    2. Advance time :math:`t` to :math:`t_e` and determine the system state
       at that point in time.
    3. If the associated event has any handlers:
        1. Execute them, updating the system state.
        2. Provide the current system state to the caller.
        3. Continue at the start of the simulation loop (ignoring any possible
           further zero-crossing events).
    4. If the associated event has no handlers:
        1. Provide the current system state to the caller.
        2. Continue handling possible further zero-crossing events.

3. Advance the time :math:`t` to :math:`t'` and set the system state to the
   state provided by the ODE-solver.
4. If there are any clocks expiring at the current time, execute their event
   handlers, if any.
5. Provide the current system state to the caller.

The determination of whether a zero-crossing event has occurred is based on the
event function.
Currently, the simulator only checks whether the event function has changed its
sign between the last time step and the current time step according to the
specified direction for the event.
If not, it is assumed that the event did not occur in the time step.

*NOTE*: This means that even numbers of zero-crossings may be missed that way.
The algorithm implicitly assumes that the steps are small enough for events to
only occur at most once in a single integration interval.

The state updates introduced by zero-crossing- and clock-event handlers may
lead to additional zero-crossing-events to occur.
The simulator will detect these and run them as well.
This can lead to endless event sequences, e.g., if the handlers of two events
introduce changes that will lead to occurrence of the respective other event.

Detecting Excessive Event Sequences
-----------------------------------

The simulator keeps count of the number of events occurring in sequence.
That count will be reset to zero when at least a single integration step has
been executed without any events occurring.
If the maximum number of consecutive events is exceeded, the simulator will
throw an exception and abort simulation.

Simulating Discrete-Time Systems
--------------------------------

For systems that do not contain any continuous-time element, i.e. for which the
derivatives of all states are constant zero, a slightly different algorithm is
used for performance reasons:



1. Determine :math:`t_s:=min\left(t_n,t_f\right)`, where :math:`t_n` is the next
   time at which a clock will expire.
2. Advance the time to :math:`t_s`.
3. For any clocks expiring at the current time step, execute the associated
   event handlers, if any, and handle any zero-crossing events occurring as a
   consequence.
4. Provide the current system state to the caller.

*NOTE* Here, in discrete-only systems, zero-crossing events can only occur as a
consequence of a state update due to an expiring clock.
