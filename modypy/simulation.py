"""
Provide classes for simulation.
"""
from collections.abc import Sequence
import warnings

import numpy as np
import scipy.integrate
import scipy.optimize

from modypy.model import State, InputSignal, SystemState
from modypy.model.events import ClockQueue
from modypy.model.system import System

INITIAL_RESULT_SIZE = 16
RESULT_SIZE_EXTENSION = 16

DEFAULT_INTEGRATOR = scipy.integrate.DOP853


class SimulationError(RuntimeError):
    """Exception raised when an error occurs during simulation"""


class IntegrationError(SimulationError):
    """Exception raised when an error is reported by the integrator"""


class ExcessiveEventError(SimulationError):
    """
    Exception raised when an excessive number of successive events occurs.
    """


class SimulationResult(Sequence):
    """The results provided by a simulation.

    A `SimulationResult` object captures the time series provided by a
    simulation. It has properties `t`, `state` and `inputs` representing the
    time, state vector and inputs vector for each individual sample.
    """

    def __init__(self, system: System, source=None):
        self.system = system
        self._t = np.empty(INITIAL_RESULT_SIZE)
        self._inputs = np.empty((self.system.num_inputs, INITIAL_RESULT_SIZE))
        self._state = np.empty((self.system.num_states, INITIAL_RESULT_SIZE))

        self.current_idx = 0

        if source is not None:
            self.collect_from(source)

    @property
    def time(self):
        """The time vector of the simulation result"""
        return self._t[0 : self.current_idx]

    @property
    def inputs(self):
        """The input vector of the simulation result"""
        return self._inputs[:, 0 : self.current_idx]

    @property
    def state(self):
        """The state vector of the simulation result"""
        return self._state[:, 0 : self.current_idx]

    def collect_from(self, source):
        """Collect data points from the given source

        The source must be an iterable providing a series system states
        representing the system states at the individual time points"""

        for state in source:
            self.append(state)

    def append(self, system_state):
        """Append an entry to the result vectors.

        Args:
            system_state: The system state to append
        """
        self._append(system_state.time, system_state.inputs, system_state.state)

    def _append(self, time, inputs, state):
        """Append an entry to the result vectors.

        Args:
          time: The time tag for the entry
          inputs: The input vector
          state: The state vector
        """

        if self.current_idx >= self._t.size:
            self.extend_space()
        self._t[self.current_idx] = time
        self._inputs[:, self.current_idx] = inputs
        self._state[:, self.current_idx] = state

        self.current_idx += 1

    def extend_space(self):
        """Extend the storage space for the vectors"""
        self._t = np.r_[self._t, np.empty(RESULT_SIZE_EXTENSION)]
        self._inputs = np.c_[
            self._inputs,
            np.empty((self.system.num_inputs, RESULT_SIZE_EXTENSION)),
        ]
        self._state = np.c_[
            self._state,
            np.empty((self.system.num_states, RESULT_SIZE_EXTENSION)),
        ]

    def get_state_value(self, state: State):
        """Determine the value of the given state in this result object"""

        return self.state[state.state_slice].reshape(state.shape + (-1,))

    def get_input_value(self, signal: InputSignal):
        """Determine the value of the given input in this result object"""

        return self.inputs[signal.input_slice].reshape(signal.shape + (-1,))

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return SystemState(
                system=self.system,
                time=self.time[key],
                state=self.state[:, key],
                inputs=self.inputs[:, key],
            )
        else:
            warnings.warn(
                "The dictionary access interface is deprecated",
                DeprecationWarning,
            )
            if isinstance(key, tuple):
                # In case of a tuple, the first entity is the actual object to
                # access and the remainder is the index into the object
                obj = key[0]
                idx = key[1:]
                value = obj(self)
                if len(idx) > 1:
                    return value[idx]
                return value[idx[0]]
            # Otherwise, the item is an object to access, and we simply defer to
            # the callable interface
            return key(self)

    def __len__(self):
        return self.current_idx


class Simulator:
    """Simulator for dynamic systems.

    The simulator is written with the interface of
    `scipy.integrate.OdeSolver` in mind for the solver, specifically using
    the constructor, the `step` and the `dense_output` functions as well as
    the `status` property of the return value. However, it is possible to
    use other integrators if they honor this interface.

    Args:
        system:
            The system to be simulated
        start_time:
            The start time of the simulation (optional, default=0)
        initial_condition:
            The initial condition (optional, overrides initial condition
            specified in the states)
        event_xtol:
            The absolute tolerance for identifying the time of a
            zero-crossing event.
        event_maxiter:
            The maximum number of iterations for identifying the time of a
            zero-crossing event.
        solver:
            The solver to be used for integrating continuous-time systems.
            The default is the :class:`DOP853 <scipy.integrate.DOP853>`
            solver.
        solver_options:
            Options to be passed to the solver constructor.
    """

    def __init__(
        self,
        system: System,
        start_time=0,
        initial_condition=None,
        max_successive_event_count=1000,
        event_xtol=1.0e-12,
        event_maxiter=1000,
        solver_method=DEFAULT_INTEGRATOR,
        **solver_options
    ):
        """Construct a simulator for the system."""

        # Store the parameters
        self.system = system
        self.max_successive_event_count = max_successive_event_count
        self.event_xtol = event_xtol
        self.event_maxiter = event_maxiter
        self.solver_method = solver_method
        self.solver_options = solver_options

        # Initialize the simulation state
        self.current_time = start_time
        if initial_condition is not None:
            self.current_state = initial_condition
        else:
            self.current_state = self.system.initial_condition
        self.current_inputs = self.system.initial_input

        # Reset the count of successive events
        self.successive_event_count = 0

        # Register event tolerances and directions for easier access
        self.event_tolerances = np.array(
            [event.tolerance for event in self.system.events]
        )
        self.event_directions = np.array(
            [event.direction for event in self.system.events]
        )

        # Check if we have continuous-time states
        self.have_continuous_time_states = any(
            state.derivative_function is not None
            for state in self.system.states
        )

        # Create the clock queue
        self.clock_queue = ClockQueue(
            start_time=start_time, clocks=self.system.clocks
        )

        # The current state is the left-sided limit of the time-dependent state
        # function at the current time. To proceed, we need the right-sided
        # limit, which requires us to apply all pending clock tick handlers.
        self._run_clock_ticks()

    def run_until(self, time_boundary, include_last=True):
        """Run the simulation

        Yields a series of :class:`modypy.model.system.SystemState` objects with
        each element representing one time sample of the process.

        Args:
            time_boundary:
                The end time of the simulation.
            include_last:
                Flag indicating whether the state at the end of simulation shall
                be yielded as well. In case of multiple calls to `run_until`
                this should be set to `False`. Otherwise, the last system state
                provided at the end of one call will be repeated at the
                beginning of the next call.

        Raises:
            SimulationError: if an error occurs during simulation
        """

        if self.have_continuous_time_states:
            yield from self._run_mixed_model_simulation(time_boundary)
        else:
            yield from self._run_discrete_model_simulation(time_boundary)

        if include_last:
            yield SystemState(
                system=self.system,
                time=self.current_time,
                state=self.current_state,
                inputs=self.current_inputs,
            )

    def _run_mixed_model_simulation(self, time_boundary):
        # The outer loop iterates over solver instances as necessary.
        # Events leading to state changes will invalidate the solver, so
        # a new one will have to be created. However, we'll run as long as
        # possible on a single solver to save instantiation time

        # Split events into two partitions:
        # - terminating events
        # - non-terminating events
        # We will handle them separately later.
        terminating_events = [
            event for event in self.system.events if len(event.listeners) > 0
        ]
        non_terminating_events = [
            event for event in self.system.events if len(event.listeners) == 0
        ]
        terminating_detector = _EventDetector(
            system=self.system, events=terminating_events
        )
        non_terminating_detector = _EventDetector(
            system=self.system, events=non_terminating_events
        )

        while self.current_time < time_boundary:
            terminated = False

            # The solver can run to the time boundary or the next clock tick,
            # whichever comes first.
            solver_bound = self.clock_queue.next_clock_tick
            if solver_bound is None or solver_bound > time_boundary:
                solver_bound = time_boundary

            # Create the solver
            solver = self.solver_method(
                fun=self._state_derivative,
                t0=self.current_time,
                y0=self.current_state,
                t_bound=solver_bound,
                vectorized=True,
                **self.solver_options
            )

            # Run the integration until the determined time limit
            while self.current_time < solver_bound and not terminated:
                # Yield the current state (after running the clock ticks)
                yield SystemState(
                    system=self.system,
                    time=self.current_time,
                    inputs=self.current_inputs,
                    state=self.current_state,
                )

                # Perform a solver step
                msg = solver.step()
                if msg is not None:
                    raise IntegrationError(msg)

                # Get interpolation functions for state and inputs
                state_interpolator = solver.dense_output()

                def _input_interpolator(_t):
                    return self.current_inputs

                # Check for occurrence of a terminating event and determine the
                # time of the earliest terminating event.
                first_term = terminating_detector.localize_first_event(
                    start_time=self.current_time,
                    end_time=solver.t,
                    state=state_interpolator,
                    inputs=_input_interpolator,
                )
                search_end_time = solver.t

                # Restrict the search time for non-terminating events
                if first_term is not None:
                    assert first_term[0] >= self.current_time
                    search_end_time = first_term[0]

                # Find non-terminating events
                non_term_occs = non_terminating_detector.localize_events(
                    start_time=self.current_time,
                    end_time=search_end_time,
                    state=state_interpolator,
                    inputs=_input_interpolator,
                )

                # Yield intermediate states for non-terminating events in the
                # order in which they occur
                non_term_occs.sort(key=lambda v: v[0])
                for time, event in non_term_occs:
                    event_state = SystemState(
                        system=self.system,
                        time=time,
                        state=state_interpolator(time),
                        inputs=_input_interpolator(time),
                    )
                    yield event_state

                # In case of a terminating event, advance time to the time of
                # the event and execute its handlers.
                # Otherwise, advance time to the end of the integration step
                # and update the state.
                if first_term is not None:
                    first_term_time, first_term_event = first_term
                    self.current_time = first_term_time
                    self.current_state = state_interpolator(first_term_time)
                    self._run_event_listeners([first_term_event])
                    terminated = True
                else:
                    # No terminating event occurred
                    self.current_time = solver.t
                    self.current_state = solver.y
                    # Reset the successive event counter
                    self.successive_event_count = 0

            # The current state is the left-side limit of the state function
            # over time. However, to properly proceed, we need the right-side
            # limit, so we execute all pending clock ticks now.
            self._run_clock_ticks()

    def _run_discrete_model_simulation(self, time_boundary):
        # For discrete-only systems we only need to run the clocks and advance
        # the time accordingly until we reach the time boundary.
        while self.current_time < time_boundary:
            # Yield the current state
            yield SystemState(
                system=self.system,
                time=self.current_time,
                inputs=self.current_inputs,
                state=self.current_state,
            )

            # Advance time to the next clock tick
            next_clock_tick = self.clock_queue.next_clock_tick
            if next_clock_tick is None or next_clock_tick > time_boundary:
                self.current_time = time_boundary
            else:
                self.current_time = next_clock_tick

            # The current state is the left-side limit of the state function
            # over time. However, to properly proceed, we need the right-side
            # limit, so we execute all pending clock ticks now.
            self._run_clock_ticks()

    def _run_clock_ticks(self):
        """Run all the pending clock ticks."""

        # We collect the clocks to tick here and executed all their listeners
        # later.
        clocks_to_tick = self.clock_queue.tick(self.current_time)

        # Run all the event listeners
        self._run_event_listeners(clocks_to_tick)

    def _run_event_listeners(self, event_sources):
        """Run the event listeners on the given events."""

        while len(event_sources) > 0:
            # Check for excessive counts of successive events
            self.successive_event_count += 1
            if self.successive_event_count > self.max_successive_event_count:
                raise ExcessiveEventError()

            # Prepare the system state for the state updater
            state_updater = _SystemStateUpdater(
                system=self.system,
                time=self.current_time,
                state=self.current_state,
                inputs=self.current_inputs,
            )

            # Determine the values of all event functions before running the
            # event listeners.
            last_event_values = self.system.event_values(state_updater)

            # Collect all listeners associated with the events
            # Note that we run each listener only once, even if it is associated
            # with multiple events
            listeners = set(
                listener
                for event_source in event_sources
                for listener in event_source.listeners
            )

            # Run the event listeners
            # Note that we do not guarantee any specific order of execution
            # here. Listeners thus must be written in such a way that their
            # effects are the same independent of the order in which they are
            # run.
            for listener in listeners:
                listener(state_updater)

            # Update the state
            self.current_state = state_updater.state

            # Determine the value of event functions after running the event
            # listeners
            new_event_values = self.system.event_values(state_updater)

            # Determine which events occurred as a result of the changed state
            event_mask = _find_active_events(
                start_values=last_event_values,
                end_values=new_event_values,
                tolerances=self.event_tolerances,
                directions=self.event_directions,
            )
            if any(event_mask):
                event_sources = [
                    event
                    for event, flag in zip(self.system.events, event_mask)
                    if flag
                ]
            else:
                event_sources = []

    def _state_derivative(self, time, state):
        """The state derivative function used for integrating the state over
        time.

        Args:
          time: The current time
          state: The current state vector

        Returns:
          The time-derivative of the state vector
        """

        system_state = SystemState(system=self.system, time=time, state=state)
        state_derivative = self.system.state_derivative(system_state)
        return state_derivative


class _EventDetector:
    """Helper class for detecting and localizing events"""

    def __init__(self, system, events, xtol=1e-12, maxiter=1000):
        self.system = system
        self.events = events
        self.xtol = xtol
        self.maxiter = maxiter
        self.event_tolerances = np.array([event.tolerance for event in events])
        self.event_directions = np.array([event.direction for event in events])

    def localize_first_event(self, start_time, end_time, state, inputs):
        """Localize the first event occurring in the given time frame.

        Args:
            start_time: The start time of the time frame.
            end_time: The end time of the time_frame.
            state:
                A callable, with `state(t)` being the state vector at time `t`
                for any scalar or one-dimensional array `t` with
                `start_time <= t <= end_time`.
            inputs:
                A callable, with `state(t)` being the input vector at time `t`
                for any scalar or one-dimensional array `t` with
                `start_time <= t <= end_time`.
        Returns:
            A tuple `(time, event)`, giving time and event object for the first
            event having occurred in the given time frame, or `None`, if none of
            the events has occurred in the time frame.
        """
        event_locs = self.localize_events(start_time, end_time, state, inputs)
        if len(event_locs) > 0:
            # At least one of the events has occurred, so we find the first one
            return min(event_locs, key=lambda evt: evt[0])
        return None

    def localize_events(self, start_time, end_time, state, inputs):
        """Localize the all events occurring in the given time frame.

        Args:
            start_time: The start time of the time frame.
            end_time: The end time of the time_frame.
            state:
                A callable, with `state(t)` being the state vector at time `t`
                for any scalar or one-dimensional array `t` with
                `start_time <= t <= end_time`.
            inputs:
                A callable, with `state(t)` being the input vector at time `t`
                for any scalar or one-dimensional array `t` with
                `start_time <= t <= end_time`.
        Returns:
            A list of tuples `(time, event)`, giving time and event object for
            each event having occurred in the given time frame.
        """
        start_state = SystemState(
            system=self.system,
            time=start_time,
            state=state(start_time),
            inputs=inputs(start_time),
        )
        end_state = SystemState(
            system=self.system,
            time=end_time,
            state=state(end_time),
            inputs=inputs(end_time),
        )

        # Determine the list of active events
        active_events = self._get_active_events(start_state, end_state)

        # For each of the active events, localize the zero-crossing
        locations = list()
        for event in active_events:
            event_time = self._find_event_time(
                event, start_time, end_time, state, inputs
            )
            locations.append((event_time, event))
        return locations

    def _get_active_events(self, start_state, end_state):
        """Determine which events are active in the given time frame.

        Args:
            start_state: The state at the beginning of the time frame.
            end_state: The state at the end of the time frame.
        Returns:
            List of events that have occurred in the given time frame.
        """
        start_values = np.array([event(start_state) for event in self.events])
        end_values = np.array([event(end_state) for event in self.events])

        mask = _find_active_events(
            start_values,
            end_values,
            self.event_tolerances,
            self.event_directions,
        )
        return [event for event, flag in zip(self.events, mask) if flag]

    def _find_event_time(self, event, start_time, end_time, state, inputs):
        """
        Find the time when the sign change occurs.

        Args:
            start_time: The start time of the time frame.
            end_time: The end time of the time_frame.
            state:
                A callable, with `state(t)` being the state vector at time `t`
                for any scalar or one-dimensional array `t` with
                `start_time <= t <= end_time`.
            inputs:
                A callable, with `state(t)` being the input vector at time `t`
                for any scalar or one-dimensional array `t` with
                `start_time <= t <= end_time`.
        Returns:
            A time at or after the sign change occurs
        """

        assert start_time <= end_time

        start_state = SystemState(
            system=self.system,
            time=start_time,
            state=state(start_time),
            inputs=inputs(start_time),
        )
        end_state = SystemState(
            system=self.system,
            time=end_time,
            state=state(end_time),
            inputs=inputs(end_time),
        )

        start_value = event(start_state)
        end_value = event(end_state)

        assert (
            (start_value < -event.tolerance) ^ (end_value < -event.tolerance)
        ) | ((event.tolerance < start_value) ^ (event.tolerance < end_value))

        iter_count = 0
        time_diff = end_time - start_time
        while iter_count < self.maxiter and time_diff > self.xtol:
            time_diff /= 2
            mid_time = start_time + time_diff
            mid_state = SystemState(
                system=self.system,
                time=mid_time,
                state=state(mid_time),
                inputs=inputs(mid_time),
            )
            mid_value = event(mid_state)
            mid_value = 0 if np.abs(mid_value) < event.tolerance else mid_value
            if np.sign(mid_value) == np.sign(start_value):
                # The sign change happens after mid_time
                start_time = mid_time
            iter_count += 1
        return start_time


class _SystemStateUpdater(SystemState):
    """A ``_SystemStateUpdater`` is a system state in which the states can be
    updated"""

    def __init__(self, time, system: System, state=None, inputs=None):
        super().__init__(time, system, state, inputs)
        # Make a copy of the state
        self.state = self.state.copy()

    def set_state_value(self, state: State, value):
        """Update the value of the given state"""

        self.state[state.state_slice] = np.asarray(value).ravel()

    def __setitem__(self, key, value):
        warnings.warn(
            "The dictionary access interface is deprecated", DeprecationWarning
        )
        if isinstance(key, tuple):
            # In case the key is a tuple, its first element is the object to
            # access, and the remainder is the index of the element to address
            obj = key[0]
            idx = key[1:]
            if len(idx) > 1:
                obj(self)[idx] = value
            else:
                obj(self)[idx[0]] = value
        else:
            # Otherwise, we'll fall back to the set_value interface
            key.set_value(self, value)


def _find_active_events(start_values, end_values, tolerances, directions):
    """Determine the events for which a matching sign change has occurred
    between the start- and the end-value.

    Returns:
        An array of booleans, indicating for each event whether it has seen a
        sign change or not.
    """

    up = ((start_values < -tolerances) & (end_values >= -tolerances)) | (
        (start_values <= tolerances) & (end_values > tolerances)
    )
    down = ((start_values >= -tolerances) & (end_values < -tolerances)) | (
        (start_values > tolerances) & (end_values <= tolerances)
    )
    mask = (up & (directions >= 0)) | (down & (directions <= 0))
    return mask
