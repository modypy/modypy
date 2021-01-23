"""
Provide classes for simulation.
"""
import itertools
import heapq

import numpy as np
import scipy.integrate
import scipy.optimize

from modypy.model.system import System
from modypy.model.evaluation import Evaluator, DataProvider, PortProvider

INITIAL_RESULT_SIZE = 16
RESULT_SIZE_EXTENSION = 16

DEFAULT_INTEGRATOR = scipy.integrate.DOP853
DEFAULT_INTEGRATOR_OPTIONS = {
    'rtol': 1.E-12,
    'atol': 1.E-12,
}

DEFAULT_ROOTFINDER = scipy.optimize.brentq
DEFAULT_ROOTFINDER_OPTIONS = {
    'xtol': 1.E-12,
    'maxiter': 1000
}


class SimulationResult:
    """The results provided by a simulation.

    A `SimulationResult` object captures the time series provided by a
    simulation. It has properties `t`, `state` and `output` representing the
    time, state vector and output vector for each individual sample.
    """

    def __init__(self, system: System):
        self.system = system
        self._t = np.empty(INITIAL_RESULT_SIZE)
        self._inputs = np.empty((INITIAL_RESULT_SIZE, self.system.num_inputs))
        self._state = np.empty((INITIAL_RESULT_SIZE, self.system.num_states))
        self._signals = np.empty((INITIAL_RESULT_SIZE, self.system.num_signals))
        self._events = np.empty((INITIAL_RESULT_SIZE, self.system.num_events))
        self._outputs = np.empty((INITIAL_RESULT_SIZE, self.system.num_outputs))

        self.current_idx = 0

    @property
    def time(self):
        """The time vector of the simulation result"""
        return self._t[0:self.current_idx]

    @property
    def inputs(self):
        """The input vector of the simulation result"""
        return self._inputs[0:self.current_idx]

    @property
    def state(self):
        """The state vector of the simulation result"""
        return self._state[0:self.current_idx]

    @property
    def signals(self):
        """The signal vector of the simulation result"""
        return self._signals[0:self.current_idx]

    @property
    def events(self):
        """The event vector of the simulation result"""
        return self._events[0:self.current_idx]

    @property
    def outputs(self):
        """The output vector of the simulation result"""
        return self._outputs[0:self.current_idx]

    def append(self, time, inputs, state, signals, events, outputs):
        """Append an entry to the result vectors.

        Args:
          time: The time tag for the entry
          inputs: The input vector
          state: The state vector
          signals: The signals vector
          events: The events vector
          outputs: The outputs vector
        """

        if self.current_idx >= self._t.size:
            self.extend_space()
        self._t[self.current_idx] = time
        self._inputs[self.current_idx] = inputs
        self._state[self.current_idx] = state
        self._signals[self.current_idx] = signals
        self._events[self.current_idx] = events
        self._outputs[self.current_idx] = outputs

        self.current_idx += 1

    def extend_space(self):
        """Extend the storage space for the vectors"""
        self._t = np.r_[self._t,
                        np.empty(RESULT_SIZE_EXTENSION)]
        self._inputs = np.r_[self._inputs,
                             np.empty((RESULT_SIZE_EXTENSION,
                                       self.system.num_inputs))]
        self._state = np.r_[self._state,
                            np.empty((RESULT_SIZE_EXTENSION,
                                      self.system.num_states))]
        self._signals = np.r_[self._signals,
                              np.empty((RESULT_SIZE_EXTENSION,
                                        self.system.num_signals))]
        self._events = np.r_[self._events,
                             np.empty((RESULT_SIZE_EXTENSION,
                                       self.system.num_events))]
        self._outputs = np.r_[self._outputs,
                              np.empty((RESULT_SIZE_EXTENSION,
                                        self.system.num_outputs))]


class Simulator:
    """Simulator for dynamic systems."""

    def __init__(self,
                 system,
                 start_time,
                 initial_condition=None,
                 integrator_constructor=DEFAULT_INTEGRATOR,
                 integrator_options=None,
                 rootfinder_constructor=DEFAULT_ROOTFINDER,
                 rootfinder_options=None):
        """
        Construct a simulator for the system.

        The simulator is written with the interface of
        `scipy.integrate.OdeSolver` in mind for the integrator, specifically
        using the constructor, the `step` and the `state_trajectory` functions
        as well as the `status` property. However, it is possible to use other
        integrators if they honor this interface.

        Similarly, the rootfinder is expected to comply with the interface of
        `scipy.optimize.brentq`.

        Args:
            system: The system to be simulated
            start_time: The start time of the simulation
            initial_condition: The initial condition (optional)
            integrator_constructor: The constructor function for the
                ODE integrator to be used; optional: if not given,
                ``DEFAULT_INTEGRATOR`` is used.
            integrator_options: The options for ``integrator_constructor``;
                optional: if not given, ``DEFAULT_INTEGRATOR_OPTIONS`` is used.
            rootfinder_constructor: The constructor function for the
                root finder to be used; optional: if not given,
                ``DEFAULT_ROOTFINDER`` is used.
            rootfinder_options: The options for ``rootfinder_constructor``;
                optional: if not given, ``DEFAULT_ROOTFINDER_OPTIONS`` is used
        """

        self.system = system
        self.start_time = start_time

        if initial_condition is not None:
            self.initial_condition = initial_condition
        else:
            self.initial_condition = self.system.initial_condition

        self.integrator_constructor = integrator_constructor
        if integrator_options is None:
            self.integrator_options = DEFAULT_INTEGRATOR_OPTIONS
        else:
            self.integrator_options = integrator_options

        self.rootfinder_constructor = rootfinder_constructor
        if rootfinder_options is None:
            self.rootfinder_options = DEFAULT_ROOTFINDER_OPTIONS
        else:
            self.rootfinder_options = rootfinder_options

        # Collect information about zero-crossing events
        self.event_directions = np.array([event.direction
                                          for event in self.system.events])

        self.result = SimulationResult(system)

        # Set up the state of the system
        self.current_time = self.start_time
        self.current_state = self.initial_condition

        # Start all the clocks
        self.clock_queue = []
        self.start_clocks()

        # Run the first tick of each clock
        self.run_clock_ticks()

        # Determine the initial sample
        evaluator = Evaluator(system=self.system,
                              time=self.current_time,
                              state=self.current_state)
        self.current_inputs = evaluator.inputs
        self.current_signals = evaluator.signals
        self.current_event_values = evaluator.event_values
        self.current_outputs = evaluator.outputs

        # Store the initial sample
        self.result.append(time=self.current_time,
                           inputs=self.current_inputs,
                           state=self.current_state,
                           signals=self.current_signals,
                           events=self.current_event_values,
                           outputs=self.current_outputs)

    def start_clocks(self):
        """Set up all the clock ticks"""

        for clock in self.system.clocks:
            # Start the tick generator at the current time
            tick_generator = clock.tick_generator(self.current_time)
            try:
                first_tick = next(tick_generator)
                entry = TickEntry(first_tick, clock, tick_generator)
                heapq.heappush(self.clock_queue, entry)
            except StopIteration:
                # The block did not produce any ticks at all,
                # so we just ignore it
                pass

    def step(self, t_bound):
        """Execute a single execution step.

        Args:
          t_bound: The maximum time until which the simulation may proceed

        Returns:
          ``None`` if successful, a message string otherwise
        """

        # Check if there is a clock event before the given boundary time.
        # If so, we must not advance beyond that event.
        if len(self.clock_queue) > 0 \
                and self.clock_queue[0].tick_time < t_bound:
            t_bound = self.clock_queue[0].tick_time

        # Integrate for a single step
        integrator = self.integrator_constructor(fun=self.state_derivative,
                                                 t0=self.current_time,
                                                 y0=self.current_state,
                                                 t_bound=t_bound,
                                                 **self.integrator_options)
        message = integrator.step()
        if message is not None:
            return message

        # Handle continuous-time events
        self.handle_continuous_time_events(new_time=integrator.t,
                                           new_state=integrator.y,
                                           state_interpolator=integrator.dense_output())

        # Execute any pending clock ticks
        self.run_clock_ticks()

        # Determine all characteristics of the current sample and store it
        evaluator = Evaluator(system=self.system,
                              time=self.current_time,
                              state=self.current_state)
        self.current_inputs = evaluator.inputs
        self.current_signals = evaluator.signals
        self.current_event_values = evaluator.event_values
        self.current_outputs = evaluator.outputs

        self.result.append(time=self.current_time,
                           inputs=self.current_inputs,
                           state=self.current_state,
                           signals=self.current_signals,
                           events=self.current_event_values,
                           outputs=self.current_outputs)
        return None

    def handle_continuous_time_events(self,
                                      new_time,
                                      new_state,
                                      state_interpolator):
        """
        Determine if any zero-crossing events occurred, and if so, handle
        them.

        Args:
            new_time: The time until which simulation has progressed.
            new_state: The state of the system at the new time, given continuous
                simulation
            state_interpolator: A callable for determining the state at any
                given time between ``self.current_time`` and ``new_time``.
        """

        # Capture the time and event values of the preceding state
        last_time = self.current_time
        last_event_values = self.current_event_values

        # Determine the event values for the new state
        evaluator = Evaluator(system=self.system,
                              time=new_time,
                              state=new_state)
        new_event_values = evaluator.event_values

        # Determine the events for which sign changes occurred and the direction
        # of the change
        sign_changed = np.sign(last_event_values) != np.sign(new_event_values)
        sign_change_direction = np.sign(new_event_values - last_event_values)

        # Determine for which events the conditions are met
        event_occurred = (sign_changed &
                          ((self.event_directions == 0) |
                           (self.event_directions == sign_change_direction)))

        event_indices = np.flatnonzero(event_occurred)
        if len(event_indices) > 0:
            occurred_event = [self.system.events[idx] for idx in event_indices]

            # Identify the first event that occurred
            first_event, first_event_time = \
                self.find_first_event(state_interpolator,
                                      last_time,
                                      new_time,
                                      occurred_event)

            # We will continue immediately after that event
            self.current_time = first_event_time + 1.E-3
            # Get the state at the event time
            self.current_state = state_interpolator(self.current_time)

            # Run the event handlers on the event to update the state
            self.run_event_listeners(first_event)
        else:
            # No event occurred, so we simply accept the integrator end-point as
            # the next sample point.
            self.current_time = new_time
            self.current_state = new_state

    def run_clock_ticks(self):
        """Run all the pending clock ticks."""

        while (len(self.clock_queue) > 0 and
               self.clock_queue[0].tick_time <= self.current_time):
            tick_entry = heapq.heappop(self.clock_queue)
            clock = tick_entry.clock

            # Execute all the listeners on the clock
            self.run_event_listeners(clock)

            try:
                # Get the next tick for the clock
                next_tick_time = next(tick_entry.tick_generator)
                next_tick_entry = TickEntry(next_tick_time,
                                            clock,
                                            tick_entry.tick_generator)
                # Add the clock tick to the queue
                heapq.heappush(self.clock_queue, next_tick_entry)
            except StopIteration:
                # This clock does not deliver any more ticks, so we simply
                # ignore it from now on.
                pass

    def run_event_listeners(self, event_source):
        """Run the event listeners on the given event.
        """

        update_evaluator = Evaluator(system=self.system,
                                     time=self.current_time,
                                     state=self.current_state)
        state_updater = StateUpdater(update_evaluator)
        port_provider = PortProvider(update_evaluator)
        data_provider = DataProvider(time=self.current_time,
                                     states=state_updater,
                                     signals=port_provider)
        for listener in event_source.listeners:
            listener(data_provider)

        # Update the state
        self.current_state = state_updater.new_state

    def find_first_event(self,
                         state_trajectory,
                         start_time,
                         end_time,
                         events_occurred):
        """Determine the event that occurred first.

        Args:
          state_trajectory: A callable that accepts a time in the interval
            given by ``start_time`` and ``end_time`` and provides the state
            vector for that point in time.
          start_time: The lower limit of the time range to be considered.
          end_time: The upper limit of the time range to be considered.
          events_occurred: The list of events that occurred within the
            given time interval.

        Returns: A tuple ``(event, time)``, with ``event`` being the event that
            occurred first and ``time`` being the time at which it occurred.
        """

        # For each event that occurred we determine the exact time that it
        # occurred. For that, we use the the state trajectory provided and
        # determine the time at which the event value becomes zero.
        # We do that for every event and then identify the event that has the
        # minimum time associated with it.
        event_times = np.empty(len(events_occurred))

        for list_index, event in zip(itertools.count(), events_occurred):
            def objective_function(time):
                """Determine the value of the event at different points in
                time.
                """

                intermediate_state = state_trajectory(time)
                intermediate_evaluator = Evaluator(system=self.system,
                                                   time=time,
                                                   state=intermediate_state)
                event_value = intermediate_evaluator.get_event_value(event)
                return event_value

            event_times[list_index] = \
                self.rootfinder_constructor(f=objective_function,
                                            a=start_time,
                                            b=end_time,
                                            **self.rootfinder_options)
        minimum_list_index = np.argmin(event_times)
        first_event = events_occurred[minimum_list_index]
        first_event_time = event_times[minimum_list_index]

        return first_event, first_event_time

    def run_until(self, time_boundary):
        """Run the simulation until the given end time

        Args:
          time_boundary: The end time

        Returns:
          ``None`` if successful, a message string otherwise
        """
        while self.current_time < time_boundary:
            message = self.step(time_boundary)
            if message is not None:
                return message
        return None

    def state_derivative(self, time, state):
        """The state derivative function used for integrating the state over
        time.

        Args:
          time: The current time
          state: The current state vector

        Returns:
          The time-derivative of the state vector
        """

        evaluator = Evaluator(system=self.system, time=time, state=state)
        state_derivative = evaluator.state_derivative
        return state_derivative


class StateUpdater:
    """A ``StateUpdater`` provides access to the states via indexing using the
    ``State`` objects. It also allows the state vector to be updated.
    """
    def __init__(self, evaluator):
        self.new_state = evaluator.state.copy()

    def __setitem__(self, state, value):
        start_index = state.state_index
        end_index = start_index + state.size
        self.new_state[start_index:end_index] = np.asarray(value).flatten()

    def __getitem__(self, state):
        start_index = state.state_index
        end_index = start_index + state.size
        return self.new_state[start_index:end_index].reshape(state.shape)


class TickEntry:
    """A ``TickEntry`` holds information about the next tick of a given clock.
    An order over ``TickEntry`` instances is defined by their time.
    """

    def __init__(self, tick_time, clock, tick_generator):
        self.tick_time = tick_time
        self.clock = clock
        self.tick_generator = tick_generator

    def __lt__(self, other):
        return self.tick_time < other.tick_time
