import itertools

import numpy as np
import scipy.integrate
import scipy.optimize

from simtree.model.system import System
from simtree.model.evaluator import Evaluator, DataProvider, StateUpdater, PortProvider

INITIAL_RESULT_SIZE = 16
RESULT_SIZE_EXTENSION = 16

DEFAULT_INTEGRATOR = scipy.integrate.DOP853
DEFAULT_INTEGRATOR_OPTIONS = {
    'rtol': 1.E-12,
    'atol': 1.E-15,
}

DEFAULT_ROOTFINDER = scipy.optimize.brentq
DEFAULT_ROOTFINDER_OPTIONS = {
    'xtol': 1.E-12,
    'maxiter': 1000
}


class SimulationResult:
    """
    The results provided by a simulation.

    A `SimulationResult` object captures the time series provided by a simulation.
    It has properties `t`, `state` and `output` representing the time, state vector and
    output vector for each individual sample.
    """

    def __init__(self, system: System):
        self.system = system
        self._t = np.empty(INITIAL_RESULT_SIZE)
        self._state = np.empty((INITIAL_RESULT_SIZE, self.system.num_states))
        self._signals = np.empty((INITIAL_RESULT_SIZE, self.system.num_signals))
        self._events = np.empty((INITIAL_RESULT_SIZE, self.system.num_events))

        self.current_idx = 0

    @property
    def time(self):
        return self._t[0:self.current_idx]

    @property
    def state(self):
        return self._state[0:self.current_idx]

    @property
    def signals(self):
        return self._signals[0:self.current_idx]

    @property
    def events(self):
        return self._events[0:self.current_idx]

    def append(self, time, state, signals, events):
        if self.current_idx >= self._t.size:
            self.extend_space()
        self._t[self.current_idx] = time
        self._state[self.current_idx] = state
        self._signals[self.current_idx] = signals
        self._events[self.current_idx] = events

        self.current_idx += 1

    def extend_space(self):
        self._t = np.r_[self._t,      np.empty(RESULT_SIZE_EXTENSION)]
        self._state = np.r_[self._state,  np.empty(
            (RESULT_SIZE_EXTENSION, self.system.num_states))]
        self._signals = np.r_[self._signals, np.empty(
            (RESULT_SIZE_EXTENSION, self.system.num_signals))]
        self._events = np.r_[self._events, np.empty(
            (RESULT_SIZE_EXTENSION, self.system.num_events))]


class Simulator:
    """
    Simulator for dynamic systems.
    """

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
        using the constructor, the `step` and the `state_trajectory` functions as
        well as the `status` property. However, it is possible to use other
        integrators if they honor this interface.

        Similarly, the rootfinder is expected to comply with the interface of
        `scipy.optimize.brentq`.

        :param system: The system to be simulated
        :param start_time: The start time of the simulation
        :param initial_condition: The initial condition (optional)
        :param integrator_constructor: The constructor function for the
            ODE integrator to be used; optional: if not given,
            ``DEFAULT_INTEGRATOR`` is used.
        :param integrator_options: The options for ``integrator_constructor``;
            optional: if not given, ``DEFAULT_INTEGRATOR_OPTIONS`` is used.
        :param rootfinder_constructor: The constructor function for the
            root finder to be used; optional: if not given,
            ``DEFAULT_ROOTFINDER`` is used.
        :param rootfinder_options: The options for ``rootfinder_constructor``;
            optional: if not given, ``DEFAULT_ROOTFINDER_OPTIONS`` is used
        """

        self.system = system
        self.start_time = start_time

        if initial_condition is not None:
            self.initial_condition = initial_condition
        else:
            self.initial_condition = self.system.initial_condition

        self.integrator_constructor = integrator_constructor
        self.integrator_options = integrator_options or DEFAULT_INTEGRATOR_OPTIONS

        self.rootfinder_constructor = rootfinder_constructor
        self.rootfinder_options = rootfinder_options or DEFAULT_ROOTFINDER_OPTIONS

        self.current_time = self.start_time
        self.current_state = self.initial_condition

        self.result = SimulationResult(system)

        evaluator = Evaluator(system=self.system,
                              time=self.current_time,
                              state=self.current_state)
        self.current_signals = evaluator.signals
        self.current_event_values = evaluator.event_values

        # Store the initial state
        self.result.append(time=self.current_time,
                           state=self.current_state,
                           signals=self.current_signals,
                           events=self.current_event_values)

    def step(self, t_bound=None):
        """
        Execute a single execution step.

        :param t_bound: The maximum time until which the simulation may proceed.
        """

        last_time = self.current_time
        last_event_values = self.current_event_values

        integrator = self.integrator_constructor(fun=self.state_derivative,
                                                 t0=self.current_time,
                                                 y0=self.current_state,
                                                 t_bound=t_bound,
                                                 **self.integrator_options)
        message = integrator.step()
        if message is not None:
            return message

        evaluator = Evaluator(system=self.system,
                              time=integrator.t,
                              state=integrator.y)

        # Check for events
        event_indices = np.flatnonzero(np.sign(last_event_values) !=
                                       np.sign(evaluator.event_values))
        if len(event_indices) > 0:
            events_occurred = [self.system.events[idx] for idx in event_indices]
            state_interpolator = integrator.dense_output()
            start_time = last_time
            end_time = integrator.t

            first_event, first_event_time = \
                self.find_first_event(state_interpolator,
                                      start_time,
                                      end_time,
                                      events_occurred)

            self.current_time = first_event_time + 1.E-3
            self.current_state = state_interpolator(self.current_time)

            if first_event.update_function is not None:
                # Call the event handler function to update the state
                update_evaluator = Evaluator(system=self.system,
                                             time=self.current_time,
                                             state=self.current_state)
                state_updater = StateUpdater(update_evaluator)
                port_provider = PortProvider(update_evaluator)
                data = DataProvider(time=self.current_time,
                                    states=state_updater,
                                    inputs=port_provider)
                first_event.update_function(data)
                self.current_state = state_updater.new_state

            evaluator = Evaluator(system=self.system,
                                  time=self.current_time,
                                  state=self.current_state)
            self.current_signals = evaluator.signals
            self.current_event_values = evaluator.event_values

            self.result.append(time=self.current_time,
                               state=self.current_state,
                               signals=self.current_signals,
                               events=self.current_event_values)
            return None

        # No event occurred, so we simply accept the integrator end-point as the
        # next sample point.
        self.current_time = integrator.t
        self.current_state = integrator.y
        self.current_signals = evaluator.signals
        self.current_event_values = evaluator.event_values

        self.result.append(time=self.current_time,
                           state=self.current_state,
                           signals=self.current_signals,
                           events=self.current_event_values)
        return None

    def find_first_event(self, state_trajectory, start_time, end_time, events_occurred):
        """
        Determine the event that occurred first.

        :param state_trajectory: A callable that accepts a time in the interval
            given by ``start_time`` and ``end_time`` and provides the state
            vector for that point in time.
        :param start_time: The lower limit of the time range to be considered.
        :param end_time: The upper limit of the time range to be considered.
        :param events_occurred: The list of events that occurred within the
            given time interval.
        :return: A tuple `(event, time)`, giving the event that occurred first
            and the time at which it occurred.
        """

        # For each event that occurred we determine the exact time that it
        # occurred. For that, we use the the state trajectory provided and
        # determine the time at which the event function becomes zero.
        # We do that for every event and then identify the event that has the
        # minimum time associated with it.
        event_times = np.empty(len(events_occurred))

        for list_index, event in zip(itertools.count(), events_occurred):
            def objective_function(time):
                """
                Determine the value of the event at different points in
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

    def run_until(self, t_bound):
        while self.current_time < t_bound:
            message = self.step(t_bound)
            if message is not None:
                return message
        return None

    def state_derivative(self, time, state):
        evaluator = Evaluator(system=self.system, time=time, state=state)
        state_derivative = evaluator.state_derivative
        return state_derivative
