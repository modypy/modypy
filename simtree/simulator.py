import numpy as np
import scipy.integrate
import scipy.optimize

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
    'maxiter': 1E3
}


class SimulationResult:
    """
    The results provided by a simulation.

    A `SimulationResult` object captures the time series provided by a simulation.
    It has properties `t`, `state` and `output` representing the time, state vector and
    output vector for each individual sample.
    """

    def __init__(self, system):
        self.system = system
        self._t = np.empty(INITIAL_RESULT_SIZE)
        self._state = np.empty((INITIAL_RESULT_SIZE, self.system.num_states))
        self._output = np.empty((INITIAL_RESULT_SIZE, self.system.num_outputs))
        self._events = np.zeros(
            (INITIAL_RESULT_SIZE, self.system.num_events), dtype=bool)

        self.current_idx = 0

    @property
    def t(self):
        return self._t[0:self.current_idx]

    @property
    def state(self):
        return self._state[0:self.current_idx]

    @property
    def output(self):
        return self._output[0:self.current_idx]

    @property
    def events(self):
        return self._events[0:self.current_idx]

    def append(self, t, state, output, event=None):
        """
        Append a sample to the result.
        """

        if self.current_idx >= self._t.size:
            self.extend_space()
        self._t[self.current_idx] = t
        self._state[self.current_idx] = state
        self._output[self.current_idx] = output
        if event is not None:
            self._events[self.current_idx, event] = True

        self.current_idx += 1

    def extend_space(self):
        self._t = np.r_[self._t,      np.empty(RESULT_SIZE_EXTENSION)]
        self._state = np.r_[self._state,  np.empty(
            (RESULT_SIZE_EXTENSION, self.system.num_states))]
        self._output = np.r_[self._output, np.empty(
            (RESULT_SIZE_EXTENSION, self.system.num_outputs))]
        self._events = np.r_[self._events, np.zeros(
            (RESULT_SIZE_EXTENSION, self.system.num_events), dtype=bool)]


class Simulator:
    """
    Simulator for dynamic systems.

    Dynamic systems to be simulated using this class need to support a set of functions:

    system.state_update_function(t,state,outputs)
       Determine the derivative of the state vector of the system, given
       the time `t`, state `state` and output vector `outputs`.

    system.output_function(t,state)
       Determine the value of the outputs of the system given time `t` and
       state `state`.

    system.initial_condition
       The initial value of the state vector.
    """

    def __init__(self,
                 system,
                 t0, tbound,
                 initial_condition=None,
                 input_callable=None,
                 integrator_constructor=DEFAULT_INTEGRATOR,
                 integrator_options=DEFAULT_INTEGRATOR_OPTIONS,
                 rootfinder_constructor=DEFAULT_ROOTFINDER,
                 rootfinder_options=DEFAULT_ROOTFINDER_OPTIONS):
        """
        Construct a simulator for a system.

        system
            The system to be simulated. This can be the result of a compilation
            using `simtree.compiler.Compiler`.
        t0: number
            The start time of the simulation.
        tbound: number
            The end time of the simulation. This also limits the maximum time
            until which stepping is possible.
        initial_condition: list-like of numbers, optional
            The initial condition of the system state. If not set, the initial
            condition specified in the system is used.
        input_callable: callable, optional
            The callable used to provide the inputs for the system.
            Must accept time as the single argument.
            If not given, inputs are assumed to be zero.
        integrator_constructor: callable, optional
            The constructor to be used to instantiate the integrator. If not
            given, `DEFAULT_INTEGRATOR` is used.
        integrator_options: dictionary, optional
            Additional parameters to be passed to the integrator constructor. If
            not given, `DEFAULT_INTEGRATOR_OPTIONS` is used.
        rootfinder_constructor: callable, optional
            The constructor to be used to instantiate the rootfinder. If not
            given, `DEFAULT_ROOTFINDER` is used.
        rootfinder_options: dictionary, optional
            Additional parameters to be passed to the rootfinder constructor. If
            not given, `DEFAULT_ROOTFINDER_OPTIONS` is used.

        The simulator is written with the interface of
        `scipy.integrate.OdeSolver` in mind for the integrator, specifically
        using the constructor, the `step` and the `dense_output` functions as
        well as the `status` property. However, it is possible to use other
        integrators if they honor this interface.

        Similarly, the rootfinder is expected to comply with the interface of
        `scipy.optimize.brentq`.
        """

        self.system = system
        self.tbound = tbound
        self.input_callable = input_callable
        self.integrator_constructor = integrator_constructor
        self.integrator_options = integrator_options
        self.rootfinder_constructor = rootfinder_constructor
        self.rootfinder_options = rootfinder_options

        self.result = SimulationResult(system)

        # Set up the integrator
        if initial_condition is None:
            initial_condition = self.system.initial_condition

        self.integrator = \
            self.integrator_constructor(self.state_derivative_function,
                                        t0,
                                        initial_condition,
                                        tbound,
                                        **self.integrator_options)

        # Store the initial state
        self.result.append(self.t, self.state, self.output)

    @property
    def t(self):
        """The current simulation time."""

        return self.integrator.t

    @property
    def state(self):
        """The current state of the simulated system."""

        return self.integrator.y

    @property
    def inputs(self):
        """The current input vector for the simulated system."""

        return self.input_function(self.t)

    @property
    def signals(self):
        """The current signals in the simulated system."""

        return self.signal_function(self.t, self.state)

    @property
    def output(self):
        """The current outputs of the simulated system."""

        return self.output_function(self.t, self.state)

    @property
    def event_values(self):
        """The current outputs of the event functions."""

        return self.event_function(self.t, self.state)

    @property
    def status(self):
        """The current status of the integrator."""

        return self.integrator.status

    @property
    def running(self):
        """Boolean indicating whether the simulation is still running, i.e. has
        not been finished or aborted."""

        return self.integrator.status == "running"

    def input_function(self, t):
        """Determine the inputs for the system at the given time."""

        if self.system.num_inputs > 0:
            if self.input_callable is None:
                return np.zeros(self.system.num_inputs)
            return self.input_callable(t)
        return np.empty(0)

    def state_derivative_function(self, t, state):
        """Combined state derivative function used for the integrator."""

        if self.system.num_inputs > 0:
            inputs = self.input_function(t)
            return self.system.state_update_function(t, state, inputs)
        return self.system.state_update_function(t, state)

    def signal_function(self, t, states=None):
        """Combined signal vector for the system"""
        inputs = self.input_function(t)
        if self.system.num_inputs > 0 and self.system.num_states > 0:
            return self.system.signal_function(t, states, inputs)
        if self.system.num_inputs > 0:
            return self.system.signal_function(t, inputs)
        if self.system.num_states > 0:
            return self.system.signal_function(t, states)
        return self.system.signal_function(t)

    def output_function(self, t, states=None):
        """Combined output vector for the system"""
        inputs = self.input_function(t)
        if self.system.num_inputs > 0 and self.system.num_states > 0:
            return self.system.output_function(t, states, inputs)
        if self.system.num_inputs > 0:
            return self.system.output_function(t, inputs)
        if self.system.num_states > 0:
            return self.system.output_function(t, states)
        return self.system.output_function(t)

    def event_function(self, t, states=None):
        """Combined event vector for the system"""
        inputs = self.input_function(t)
        if self.system.num_inputs > 0 and self.system.num_states > 0:
            return self.system.event_function(t, states, inputs)
        if self.system.num_inputs > 0:
            return self.system.event_function(t, inputs)
        if self.system.num_states > 0:
            return self.system.event_function(t, states)
        return self.system.event_function(t)

    def step(self):
        """Execute a single simulation step."""

        # Save the last event values
        old_event_values = self.event_values
        last_t = self.t
        message = self.integrator.step()
        if message is not None:
            # The last integration step failed
            return message

        # Check for changes in event functions
        new_event_values = self.event_values
        old_event_signs = np.sign(old_event_values)
        new_event_signs = np.sign(new_event_values)
        events_occurred = np.flatnonzero((old_event_signs != new_event_signs))

        if len(events_occurred) > 0:
            # At least one of the event functions has changed its sign, so there
            # was at least one event. We need to identify the first event that
            # occurred. To do that, we find the time of occurrence for each of
            # the events using the dense output of the integrator and the root finder.
            interpolator = self.integrator.dense_output()

            # Function to interpolate the event function across the last integration step
            def event_interpolator(t):
                state = interpolator(t)
                return self.event_function(t, state)

            # Go through all the events and find their exact time of occurrence
            tcross = []
            for eventidx in events_occurred:
                tc = scipy.optimize.brentq(f=(lambda t: event_interpolator(t)[eventidx]),
                                           a=last_t,
                                           b=self.t)
                assert last_t <= tc <= self.t
                tcross.append((eventidx, tc))

            # Sort the events by increasing time
            tcross.sort(key=(lambda entry: entry[1]))

            # Process only the first event.
            # We determine the state at the time of the event using the interpolator
            # and the outputs using the system output function.
            event_idx, event_t = tcross[0]
            event_state = interpolator(event_t)
            event_outputs = self.output_function(event_t, event_state)

            # Add the event to the result collection
            self.result.append(event_t, event_state, event_outputs, event_idx)

            # We continue right of the event in order to avoid finding the same
            # event in the next step again.
            # FIXME: We might want to try to find a time where the event value is safely on the other side
            next_t = event_t+self.rootfinder_options["xtol"]/2
            next_state = interpolator(next_t)
            next_inputs = self.input_function(next_t)

            # Let the system handle the event by updating the state.
            if self.system.num_inputs>0:
                new_state = \
                    self.system.update_state_function(next_t, next_state, next_inputs)
            else:
                new_state = \
                    self.system.update_state_function(next_t, next_state)

            # We need to reset the integrator.
            # Ideally, we would want to just reset the time and the state, but
            # proper behaviour of the integrator in this case is not guaranteed,
            # so we just create a new one.
            self.integrator = \
                self.integrator_constructor(self.state_derivative_function,
                                            next_t,
                                            new_state,
                                            self.tbound,
                                            **self.integrator_options)
        else:
            # No events to handle
            # Add the current status to the result collection
            self.result.append(self.t, self.state, self.output)
        return None

    def run(self):
        """Simulate the system until the end time of the simulation."""

        while self.running:
            message = self.step()
            if message is not None:
                return message
        return None
