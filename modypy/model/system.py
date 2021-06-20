"""
Provides classes for constructing systems and block hierarchies.
"""
import warnings
from typing import List

import numpy as np

from modypy.model.states import State


class System:
    """A system is a composition of states, signals and events."""

    def __init__(self):
        self.num_states = 0
        self.states: List[State] = list()

        self.num_inputs = 0
        self.inputs = list()

        self.events = list()

        self.clocks = set()

    @property
    def system(self):
        """The system itself"""
        return self

    @property
    def initial_condition(self):
        """The initial condition vector for the state of this system"""
        initial_condition = np.zeros(self.num_states)
        for state in self.states:
            initial_condition[
                state.state_slice
            ] = state.initial_condition.ravel()
        return initial_condition

    @property
    def initial_input(self):
        """The initial inputs of this system"""
        initial_inputs = np.zeros(self.num_inputs)
        for signal in self.inputs:
            initial_inputs[signal.input_slice] = np.ravel(signal.value)
        return initial_inputs

    def allocate_state_lines(self, count):
        """Allocate a sequence of consecutive state lines.

        Args:
          count: The number of state lines to allocate

        Returns:
          The index of the first state line allocated

        """
        start_index = self.num_states
        self.num_states += count
        return start_index

    def allocate_input_lines(self, count):
        """Allocate a sequence of consecutive input lines.

        Args:
          count: The number of input lines to allocate

        Returns:
          The index of the first input line allocated

        """
        start_index = self.num_inputs
        self.num_inputs += count
        return start_index

    def register_event(self, event):
        """Register an event.

        Args:
          event: The event to register

        Returns:
          The index of the event line allocated for the event

        """
        event_index = len(self.events)
        self.events.append(event)
        return event_index

    def register_clock(self, clock):
        """Register a clock.

        Args:
          clock: The clock to register.

        Returns:

        """
        self.clocks.add(clock)

    @property
    def num_events(self):
        """The number of events registered with this system"""
        return len(self.events)

    def event_values(self, system_state):
        """Determine the value of all event functions for the given system
        state.

        Args:
            system_state: The state for which to determine the event values.

        Returns:
            The vector containing the value of all event functions for the
            given system state.
        """
        event_vector = np.empty(self.num_events)
        for event_instance in self.events:
            event_vector[event_instance.event_index] = event_instance(
                system_state
            )
        return event_vector

    def state_derivative(self, system_state):
        """Determine the value of all state derivative functions for the given
        system state.

        Args:
            system_state: The state for which to determine the event values.

        Returns:
            The vector of state derivatives for this system.
        """
        state_derivative = np.zeros(self.num_states)
        for state_instance in self.states:
            if state_instance.derivative_function is not None:
                state_derivative[state_instance.state_slice] = np.ravel(
                    state_instance.derivative_function(system_state)
                )
        return state_derivative


class Block:
    """A block is a re-usable building block for systems."""

    def __init__(self, parent):
        self.parent = parent
        self.system = self.parent.system


class SystemState:
    """This class allows to evaluate the individual aspects (signals, state
    derivatives, ...) of a system at any given time."""

    def __init__(self, time, system: System, state=None, inputs=None):
        self.time = time
        self.system = system

        if state is None:
            state = system.initial_condition.copy()
        self.state = state
        if inputs is None:
            inputs = system.initial_input.copy()
        self.inputs = inputs

    def get_state_value(self, state: State):
        """Determine the value of a given state.

        Args:
          state: The state

        Returns:
          The value of the state
        """
        if np.isscalar(self.time):
            return self.state[state.state_slice].reshape(state.shape)
        else:
            return self.state[state.state_slice].reshape(state.shape + (-1,))

    def get_input_value(self, signal):
        """Determine the value of a given input signal.

        Args:
            signal: The input signal

        Returns:
            The value of the input signal
        """
        if np.isscalar(self.time):
            return self.inputs[signal.input_slice].reshape(signal.shape)
        else:
            return self.inputs[signal.input_slice].reshape(signal.shape + (-1,))

    def __getitem__(self, key):
        warnings.warn(
            "The dictionary access interface is deprecated", DeprecationWarning
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
