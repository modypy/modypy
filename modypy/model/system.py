"""
Provides classes for constructing systems and block hierarchies.
"""
from typing import List, Set

import numpy as np

from modypy.model.states import State
from modypy.model.events import ZeroCrossEventSource, Clock
from modypy.model.ports import InputSignal, OutputPort, Signal


class System:
    """A system is a composition of states, signals and events."""
    def __init__(self):
        self.num_signals = 0
        self.signals: List[Signal] = list()

        self.num_states = 0
        self.states: List[State] = list()

        self.events: List[ZeroCrossEventSource] = list()

        self.num_inputs = 0
        self.inputs: List[InputSignal] = list()

        self.num_outputs = 0
        self.outputs: List[OutputPort] = list()

        self.clocks: Set[Clock] = set()

    @property
    def system(self):
        """The system itself"""
        return self

    @property
    def initial_condition(self):
        """The initial condition vector for the state of this system"""
        initial_condition = np.zeros(self.num_states)
        for state in self.states:
            start_index = state.state_index
            end_index = start_index + state.size
            initial_condition[start_index:end_index] = \
                state.initial_condition.flatten()
        return initial_condition

    @property
    def initial_input(self):
        """The initial inputs of this system"""
        initial_inputs = np.zeros(self.num_inputs)
        for signal in self.inputs:
            initial_inputs[signal.input_slice] = signal.value.flatten()
        return initial_inputs

    def allocate_signal_lines(self, count):
        """Allocate a sequence of consecutive signal lines.

        Args:
          count: The number of signal lines to allocate

        Returns:
          The index of the first signal line allocated

        """
        start_index = self.num_signals
        self.num_signals += count
        return start_index

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

    def allocate_output_lines(self, count):
        """Allocate a sequence of consecutive input lines.

        Args:
          count: The number of input lines to allocate

        Returns:
          The index of the first input line allocated

        """
        start_index = self.num_outputs
        self.num_outputs += count
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
            event_vector[event_instance.event_index] = \
                event_instance(system_state)
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
                state_derivative[state_instance.state_slice] = \
                    np.ravel(state_instance.derivative_function(system_state))
        return state_derivative


class Block:
    """A block is a re-usable building block for systems."""
    def __init__(self, parent):
        self.parent = parent
        self.system = self.parent.system
