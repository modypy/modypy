"""
Provides the ``Evaluator`` class, which can be used to evaluate the individual
aspects (signals, state derivatives, ...) at any given point in time.
"""
import warnings
from typing import Union, Callable

import numpy as np

from .events import EventPort
from .system import System
from .ports import Port, PortNotConnectedError, InputSignal
from .states import State


class AlgebraicLoopError(RuntimeError):
    """Exception raised when an algebraic loop is detected on evaluation"""


class Evaluator:
    """This class allows to evaluate the individual aspects (signals, state
    derivatives, ...) of a system at any given time."""

    def __init__(self, time, system: System, state=None, inputs=None):
        self.time = time
        self.system = system

        if state is None:
            state = system.initial_condition.copy()
        self._state = state
        if inputs is None:
            inputs = system.initial_input.copy()
        self._inputs = inputs

    @property
    def state(self):
        """The current state"""
        return self._state

    @property
    def state_derivative(self):
        """The state derivative vector for the complete system"""
        state_derivative = np.empty(self.system.num_states)
        for state_instance in self.system.states:
            state_derivative[state_instance.state_slice] = \
                self.get_state_derivative(state_instance).flatten()
        return state_derivative

    @property
    def inputs(self):
        """The input vector for the complete system"""
        return self._inputs

    @property
    def signals(self):
        """The signal vector for the complete system."""
        signal_vector = np.empty(self.system.num_signals)
        for signal_instance in self.system.signals:
            signal_vector[signal_instance.signal_slice] = \
                self.get_port_value(signal_instance).flatten()
        return signal_vector

    @property
    def outputs(self):
        """The output vector for the complete system"""
        output_vector = np.empty(self.system.num_outputs)
        for port in self.system.outputs:
            port_value = port(self)
            output_vector[port.output_slice] = port_value.flatten()
        return output_vector

    @property
    def event_values(self):
        """The event vector for the complete system"""
        event_vector = np.empty(self.system.num_events)
        for event_instance in self.system.events:
            event_vector[event_instance.event_index] = \
                self.get_event_value(event_instance)
        return event_vector

    def get_state_value(self, state: State):
        """Determine the value of a given state.

        Args:
          state: The state

        Returns:
          The value of the state
        """
        return self._state[state.state_slice].reshape(state.shape)

    def get_input_value(self, signal: InputSignal):
        """Determine the value of a given input signal.

        Args:
            signal: The input signal

        Returns:
            The value of the input signal
        """
        return self._inputs[signal.input_slice].reshape(signal.shape)

    def get_port_value(self, port: Port):
        """Determine the value of the given port.

        If the value has not yet been calculated, it will be calculated before
        this method returns. If an algebraic loop is encountered during
        calculation, an ``AlgebraicLoopError`` will be raised.

        Args:
          port: The port for which the value shall be determined

        Returns:
          The value of the port

        Raises:
          AlgebraicLoopError: if an algebraic loop is encountered while
            evaluating the value of the signal
          PortNotConnectedError: if the port is not connected to a signal
        """

        if port.size == 0:
            # An empty port has no value
            return np.empty(port.shape)

        signal = port.signal

        if signal is None:
            # This port is not connected to any signal, so we cannot determine
            # its value.
            raise PortNotConnectedError()

        try:
            signal_value = self._inputs[signal.input_slice]
        except AttributeError:
            # Perform evaluation
            data = DataProvider(evaluator=self,
                                time=self.time)
            if callable(signal.value):
                signal_value = signal.value(data)
            else:
                signal_value = signal.value

        # Ensure that the signal has the correct shape
        signal_value = np.asarray(signal_value).reshape(signal.shape)

        # Return the value of the signal
        return signal_value

    def get_state_derivative(self, state):
        """Get the state derivative of the given state.

        Args:
          state: The state for which the derivative shall be determined

        Returns:
          The state derivative

        Raises:
          AlgebraicLoopError: if an algebraic loop is encountered while
            evaluating the derivative of the state instance

        """
        if state.derivative_function is not None:
            data = DataProvider(evaluator=self,
                                time=self.time)
            state_derivative = state.derivative_function(data)
            state_derivative = \
                np.asarray(state_derivative).reshape(state.shape)
        else:
            state_derivative = np.zeros(state.shape)
        return state_derivative

    def get_event_value(self, event):
        """Get the value of the event function of the given event

        Args:
          event: The event for which to calculate the value

        Returns:
          The value of the event function

        Raises:
          AlgebraicLoopError: if an algebraic loop is encountered while
            evaluating the value of the event function
        """

        data = DataProvider(evaluator=self,
                            time=self.time)
        event_value = event.event_function(data)
        return event_value


class DataProvider:
    """A ``DataProvider`` provides access to the data about the current point in
    time in the simulation. It has the following properties:

    ``time``
        The current time
    ``states``
        The contents of the current states, accessed by indexing using the
        ``State`` objects.
    ``signals``
        The contents of the current signals, accessed by indexing using the
        ``Port`` objects.
    """

    def __init__(self, evaluator, time):
        self.evaluator = evaluator
        self.time = time

    def get_state_value(self, state: State):
        return self.evaluator.get_state_value(state)

    def get_event_value(self, event: EventPort):
        return self.evaluator.get_event_value(event)

    def get_input_value(self, input_signal: InputSignal):
        return self.evaluator.get_input_value(input_signal)

    @property
    def states(self):
        """Old way of accessing the states dictionary. Deprecated."""
        warnings.warn(DeprecationWarning("The ``states`` property of the "
                                         "``DataProvider`` class is deprecated "
                                         "and will be removed in the future. "
                                         "Use direct indexing instead."))
        return self

    @property
    def signals(self):
        """Old way of accessing the signals dictionary. Deprecated."""
        warnings.warn(DeprecationWarning("The ``signals`` property of the "
                                         "``DataProvider`` class is deprecated "
                                         "and will be removed in the future. "
                                         "Use direct indexing instead."))
        return self

    @property
    def inputs(self):
        """Old way of accessing the signals dictionary. Deprecated."""
        warnings.warn(DeprecationWarning("The ``inputs`` property of the "
                                         "``DataProvider`` class is deprecated "
                                         "and will be removed in the future. "
                                         "Use direct indexing instead."))
        return self
