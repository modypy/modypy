"""
Provides the ``Evaluator`` class, which can be used to evaluate the individual
aspects (signals, state derivatives, ...) at any given point in time.
"""
import warnings

import numpy as np

from .system import System
from .ports import InputSignal
from .states import State


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
