"""
Functions and classes for finding the steady state of a system.

To determine a steady state, set up a :class:`SteadyStateConfiguration` object
and pass it to :func:`find_steady_state`.
"""
from collections.abc import Mapping
from functools import partial
from itertools import accumulate
from typing import Union

import numpy as np
import scipy.optimize as opt

from modypy.model import SystemState, Port, System, State, InputSignal


class SteadyStateConfiguration:
    """Represents the configuration for the steady state determination

    Attributes:
        system
            The system for which a steady state analysis shall be carried out
        time
            The system time for which the steady state analysis shall be carried
            out.
        objective
            An objective to minimize. This is either `None` (default), a
            callable or a `Port`. If no objective is specified, any steady state
            that satisfies the other constraints may be returned.
        states
            A read-only dictionary mapping states to :class:`StateConstraint`
            instances for the respective state. The state constraint can be
            configured by modifying its properties.
        ports
            A read-only dictionary mapping ports to :class:`PortConstraint`
            instances for the respective port. The port constraint can be
            configured by modifying its properties.
        inputs
            A read-only dictionary mapping ports to :class:`InputConstraint`
            instances for the respective port. The port constraint can be
            configured by modifying its properties.
        initial_condition
            The initial estimate for the states. The default is the initial
            condition of the system.
        initial_input
            The initial estimate for the inputs. The default is the initial
            values specified for the inputs in the system.
        input_bounds
            An array of shape (n,2), where n is the number of inputs for the
            system. Each entry ``input_bounds[k]`` is a tuple ``(lb, ub)``, with
            ``lb`` giving the lower and ``ub`` giving the upper bound for the
            respective input line. The initial value for all bounds is
            ``(-np.inf, np.inf)``, representing an unconstrained input line.
        state_bounds
            An array of shape (n,2), where n is the number of states for the
            system. The format is the same as for `input_bounds`.
        steady_states
            An array of shape (n,), where n is the number of states for the
            system. The entry ``steady_states[k]`` is a boolean indicating
            whether the state ``k`` shall be steady, i.e. whether its derivative
            shall be zero. By default, all states are set to be steady.
        solver_options
            Dictionary with additional keyword options for the solver.
    """

    def __init__(self,
                 system: System,
                 time: float = 0,
                 objective: Union[callable, Port] = None):
        self.system = system
        self.time = time

        self.objective = objective

        # Set up the initial state estimates
        self.initial_condition = self.system.initial_condition
        # Set up the initial state bounds
        self.state_bounds = np.full(shape=(self.system.num_states, 2),
                                    fill_value=(-np.inf, np.inf))
        # Set up the set of steady states
        self.steady_states = np.full(shape=self.system.num_states,
                                     fill_value=True)

        # Set up the initial input estimates
        self.initial_input = self.system.initial_input
        # Set up the initial input bounds
        self.input_bounds = np.full(shape=(self.system.num_inputs, 2),
                                    fill_value=(-np.inf, np.inf))

        # Set up the dictionary for solver options
        self.solver_options = dict()

        # Set up the dictionaries for the specific constraints
        self.ports = _ConstraintDictionary(PortConstraint, self)
        self.states = _ConstraintDictionary(StateConstraint, self)
        self.inputs = _ConstraintDictionary(InputConstraint, self)


class PortConstraint(opt.NonlinearConstraint):
    """A ``PortConstraint`` represents constraints on a single port.

    Properties:
        port
            The port to be constrained
        lower_bounds
            A numerical value or an array representing the lower limit for
            the port. The default is negative infinity, i.e., no lower limit.
        upper_bounds
            A numerical value or an array representing the upper limit for
            the port. The default is positive infinity, i.e., no upper limit.
    """

    def __init__(self,
                 config: SteadyStateConfiguration,
                 port: Port,
                 lower_limit=-np.inf,
                 upper_limit=np.inf):
        self.config = config
        self.port = port

        super().__init__(fun=self._evaluate,
                         lb=np.ravel(np.full(self.port.shape, lower_limit)),
                         ub=np.ravel(np.full(self.port.shape, upper_limit)))

    @property
    def lower_bounds(self):
        """Return the lower bounds currently set for this port"""
        return self.lb.reshape(self.port.shape)

    @lower_bounds.setter
    def lower_bounds(self, value):
        self.lb[:] = np.ravel(value)

    @property
    def upper_bounds(self):
        return self.ub.reshape(self.port.shape)

    @upper_bounds.setter
    def upper_bounds(self, value):
        self.ub[:] = np.ravel(value)

    def _evaluate(self, x):
        """Calculate the value vector of the port"""

        state = x[:self.config.system.num_states]
        inputs = x[self.config.system.num_states:]
        system_state = SystemState(time=self.config.time,
                                   system=self.config.system,
                                   state=state,
                                   inputs=inputs)
        return np.ravel(self.port(system_state))


class StateConstraint:
    """A ``StateConstraint`` object represents the constraints on a state.

    Properties:
        lower_bounds:
            A matrix with the shape of the state, representing the lower bound
            for each component of the state (default: -inf)
        upper_bounds:
            A matrix with the shape of the state, representing the upper bound
            for each component of the state (default: +inf)
        steady_state:
            A matrix of booleans with the shape of the state, indicating whether
            the respective component of the state shall be a steady-state, i.e.,
            whether its derivative shall be constrained to zero (default: True)
        initial_condition:
            A matrix with the shape of the state, representing the initial
            steady state guess for each component of the state
            (default: initial condition of the state)"""
    def __init__(self,
                 config: SteadyStateConfiguration,
                 state: State):
        self.config = config
        self.state = state

        flat_state_bounds = self.config.state_bounds[self.state.state_slice]
        self._state_bounds = flat_state_bounds.reshape(self.state.shape+(2,))

        flat_steady_states = self.config.steady_states[self.state.state_slice]
        self._steady_states = flat_steady_states.reshape(self.state.shape)

        flat_initial_condition = \
            self.config.initial_condition[self.state.state_slice]
        self._initial_condition = \
            flat_initial_condition.reshape(self.state.shape)

    @property
    def lower_bounds(self):
        return self._state_bounds[:, 0]

    @lower_bounds.setter
    def lower_bounds(self, value):
        self._state_bounds[:, 0] = value

    @property
    def upper_bounds(self):
        return self._state_bounds[:, 1]

    @upper_bounds.setter
    def upper_bounds(self, value):
        self._state_bounds[:, 1] = value

    @property
    def steady_state(self):
        return self._steady_states

    @steady_state.setter
    def steady_state(self, value):
        self.steady_state[:] = value

    @property
    def initial_condition(self):
        return self._initial_condition

    @initial_condition.setter
    def initial_condition(self, value):
        self.initial_condition[:] = value


class InputConstraint:
    """A ``InputConstraint`` object represents the constraints on an input port.

    Properties:
        lower_bounds:
            A matrix with the shape of the input, representing the lower bound
            for each component of the port (default: -inf)
        upper_bounds:
            A matrix with the shape of the input, representing the upper bound
            for each component of the port (default: +inf)
        initial_guess:
            A matrix with the shape of the input, representing the initial
            steady state guess for each component of the input
            (default: value of the input at the time of creation of the
            :class:`SteadyStateConfiguration` object)"""
    def __init__(self,
                 config: SteadyStateConfiguration,
                 input_signal: InputSignal):
        self.config = config
        self.input_signal = input_signal

        flat_input_bounds = \
            self.config.input_bounds[self.input_signal.input_slice]
        self._input_bounds = \
            flat_input_bounds.reshape(self.input_signal.shape+(2,))

        flat_initial_guess = \
            self.config.initial_input[self.input_signal.input_slice]
        self._initial_guess = \
            flat_initial_guess.reshape(self.input_signal.shape)

    @property
    def lower_bounds(self):
        return self._input_bounds[:, 0]

    @lower_bounds.setter
    def lower_bounds(self, value):
        self._input_bounds[:, 0] = value

    @property
    def upper_bounds(self):
        return self._input_bounds[:, 1]

    @upper_bounds.setter
    def upper_bounds(self, value):
        self._input_bounds[:, 1] = value

    @property
    def initial_guess(self):
        return self._initial_guess

    @initial_guess.setter
    def initial_guess(self, value):
        self.initial_guess[:] = value


def find_steady_state(config: SteadyStateConfiguration):
    """Run the steady-state determination

    Args:
        config: The configuration for the steady-state analysis

    Returns:
        An :class:`OptimizeResult <scipy.optimize.OptimizeResult>` object with
        additional fields.

        state: ndarray
            The state part of the solution
        inputs: ndarray
            The input part of the solution
        system_state: modypy.model.evaluation.SystemState
            An :class:`SystemState <modypy.model.evaluation.SystemState>`
            object, configured to evaluate the system at the determined
            steady-state
    """

    # Set up the initial estimate
    x0 = np.concatenate((config.initial_condition, config.initial_input))
    # Set up the bounds
    bounds = np.concatenate((config.state_bounds, config.input_bounds))

    # Set up the constraints
    constraints = list()

    constraints += config.ports.values()

    if config.objective is not None:
        # We have an actual objective function, so we can use the steady-state
        # constraint as actual constraint.
        if any(config.steady_states):
            # Set up the state derivative constraint
            steady_state_constraint = _StateDerivativeConstraint(config)
            constraints.append(steady_state_constraint)

        # Translate the objective function
        if callable(config.objective):
            objective_function = partial(_general_objective_function, config)
        else:
            raise ValueError("The objective function must be either a Port or "
                             "a callable")
    elif any(config.steady_states):
        # No objective function was specified, but we can use the steady-state
        # constraint function. The value of this function is intended to be
        # zero, so the minimum value of its square is zero.
        steady_state_constraint = _StateDerivativeConstraint(config)
        constraints.append(steady_state_constraint)
        objective_function = steady_state_constraint.evaluate_squared
    else:
        # We have neither an objective function to minimize nor do we have any
        # state that is intended to be steady. We cannot use the signal
        # constraints to minimize, as these may specify ranges instead of a
        # target value. We cannot do anything about this.
        raise ValueError("Either an objective function or at least one steady "
                         "state is required")

    result = opt.minimize(fun=objective_function,
                          x0=x0,
                          method="trust-constr",
                          bounds=bounds,
                          constraints=constraints,
                          options=config.solver_options)

    result.config = config
    result.state = result.x[:config.system.num_states]
    result.inputs = result.x[config.system.num_states:]
    result.system_state = SystemState(time=config.time,
                                      system=config.system,
                                      state=result.state,
                                      inputs=result.inputs)

    return result


class _StateDerivativeConstraint(opt.NonlinearConstraint):
    """Represents the steady-state constraints on the state derivatives"""

    def __init__(self, config: SteadyStateConfiguration):
        self.config = config
        # Steady-state constraints can be defined on the level of individual
        # state components. To optimize evaluation, we only evaluate derivatives
        # of those states that have at least one of their components
        # constrained.
        self.constrained_states = [
            state for state in self.config.system.states
            if any(self.config.steady_states[state.state_slice])]

        # We will build a vector of the constrained derivatives, and for that
        # we assign offsets for the states in that vector
        self.state_offsets = \
            [0] + list(accumulate(state.size
                                  for state in self.constrained_states))

        num_states = self.state_offsets[-1]

        # Now we set up the bounds for each of these
        ub = np.full(num_states, np.inf)
        ub[self.config.steady_states] = 0
        lb = -ub

        opt.NonlinearConstraint.__init__(self,
                                         fun=self.evaluate,
                                         lb=lb,
                                         ub=ub)

    def evaluate(self, x):
        """Determine the value of the derivatives of the vector of constrained
        states"""

        state = x[:self.config.system.num_states]
        inputs = x[self.config.system.num_states:]
        system_state = SystemState(time=self.config.time,
                                   system=self.config.system,
                                   state=state,
                                   inputs=inputs)
        derivative_vector = self.config.system.state_derivative(system_state)
        return derivative_vector

    def evaluate_squared(self, x):
        """Determine the 2-norm of the derivatives vector of constrained
        states"""

        return np.sum(np.square(self.evaluate(x)))


def _general_objective_function(config: SteadyStateConfiguration, x):
    """Implementation of the general objective function

    This calls the objective function with a `DataProvider` as single parameter.

    Args:
        config: The configuration for the steady-state determination
        x: The vector of the current values of states and input

    Returns:
        The current value of the objective function
    """

    state = x[:config.system.num_states]
    inputs = x[config.system.num_states:]
    system_state = SystemState(time=config.time,
                               system=config.system,
                               state=state,
                               inputs=inputs)
    return config.objective(system_state)


class _ConstraintDictionary(Mapping):
    """Dictionary to hold constraints

    When a key is requested for which there is no entry yet, the given
    constructor is called with the args, the key and the keyword args to
    create a new entry."""

    def __init__(self, constructor, *args, **kwargs):
        self.data = dict()
        self.constructor = constructor
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, key):
        try:
            return self.data[key]
        except KeyError:
            new_item = self.constructor(*self.args, key, **self.kwargs)
            self.data[key] = new_item
            return new_item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)
