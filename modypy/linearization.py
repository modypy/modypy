"""
Provides functions to determine the jacobi matrix for linearizing the system
around a given state with specified inputs.
"""
from typing import List

import numpy as np
from scipy.misc import central_diff_weights

from modypy.model import Port, System, SystemState


class LinearizationConfiguration:
    """
    Represents the configuration for the determination of the system jacobian


    Attributes:
        system
            The system for which the jacobian shall be determined
        time
            The system time for which the jacobian shall be determined
            (default: 0)
        state
            The state around which the jacobian shall be determined (default: 0)
        inputs
            The input values around which the jacobian shall be determined
            (default: 0)
        outputs
            List of :class:`OutputDescriptor` instances for the signals to be
            considered as outputs (default: representations of all output ports
            in the system)
        num_outputs
            The sum of sizes of all signals to be considered as outputs
        default_step_size
            The default step size to use for the numerical differentiation
            (default: 0.1)
        interpolation_order
            The interpolation order to use for numerical differentiation
            (default: 3)

    """

    def __init__(self,
                 system: System,
                 time=0,
                 state=None,
                 inputs=None):
        self.system = system
        self.time = time

        if state is None:
            self.state = np.zeros(self.system.num_states)
        else:
            self.state = state

        if inputs is None:
            self.inputs = np.zeros(self.system.num_inputs)
        else:
            self.inputs = inputs

        self.outputs: List[OutputDescriptor] = list()
        self.num_outputs = 0

        self.default_step_size = 0.1
        self.interpolation_order = 3


class OutputDescriptor:
    """
    Represents information about an output signal for the determination of the
    system jacobian

    Attributes:
        config
            The :class:`LinearizationConfiguration` instance to which this
            output descriptor belongs
        port
            The :class:`port <modypy.model.Port>` object that is used as output
        output_index
            The index of the first line allocated to this signal in the
            output and feed-through matrices
    """

    def __init__(self, config: LinearizationConfiguration, port: Port):
        self.config = config
        self.port = port

        # Assign an output index
        self.output_index = self.config.num_outputs
        self.config.num_outputs += self.port.size
        self.config.outputs.append(self)

    @property
    def output_slice(self):
        """A slice representing the index range of this output"""

        return slice(self.output_index, self.port.size)


def system_jacobian(config: LinearizationConfiguration,
                    single_matrix=False):
    """Numerically determine the jacobian of the system at the given state and
    input setting.

    The function uses polygonal interpolation of the given order on the
    components of the system derivative and output function, choosing
    interpolation points at a given distance from the given state and input
    values.

    This can be used in conjunction with `find_steady_state` to determine
    an LTI approximating the behaviour around the steady state.

    NOTE: This function currently does not honor clocks.

    Args:
        config
            The :class:`LinearizationConfiguration` instance describing the
            linearization to be performed
        single_matrix
            Flag indicating whether a single matrix shall be returned. The
            default is `False`.

    Returns:
        the jacobian, if `single_matrix` is `True`, and a tuple of
        system matrix, input matrix, output matrix and feed-through matrix, if
        `single_matrix` is `False`.
    Raises:
        ValueError if the system does not have states or inputs
    """

    if config.system.num_states + config.system.num_inputs == 0:
        raise ValueError("Cannot linearize system without states and inputs")

    num_invars = config.system.num_states + config.system.num_inputs
    num_outvars = config.system.num_states + config.num_outputs
    half_offset = config.interpolation_order >> 1
    weights = _get_central_diff_weights(config.interpolation_order)

    jac = np.zeros((num_outvars, num_invars))
    x_ref0 = np.concatenate((config.state, config.inputs), axis=None)

    for var_ind in range(num_invars):
        x_step = config.default_step_size * np.eye(N=1,
                                                   M=num_invars,
                                                   k=var_ind).flatten()
        for k in range(config.interpolation_order):
            x_k = x_ref0 + (k - half_offset) * x_step
            y_k = _system_function(config, x_k)
            jac[:, var_ind] += weights[k] * y_k
        jac[:, var_ind] /= config.default_step_size

    if single_matrix:
        return jac
    return jac[:config.system.num_states, :config.system.num_states], \
           jac[:config.system.num_states, config.system.num_states:], \
           jac[config.system.num_states:, :config.system.num_states], \
           jac[config.system.num_states:, config.system.num_states:]


def _get_central_diff_weights(order):
    """Determine the weights for central differentiation"""

    if order == 3:
        weights = np.array([-1, 0, 1]) / 2.0
    elif order == 5:
        weights = np.array([1, -8, 0, 8, -1]) / 12.0
    elif order == 7:
        weights = np.array([-1, 9, -45, 0, 45, -9, 1]) / 60.0
    elif order == 9:
        weights = np.array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840.0
    else:
        weights = central_diff_weights(order, 1)
    return weights


def _system_function(config: LinearizationConfiguration, x_ref):
    """
    Determine the value of the vector of state derivatives and outputs
    given the vector of states and inputs

    Args:
        config
            The :class:`LinearizationConfiguration` instance describing the
            linearization to be performed
        x_ref
            The vector of states and inputs

    Returns:
        The vector of state derivatives and outputs
    """
    state = x_ref[:config.system.num_states]
    inputs = x_ref[config.system.num_states:]

    system_state = SystemState(time=config.time,
                               system=config.system,
                               state=state,
                               inputs=inputs)

    outputs = np.zeros(config.num_outputs)
    for output in config.outputs:
        outputs[output.output_index:output.output_index + output.port.size] = \
            np.ravel(output.port(system_state))

    return np.concatenate((config.system.state_derivative(system_state), outputs))
