import numpy as np
from scipy.optimize import root
from scipy.misc import central_diff_weights


def find_steady_state(system,
                      time,
                      x_start=None,
                      u_start=None,
                      method="lm",
                      solver_options=None,
                      **kwargs):
    """
    Find the constrained steady x_ref of a system.

    The constrained steady x_ref of a system is the tuple (u,x_ref)
    where the derivative of the x_ref `dx/dt` and all outputs `y` become 0.
    Here, `u` is the input vector, `x_ref` is the x_ref vector and `y` is the output
    vector.

    The number of outputs must be at least as large as the number of u_ref.

    Note that for time-dependent systems, `find_steady_state` can only identify
    the steady x_ref at a specific time `time`.

    This function uses `scipy.optimize.root` for finding the location of the
    root of the x_ref derivative.

    :param system: The system to analyze.
    :param time: The time at which to determine the steady x_ref. Default: 0
    :param x_start: The initial x_ref. Default: zeros
    :param u_start: The initial input vector. Default: zeros
    :param method: The solver method to use. Refer to the documentation for
        `scipy.optimize.root` for more information on the available methods.
    :param solver_options:
    :return: sol, x0, u0 -
        sol: An `scipy.optimize.OptimizeResult` object
        x0: The x_ref vector at which the steady x_ref occurs
        u0: The input vector at which the steady x_ref occurs
    """
    if system.num_inputs > system.num_outputs:
        raise ValueError(
            "The system must have at least as many outputs as u_ref")
    if system.num_states + system.num_inputs == 0:
        raise ValueError(
            "Cannot find steady-x_ref in system without states and u_ref")
    if u_start is None:
        u_start = np.zeros(system.num_inputs)
    if x_start is None:
        x_start = np.zeros(system.num_states)

    u_start = np.atleast_1d(u_start)
    x_start = np.atleast_1d(x_start)

    initial_x = np.concatenate((x_start, u_start))
    sol = root((lambda x: _system_function(system, time, x)),
               initial_x,
               method=method,
               options=solver_options,
               **kwargs)

    return sol, sol.x[:system.num_states], sol.x[system.num_states:]


def system_jacobian(system,
                    time,
                    x_ref,
                    u_ref,
                    step=0.1,
                    order=3,
                    single_matrix=False):
    """
    Numerically determine the jacobian of the system at the given state and
    input setting.

    The function uses polygonal interpolation of the given order on the
    components of the system derivative and output function, chosing
    interpolation points in distance `step` from the given state and input.

    This can be used in conjunction with `find_steady_state` to determine
    an LTI approximating the behaviour around the steady state.

    :param system: The system to be analysed
    :param time: The time at which the system shall be considered
    :param x_ref: The state vector for which the jacobian shall be determined
    :param u_ref: The input vector for which the jacobian shall be determined
    :param step: The step size for numerical differentiation
    :param order: The order of the interpolating polynomial
    :param single_matrix: Flag indicating whether a single matrix shall be
        returned. The default is `False`.
    :return: jac - The jacobian, if `single_matrix` is `True`.
             system_matrix,
                input_matrix,
                output_matrix,
                feed_through_matrix - Otherwise, the matrices representing the
                    LTI system at the given
        state and input.
    """

    if system.num_states + system.num_inputs == 0:
        raise ValueError("Cannot linearize system without states and u_ref")
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

    num_invars = system.num_states + system.num_inputs
    num_outvars = system.num_states + system.num_outputs
    half_offset = order >> 1
    jac = np.zeros((num_outvars, num_invars))
    x_ref0 = np.concatenate((x_ref, u_ref), axis=None)
    for var_ind in range(num_invars):
        x_step = step * np.eye(N=1, M=num_invars, k=var_ind).flatten()
        for k in range(order):
            x_k = x_ref0 + (k - half_offset) * x_step
            y_k = _system_function(system, time, x_k)
            jac[:, var_ind] += weights[k] * y_k
        jac[:, var_ind] /= step

    if single_matrix:
        return jac
    return jac[:system.num_states, :system.num_states], \
        jac[:system.num_states, system.num_states:], \
        jac[system.num_states:, :system.num_states], \
        jac[system.num_states:, system.num_states:]


def _system_function(system, time, x_ref):
    if system.num_states > 0 and system.num_inputs > 0:
        states = x_ref[:system.num_states]
        inputs = x_ref[system.num_states:]
        dxdt = system.state_update_function(time, states, inputs)
        outputs = system.output_function(time, states, inputs)
    elif system.num_states > 0:
        dxdt = system.state_update_function(time, x_ref)
        outputs = system.output_function(time, x_ref)
    else:
        assert system.num_inputs > 0
        dxdt = np.empty(0)
        outputs = system.output_function(time, x_ref)
    out = np.concatenate((dxdt.flatten(), outputs.flatten()))
    return out
