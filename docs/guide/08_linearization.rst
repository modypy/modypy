Linearizing the Water Tank
==========================

In the previous exercise we said that steady states can be used to linearize
systems around given operating points. In this exercise we will do just that
with our tank example.

We reuse the code from the previous exercise. However, now we also need the
function :func:`system_jacobian <modypy.linearization.system_jacobian>`, so
we will extend our imports:

.. code-block:: python

    from modypy.linearization import find_steady_state, system_jacobian

At the end of our example code, we add a call to the `system_jacobian`-function:

.. code-block:: python

    # Find the system jacobian at the steady state
    jac_A, jac_B, jac_C, jac_D = system_jacobian(system,
                                                 time=0,
                                                 x_ref=steady_state,
                                                 u_ref=steady_inputs,
                                                 single_matrix=False)
    print("Linearization at steady-state point:")
    print("A=%s" % jac_A)
    print("B=%s" % jac_B)
    print("C=%s" % jac_C)
    print("D=%s" % jac_D)

Running the example again, we get this output:

.. code-block::

    Target height: 5.000000
    Steady state height: 5.000000
    Steady state inflow: 19.809089
    Theoretical state state inflow: 19.809089
    Linearization at steady-state point:
    A=[[-0.0990504]]
    B=[[0.05]]
    C=[[1.]]
    D=[[0.]]

The matrices :math:`A`, :math:`B`, :math:`C` and :math:`D` represent a linear
dynamical system where the state is :math:`\delta h\left(t\right)` and the input
is :math:`\delta v_1\left(t\right)`. These represent the difference between the
actual height and inflow velocity and those at the steady-state point:

.. math::
    \delta h\left(t\right) &= h\left(t\right) - h_0 \\
    \delta v_1\left(t\right) &= v_1\left(t\right) - v_{1,0}

As we can see from :math:`A`, the time derivative of the height is negatively
correlated with the height itself, i.e. the higher the height, the faster the
water flows out.
The matrix :math:`B` tell us that we can increase the derivative of the height
by increasing the inflow.

The matrices :math:`C` and :math:`D` describe the relationship of our state and
input signals to the value of the output ports.
Clearly, our output is the height difference, as we defined it, and the inflow
velocity does not directly influence it.