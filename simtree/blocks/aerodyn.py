"""
A collection of blocks useful for aerodynamics.
"""
import math

import numpy as np

from simtree.blocks import LeafBlock


class Propeller(LeafBlock):
    """
    A block representing a propeller.

    The block has two inputs:
        - The speed of the propeller n, and
        - the density of the air rho.

    It provides three outputs:
        - The thrust vector F,
        - the braking torque tau,
        - the total braking power P.

    The magnitudes of thrust, torque and power are determined by the following
    formulae:

        F = ct(n)*rho*D^5*n^2
        tau = cp(n)/(2*pi)*rho*D^5*n^2
        P = cp(n)*rho*D^5*n^3

    Here,
        - ct(n) is the thrust coefficient at speed n,
        - cp(n) is the power coefficient at speed n, and
        - D is the diameter.
    """

    def __init__(self,
                 thrust_coeff,
                 power_coeff,
                 diameter,
                 **kwargs):
        LeafBlock.__init__(self, num_inputs=2, num_outputs=3, **kwargs)
        if not callable(thrust_coeff):
            thrust_coeff_value = thrust_coeff
            thrust_coeff = (lambda n: thrust_coeff_value)
        if not callable(power_coeff):
            power_coeff_value = power_coeff
            power_coeff = (lambda n: power_coeff_value)

        self.thrust_coeff = thrust_coeff
        self.power_coeff = power_coeff
        self.diameter = diameter

    def output_function(self, t, inputs):
        n = inputs[0]
        rho = inputs[1]

        thrust = self.thrust_coeff(n) * rho * self.diameter**4 * n**2
        torque = self.power_coeff(n)/(2*math.pi) * rho * self.diameter**5 * n**2
        power = self.power_coeff(n) * rho * self.diameter**5 * n**3

        return np.c_[thrust, torque, power]


class Thruster(LeafBlock):
    """
    A block representing a thruster.

    A thruster converts scalar thrust and torque to vectorial thrust and torque.
    The thrust is considered to work in a (constant) thrust direction, determined
    by the thrust axis.

    Torque is combined thrust along the thrust axis and torque due to the thrust
    working at the end of a (constant) arm relative to the center of gravity
    (CoG) of a rigid body.

    The block has two inputs:
        - The scalar thrust in the direction of the thrust axis, and
        - the scalar torque along the thrust axis.

    The block has six outputs:
        - The thrust vector (3 components), and
        - the torque vector (3 components).

    It is configured by
        - the thrust axis (vector X,Y,Z),
        - the thrust arm (vector X,Y,Z), and
        - the turning direction (scalar: +1 or -1).

    The right-hand rule applied to the thrust axis gives the positive direction
    of the thruster. Torque acts in the *opposite* direction, i.e. if the
    thruster turns clockwise, torque acts counter-clockwise.
    """
    def __init__(self,
                 vector,
                 arm,
                 direction=1,
                 **kwargs):
        LeafBlock.__init__(self,
                           num_inputs=2,
                           num_outputs=6,
                           **kwargs)
        self.vector = vector
        self.arm = arm
        self.direction = direction

    def output_function(self, t, inputs):
        del t  # unused

        thrust = inputs[0]
        torque = inputs[1]

        thrust_vector = self.vector * thrust
        torque_vector = self.direction * self.vector * torque + \
            np.cross(self.arm, thrust_vector)

        return np.concatenate(
            (thrust_vector, torque_vector),
            axis=None
        )
