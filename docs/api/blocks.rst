The Blocks Library
==================

The Blocks Library provides re-usable system elements:

Sources (:mod:`modypy.blocks.sources`)
    are blocks and signals that provide possibly time-dependent input data.
    A specific variant is the (:mod:`modypy.blocks.sources.constant`) signal,
    which provides a constant value.

Linear blocks (:mod:`modypy.blocks.linear`)
    provide building-blocks for linear systems, such as the
    :class:`modypy.blocks.linear.LTISystem` class for linear time-invariant
    systems in the typical state-space formulation.

Electro-mechanical blocks (:mod:`modypy.blocks.elmech`)
    are useful for modelling electro-mechanical systems. The main constituent of
    this module currently is the :class:`modypy.blocks.elmech.DCMotor` class for
    modelling DC-motors.

Aerodynamic blocks (:mod:`modypy.blocks.aerodyn`)
    are blocks and signals for aerodynamic modelling. Most prominent is the
    :class:`modypy.blocks.aerodyn.Propeller` class for modelling a propeller in
    static configuration, and the :class:`modpyp.blocks.aerodyn.Thruster` class
    to model a directional thruster with thrust and torque.

Rigid-body motion blocks (:mod:`modypy.blocks.rigid`)
    allow modelling the translational and rotational motion of rigid bodies.

.. toctree::
    :hidden:
    :glob:
    :maxdepth: 4

    packages/blocks/*
