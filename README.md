[![Build Status](https://travis-ci.com/ralfgerlich/simtree.svg?branch=master)](https://travis-ci.com/ralfgerlich/simtree)
[![codecov](https://codecov.io/gh/ralfgerlich/simtree/branch/master/graph/badge.svg)](https://codecov.io/gh/ralfgerlich/simtree)

SimuTree is a Python package for hierarchical modelling and simulation of dynamical systems.

It was inspired by [simupy](https://github.com/simupy/simupy) developed by Ben Margolis.
The architecture of SimuTree differs in that it was explicitly designed to support hierarchical modelling, were models can be built from both atomic and composite blocks.

SimuTree features a model compiler that flattens the hierarchy, resolves hierarchical connections and determines a suitable execution order of the blocks based on their interdependencies.

The model compiler can be used to establish a simulation model or for code generation.
