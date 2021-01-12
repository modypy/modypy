import pytest
import numpy as np
import numpy.testing as npt

from modypy.blocks.linear import LTISystem, Gain
from modypy.blocks.sources import constant
from modypy.model import System
from modypy.model.evaluation import Evaluator


def test_lti_nonsquare_state_matrix():
    system = System()
    with pytest.raises(ValueError):
        LTISystem(parent=system,
                  system_matrix=[[-1, 0]],
                  input_matrix=[[]],
                  output_matrix=[[1, 0]],
                  feed_through_matrix=[[]])


def test_lti_state_input_mismatch():
    system = System()
    with pytest.raises(ValueError):
        LTISystem(parent=system,
                  system_matrix=[[-1, 0],
                                 [0, -1]],
                  input_matrix=[[1]],
                  output_matrix=[[1, 0]],
                  feed_through_matrix=[[1]])


def test_lti_state_output_mismatch():
    system = System()
    with pytest.raises(ValueError):
        LTISystem(parent=system,
                  system_matrix=[[-1, 0],
                                 [0, -1]],
                  input_matrix=[[1], [1]],
                  output_matrix=[[1]],
                  feed_through_matrix=[[1]])


def test_lti_output_feed_through_mismatch():
    system = System()
    with pytest.raises(ValueError):
        LTISystem(parent=system,
                  system_matrix=[[-1, 0],
                                 [0, -1]],
                  input_matrix=[[1], [1]],
                  output_matrix=[[1, 1]],
                  feed_through_matrix=[[1], [0]])


def test_lti_input_feed_through_mismatch():
    system = System()
    with pytest.raises(ValueError):
        LTISystem(parent=system,
                  system_matrix=[[-1, 0],
                                 [0, -1]],
                  input_matrix=[[1, 1],
                                [1, 1]],
                  output_matrix=[[1, 1]],
                  feed_through_matrix=[[1]])


def test_lti_no_states():
    system = System()
    lti = LTISystem(parent=system,
                    system_matrix=np.empty((0, 0)),
                    input_matrix=np.empty((0, 1)),
                    output_matrix=np.empty((1, 0)),
                    feed_through_matrix=1)
    source = constant(system, value=1)
    source.connect(lti.input)

    evaluator = Evaluator(time=0, system=system)
    state_derivative = evaluator.get_state_derivative(lti.state)
    assert state_derivative.size == 0


def test_lti_empty():
    system = System()
    lti = LTISystem(parent=system,
                    system_matrix=np.empty((0, 0)),
                    input_matrix=np.empty((0, 0)),
                    output_matrix=np.empty((0, 0)),
                    feed_through_matrix=np.empty((0, 0)))

    evaluator = Evaluator(time=0, system=system)
    output = evaluator.get_port_value(lti.output)
    assert output.size == 0


def test_gain():
    system = System()
    gain = Gain(system, k=[[1, 2], [3, 4]])
    gain_in = constant(system, value=[3, 4])
    gain.input.connect(gain_in)

    evaluator = Evaluator(time=0, system=system)
    npt.assert_almost_equal(evaluator.get_port_value(gain.output),
                            [11, 25])
