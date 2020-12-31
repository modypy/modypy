import pytest
import numpy as np
from simtree.blocks.linear import LTISystem


def test_lti_nonsquare_state_matrix():
    with pytest.raises(ValueError):
        LTISystem(system_matrix=[[-1, 0]],
                  input_matrix=[[]],
                  output_matrix=[[1, 0]],
                  feed_through_matrix=[[]])


def test_lti_state_input_mismatch():
    with pytest.raises(ValueError):
        LTISystem(system_matrix=[[-1, 0],
                                 [0, -1]],
                  input_matrix=[[1]],
                  output_matrix=[[1, 0]],
                  feed_through_matrix=[[1]])


def test_lti_state_output_mismatch():
    with pytest.raises(ValueError):
        LTISystem(system_matrix=[[-1, 0],
                                 [0, -1]],
                  input_matrix=[[1], [1]],
                  output_matrix=[[1]],
                  feed_through_matrix=[[1]])


def test_lti_output_feed_through_mismatch():
    with pytest.raises(ValueError):
        LTISystem(system_matrix=[[-1, 0],
                                 [0, -1]],
                  input_matrix=[[1], [1]],
                  output_matrix=[[1, 1]],
                  feed_through_matrix=[[1], [0]])


def test_lti_input_feed_through_mismatch():
    with pytest.raises(ValueError):
        LTISystem(system_matrix=[[-1, 0],
                                 [0, -1]],
                  input_matrix=[[1, 1],
                                [1, 1]],
                  output_matrix=[[1, 1]],
                  feed_through_matrix=[[1]])


def test_lti_no_states():
    sys = LTISystem(system_matrix=np.empty((0, 0)),
                    input_matrix=np.empty((0, 1)),
                    output_matrix=np.empty((1, 0)),
                    feed_through_matrix=1)
    state_update = sys.state_update_function(0, [1])
    assert state_update.size == 0

def test_lti_empty():
    sys = LTISystem(system_matrix=np.empty((0, 0)),
                    input_matrix=np.empty((0, 0)),
                    output_matrix=np.empty((0, 0)),
                    feed_through_matrix=np.empty((0, 0)))
    output = sys.output_function(0)
    assert output.size == 0
