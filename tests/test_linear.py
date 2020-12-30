import pytest
import numpy as np
from simtree.blocks.linear import LTISystem


def test_lti_nonsquare_state_matrix():
    with pytest.raises(ValueError):
        LTISystem(A=[[-1, 0]], B=[[]], C=[[1, 0]], D=[[]])


def test_lti_state_input_mismatch():
    with pytest.raises(ValueError):
        LTISystem(A=[[-1, 0], [0, -1]], B=[[1]], C=[[1, 0]], D=[[1]])


def test_lti_state_output_mismatch():
    with pytest.raises(ValueError):
        LTISystem(A=[[-1, 0], [0, -1]], B=[[1], [1]], C=[[1]], D=[[1]])


def test_lti_output_feed_through_mismatch():
    with pytest.raises(ValueError):
        LTISystem(A=[[-1, 0], [0, -1]], B=[[1], [1]], C=[[1, 1]], D=[[1], [0]])


def test_lti_input_feed_through_mismatch():
    with pytest.raises(ValueError):
        LTISystem(A=[[-1, 0], [0, -1]], B=[[1, 1], [1, 1]], C=[[1, 1]], D=[[1]])


def test_lti_no_states():
    sys = LTISystem(A=np.empty((0, 0)),
                    B=np.empty((0, 1)),
                    C=np.empty((1, 0)),
                    D=1)
    state_update = sys.state_update_function(0, [1])
    assert state_update.size == 0

def test_lti_empty():
    sys = LTISystem(A=np.empty((0, 0)),
                    B=np.empty((0, 0)),
                    C=np.empty((0, 0)),
                    D=np.empty((0, 0)))
    output = sys.output_function(0)
    assert output.size == 0