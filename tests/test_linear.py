import pytest;
from simutree.blocks.linear import LTISystem;

def test_lti_nonsquare_state_matrix():
   with pytest.raises(ValueError):
      LTISystem(A=[[-1,0]],B=[[]],C=[[1,0]],D=[[]]);

def test_lti_state_input_mismatch():
   with pytest.raises(ValueError):
      LTISystem(A=[[-1,0],[0,-1]],B=[[1]],C=[[1,0]],D=[[1]]);

def test_lti_state_output_mismatch():
   with pytest.raises(ValueError):
      LTISystem(A=[[-1,0],[0,-1]],B=[[1],[1]],C=[[1]],D=[[1]]);

def test_lti_output_feedthrough_mismatch():
   with pytest.raises(ValueError):
      LTISystem(A=[[-1,0],[0,-1]],B=[[1],[1]],C=[[1,1]],D=[[1],[0]]);
