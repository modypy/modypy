from simtree.blocks import LeafBlock;
import numpy as np;

"""
Implementation of a linear, time-invariant system of the format

  dx/dt = A*x+B*u
  y = C*x+D*u
  
The matrices A,B,C and D define the state and output behaviour of the sytstem.
"""
class LTISystem(LeafBlock):
   def __init__(self,A,B,C,D,**kwargs):
      A=np.asarray(A);
      B=np.asarray(B);
      C=np.asarray(C);
      D=np.asarray(D);
      LeafBlock.__init__(self,
                         num_inputs=B.shape[1],
                         num_states=A.shape[0],
                         num_outputs=C.shape[0],
                         **kwargs);
      if A.shape[0]!=A.shape[1]:
         raise ValueError("The state update matrix A must be square");
      if A.shape[0]!=B.shape[0]:
         raise ValueError("The height of the state update matrix A and the input matrix B must be the same");
      if A.shape[1]!=C.shape[1]:
         raise ValueError("The width of the state update matrix A and the state output matrix C must be the same");
      if C.shape[0]!=D.shape[0]:
         raise ValueError("The height of the state output matrix C and the feedthrough matrix D must be the same");
      self.A = A;
      self.B = B;
      self.C = C;
      self.D = D;
   
   def state_update_function(self,t,states,inputs=None):
      if self.num_inputs>0:
         return np.matmul(self.A,states)+np.matmul(self.B,inputs);
      else:
         return np.matmul(self.A,states);
   
   def output_function(self,t,states,inputs=None):
      if self.num_inputs>0:
         return np.matmul(self.C,states)+np.matmul(self.D,inputs);
      else:
         return np.matmul(self.C,states);
