import pytest;
from simtree.blocks import LeafBlock, NonLeafBlock;
from simtree.blocks.linear import LTISystem;
from simtree.blocks.sources import Constant;
import math;
import numpy as np;

class BouncingBall(LeafBlock):
   def __init__(self,g=-9.81,gamma=0.3,**kwargs):
      LeafBlock.__init__(self,num_states=4,num_outputs=2,**kwargs);
      self.g = g;
      self.gamma = gamma;
   
   def output_function(self,t,state):
      return [state[1]];
   
   def state_update_function(self,t,state):
      return [state[2],state[3],
              -self.gamma*state[2],self.g-self.gamma*state[3]];

class StaticPropeller(LeafBlock):
   def __init__(self,ct,cp,D,**kwargs):
      LeafBlock.__init__(self,num_inputs=2,num_outputs=3,**kwargs);
      self.cp = cp;
      self.ct = ct;
      self.D = D;
   
   def output_function(self,t,inputs):
      return [self.ct*inputs[1]*self.D**4*inputs[0]**2,
              self.cp*inputs[1]*self.D**5*inputs[0]**3,
              self.cp/(2*math.pi)*inputs[1]*self.D**5*inputs[0]**2];

class DCMotor(LTISystem):
   def __init__(self,Kv,R,L,J,**kwargs):
      LTISystem.__init__(self,
                         A=[[0,Kv/J],[-Kv/L,-R/L]],
                         B=[[0,-1/J],[1/L,0]],
                         C=[[1/(2*math.pi),0],[0,1]],
                         D=[[0,0],[0,0]],
                         feedthrough_inputs=[],
                         **kwargs);

class DCMotorModel:
   def __init__(self):
      self.static_propeller = StaticPropeller(0.09,0.04,8*25.E-3,name="static_propeller");
      self.dcmotor = DCMotor(789.E-6,43.3E-3,1.9E-3,5.284E-6,name="dcmotor");

      self.engine = NonLeafBlock(name="engine",
                                 children=[self.dcmotor,self.static_propeller],
                                 num_inputs=2,
                                 num_outputs=2);
      
      self.engine.connect(self.dcmotor,0,self.static_propeller,0);
      self.engine.connect(self.static_propeller,1,self.dcmotor,1);
      
      self.engine.connect_input(0,self.dcmotor,0);
      self.engine.connect_input(1,self.static_propeller,1);
      
      self.engine.connect_output(self.static_propeller,0,0);
      self.engine.connect_output(self.dcmotor,1,1);
      
      self.voltage = Constant(0.4*11.1,name="voltage");
      self.density = Constant(1.29,name="rho");
      
      self.system = NonLeafBlock(name="self.system",
                            children=[self.voltage,self.density,self.engine]);
      self.system.connect(self.voltage,0,self.engine,0);
      self.system.connect(self.density,0,self.engine,1);
      
      self.leaf_blocks = set([self.dcmotor, self.static_propeller, self.voltage, self.density]);

class ExponentialDecay(LTISystem):
   def __init__(self,T,**kwargs):
      LTISystem.__init__(self,A=[[-1/T]],B=np.zeros((1,0)),C=[[1]],D=np.zeros((1,0)),**kwargs);

class ExponentialDecayNoState(LeafBlock):
   def __init__(self,x0,T,**kwargs):
      LeafBlock.__init__(self,num_outputs=1,**kwargs);
      self.x0 = x0;
      self.T = T;
   
   def output_function(self,t):
      return [math.exp(-t/self.T)*self.x0];

@pytest.fixture
def decay_model():
   return ExponentialDecay(T=1.,initial_condition=[10.0],name="decay");

@pytest.fixture
def decay_model_no_state():
   exp_decay_no_state = ExponentialDecayNoState(T=1.,x0=10.0,name="decay_no_state");
   exp_decay_state = ExponentialDecay(T=1.,initial_condition=[10.0],name="decay_state");
   sys = NonLeafBlock(children=[exp_decay_no_state,exp_decay_state]);
   return sys;
   
@pytest.fixture
def dcmotor_model():
   return DCMotorModel();

@pytest.fixture
def bouncing_ball_model():
   return BouncingBall(initial_condition=[0,10.0,0,0]);