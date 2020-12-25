import pytest;
from simutree.blocks import LeafBlock, NonLeafBlock;
import math;

class Constant(LeafBlock):
   def __init__(self,value,**kwargs):
      LeafBlock.__init__(self,num_outputs=1,**kwargs);
      self.value = value;
   
   def output_function(self,t):
      return [self.value];

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

class DCMotor(LeafBlock):
   def __init__(self,Kv,R,L,J,**kwargs):
      LeafBlock.__init__(self,num_inputs=2,num_states=2,num_outputs=2,feedthrough_inputs=[],**kwargs);
      self.Kv = Kv;
      self.R = R;
      self.L = L;
      self.J = J;
   
   def state_update_function(self,t,states,inputs):
      return [(inputs[0]-self.Kv*states[0]-self.R*states[1])/self.L,
              (self.Kv*states[0]-inputs[1])/self.J];
   
   def output_function(self,t,states,inputs):
      return [states[0]/(2*math.pi),states[1]];

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

class ExponentialDecay(LeafBlock):
   def __init__(self,T,**kwargs):
      LeafBlock.__init__(self,num_states=1,num_outputs=1,**kwargs);
      self.T = T;
   
   def state_update_function(self,t,states):
      return -states[0]/self.T;
   
   def output_function(self,t,states):
      return states;

@pytest.fixture
def decay_model():
   return ExponentialDecay(T=1.,initial_condition=[10.0]);
   
@pytest.fixture
def dcmotor_model():
   return DCMotorModel();
