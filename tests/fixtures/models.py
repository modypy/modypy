import pytest;
from simutree.blocks import LeafBlock, NonLeafBlock;

class DCMotorModel:
   def __init__(self):
      self.dcmotor = LeafBlock(name="dcmotor", num_inputs=2, num_outputs=2, num_states=2, feedthrough_inputs=[]);
      self.static_propeller = LeafBlock(name="propeller", num_inputs=2, num_outputs=2);

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
      
      self.voltage=LeafBlock(name="self.voltage",num_outputs=1);
      self.density=LeafBlock(name="rho",num_outputs=1);
      
      self.system = NonLeafBlock(name="self.system",
                            children=[self.voltage,self.density,self.engine]);
      self.system.connect(self.voltage,0,self.engine,0);
      self.system.connect(self.density,0,self.engine,1);
      
      self.leaf_blocks = set([self.dcmotor, self.static_propeller, self.voltage, self.density]);

@pytest.fixture
def dcmotor_model():
   return DCMotorModel();
