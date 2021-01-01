from simtree.blocks import LeafBlock;
from simtree.compiler import Compiler;
from simtree.simulator import Simulator,DEFAULT_INTEGRATOR_OPTIONS;
import matplotlib.pyplot as plt;

class BouncingBall(LeafBlock):
   def __init__(self,g=-9.81,gamma=0.3,**kwargs):
      LeafBlock.__init__(self,num_states=4,num_events=1,num_outputs=2,**kwargs);
      self.g = g;
      self.gamma = gamma;
   
   def output_function(self,t,state):
      return [state[1]];
   
   def state_update_function(self,t,state):
      return [state[2],state[3],
              -self.gamma*state[2],self.g-self.gamma*state[3]];
   
   def event_function(self,t,state):
      return [state[1]];
   
   def update_state_function(self,t,state):
      return [state[0],abs(state[1]),state[2],-state[3]];

DEFAULT_INTEGRATOR_OPTIONS['max_step']=0.05;

sys = BouncingBall(initial_condition=[0,10,1,0]);

compiler = Compiler(sys);
sys_compiled = compiler.compile();

simulator = Simulator(sys_compiled, t0=0., t_bound=10.0);
simulator.run();

plt.plot(simulator.result.state[:,0],simulator.result.state[:,1]);
plt.show();