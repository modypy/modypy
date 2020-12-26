import pytest;
import numpy as np;
from simtree.blocks import LeafBlock, NonLeafBlock;
from simtree.compiler import Compiler;
from simtree.simulator import Simulator;
from fixtures.models import dcmotor_model, decay_model, decay_model_no_state;

def test_run_dcmotor(dcmotor_model):
   compiler = Compiler(dcmotor_model.system);
   result = compiler.compile();
   
   simulator = Simulator(result,t0=0.,tbound=2.0 );
   simulator.run();
   
   # Check that simulator is done
   assert simulator.status == 'finished';
   
   # Check that simulator has run til the end
   assert simulator.result.t[-1] == pytest.approx(2.0);
   
   # Check for correct localisation of events
   dcmotor_idx = result.block_index[dcmotor_model.dcmotor];
   propeller_idx = result.block_index[dcmotor_model.static_propeller];
   
   evt_speed   = result.first_event_index[dcmotor_idx]+0;
   evt_current = result.first_event_index[dcmotor_idx]+1;
   evt_power   = result.first_event_index[propeller_idx]+0;
   
   state_speed   = result.first_state_index[dcmotor_idx]+0;
   state_current = result.first_state_index[dcmotor_idx]+1;
   
   output_power  = result.first_output_index[propeller_idx]+1;
   
   evt_speed_idxs   = np.flatnonzero(simulator.result.events[:,evt_speed]);
   evt_current_idxs = np.flatnonzero(simulator.result.events[:,evt_current]);
   evt_power_idxs   = np.flatnonzero(simulator.result.events[:,evt_power]);
   
   assert simulator.result.state[evt_speed_idxs  ,state_speed  ] == pytest.approx(101.);
   assert simulator.result.state[evt_current_idxs,state_current] == pytest.approx(87.75);
   assert simulator.result.output[evt_power_idxs  ,output_power] == pytest.approx(0.04);

def test_run_decay_model(decay_model):
   compiler = Compiler(decay_model);
   result = compiler.compile();
   
   simulator = Simulator(result,t0=0.,tbound=10.0,initial_condition=[1.0]);
   simulator.run();
   
   assert simulator.status == 'finished';
   assert simulator.result.t[-1] == pytest.approx(10.0);
   assert simulator.result.output[-1,0] == pytest.approx(4.5E-5,rel=0.1);
   assert simulator.result.output == pytest.approx(simulator.result.state);

def test_run_decay_model_nostate(decay_model_no_state):
   compiler = Compiler(decay_model_no_state);
   result = compiler.compile();
   
   simulator = Simulator(result,t0=0.,tbound=10.0);
   simulator.run();
   
   assert simulator.status == 'finished';
   assert simulator.result.t[-1] == pytest.approx(10.0);
   assert simulator.result.output[-1,0] == pytest.approx(4.5E-4,rel=0.1);

"""
Mockup integrator class to force integration error.
"""
class MockupIntegrator:
   def __init__(self,fun,t0,y,tbound):
      self.status = "running";
      self.t = t0;
      self.y = y;
   
   def step(self):
      self.status = "failed";
      return "failed";

def test_run_simulation_failure(decay_model):
   compiler = Compiler(decay_model);
   result = compiler.compile();
   
   simulator = Simulator(result,t0=0.,tbound=10.0,initial_condition=[1.0],integrator_constructor=MockupIntegrator,integrator_options={});
   message = simulator.run();
   assert message == "failed";
