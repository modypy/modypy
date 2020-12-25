import pytest;
from simutree.blocks import LeafBlock, NonLeafBlock;
from simutree.compiler import Compiler;
from simutree.simulator import Simulator;
from fixtures.models import dcmotor_model, decay_model;

def test_run_dcmotor(dcmotor_model):
   compiler = Compiler(dcmotor_model.system);
   result = compiler.compile();
   
   simulator = Simulator(result,t0=0.,tbound=2.0 );
   simulator.run();
   
   assert simulator.status == 'finished';
   assert simulator.result.t[-1] == pytest.approx(2.0);

def test_run_decay_model(decay_model):
   compiler = Compiler(decay_model);
   result = compiler.compile();
   
   simulator = Simulator(result,t0=0.,tbound=10.0,initial_condition=[1.0]);
   simulator.run();
   
   assert simulator.status == 'finished';
   assert simulator.result.t[-1] == pytest.approx(10.0);
   assert simulator.result.state[-1,0] < 4.6E-5;
   assert simulator.result.state[-1,0] > 0;
   assert simulator.result.output == pytest.approx(simulator.result.state);

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
   