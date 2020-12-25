import numpy as np;
from scipy.integrate import DOP853;

INITIAL_RESULT_SIZE = 16;
RESULT_SIZE_EXTENSION = 16;

DEFAULT_INTEGRATOR = DOP853;
DEFAULT_INTEGRATOR_OPTIONS = {
   'rtol': 1.E-6,
   'atol': 1.E-6
};

"""
The results provided by a simulation.

A `SimulationResult` object captures the time series provided by a simulation.
It has properties `t`, `state` and `output` representing the time, state vector and
output vector for each individual sample.
"""
class SimulationResult:
   def __init__(self,system):
      self.system = system;
      self._t = np.empty(INITIAL_RESULT_SIZE);
      self._state = np.empty((INITIAL_RESULT_SIZE,self.system.num_states));
      self._output = np.empty((INITIAL_RESULT_SIZE,self.system.num_outputs));
      
      self.current_idx = 0;
   
   @property
   def t(self):
      return self._t[0:self.current_idx];
   
   @property
   def state(self):
      return self._state[0:self.current_idx];
   
   @property
   def output(self):
      return self._output[0:self.current_idx];
   
   """
   Append a sample to the result.
   """
   def append(self,t,state,output):
      if self.current_idx >= self._t.size:
         self.extend_space();
      self._t     [self.current_idx] = t;
      self._state [self.current_idx] = state;
      self._output[self.current_idx] = output;
      
      self.current_idx += 1;
   
   def extend_space(self):
      self._t      = np.r_[self._t,      np.empty(RESULT_SIZE_EXTENSION)];
      self._state  = np.r_[self._state,  np.empty((RESULT_SIZE_EXTENSION,self.system.num_states))];
      self._output = np.r_[self._output, np.empty((RESULT_SIZE_EXTENSION,self.system.num_outputs))];

"""
Simulator for dynamic systems.

Dynamic systems to be simulated using this class need to support a set of functions:

system.state_update_function(t,state,outputs)
   Determine the derivative of the state vector of the system, given
   the time `t`, state `state` and output vector `outputs`.

system.output_function(t,state)
   Determine the value of the outputs of the system given time `t` and
   state `state`.

system.initial_condition
   The initial value of the state vector.
"""
class Simulator:
   """
   Construct a simulator for a system.
   
   system
     The system to be simulated. This can be the result of a compilation using `simutree.compiler.Compiler`.
   t0: number
     The start time of the simulation.
   tbound: number
     The end time of the simulation. This also limits the maximum time until which stepping is possible.
   initial_condition: list-like of numbers, optional
     The initial condition of the system state. If not set, the initial condition specified in the system is used.
   integrator_constructor: function or class, optional
     The constructor to be used to instantiate the integrator. If not given, `DEFAULT_INTEGRATOR` is used.
   integrator_options: dictionary, optional
     Additional parameters to be passed to the integrator constructor. If not given, `DEFAULT_INTEGRATOR_OPTIONS` is used.
   
   The simulator is written with the interface of `scipy.integrate.OdeSolver` in mind for the integrator, specifically
   using the constructor, the `step` and the `dense_output` functions as well as the `status` property. However, it is
   possible to use other integrators if they honor this interface.
   """
   def __init__(self,system,t0,tbound,initial_condition=None,integrator_constructor=DEFAULT_INTEGRATOR,integrator_options=DEFAULT_INTEGRATOR_OPTIONS):
      self.system = system;
      self.result = SimulationResult(system);

      # Set up the integrator
      if initial_condition is None:
         initial_condition = self.system.initial_condition;
      
      # Define the state derivative function for the integrator
      def state_derivative_function(t,state):
         outputs = self.system.output_function(t,state);
         dxdt = self.system.state_update_function(t,state,outputs);
         return dxdt;
      
      self.integrator = integrator_constructor(state_derivative_function,t0,initial_condition,tbound,**integrator_options);
      
      # Store the initial state
      self.result.append(self.t,self.state,self.output);

   """The current simulation time."""
   @property
   def t(self):
      return self.integrator.t;
   
   """The current state of the simulated system."""
   @property
   def state(self):
      return self.integrator.y;

   """The current outputs of the simulated system."""
   @property
   def output(self):
      return self.system.output_function(self.t,self.state);
   
   """The current status of the integrator."""
   @property
   def status(self):
      return self.integrator.status;
   
   """Boolean indicating whether the simulation is still running, i.e. has not been finished or aborted."""
   @property
   def running(self):
      return self.integrator.status=="running";
   
   """Execute a single simulation step."""
   def step(self):
      message = self.integrator.step();
      if message is None:
         # The last integration step was successful.
         # Add the current status to the result collection
         self.result.append(self.t,self.state,self.output);
      return message;
   
   """Simulate the system until the end time of the simulation."""
   def run(self):
      while self.running:
         message=self.step();
         if message is not None:
            return message;
      return None;
 