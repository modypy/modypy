import numpy as np;
import itertools;

class CompiledSystem:
   def __init__(self,
                blocks,
                block_index,
                first_input_index,
                first_state_index,
                first_output_index,
                input_index,
                global_block_index,
                global_first_input_index,
                global_first_output_index,
                global_input_index,
                global_output_index):
      self.blocks = blocks;
      self.block_index = block_index;
      self.first_input_index = first_input_index;
      self.first_state_index = first_state_index;
      self.first_output_index = first_output_index;
      self.input_index = input_index;
      self.global_block_index = global_block_index;
      self.global_first_input_index = global_first_input_index;
      self.global_first_output_index = global_first_output_index;
      self.global_input_index = global_input_index;
      self.global_output_index = global_output_index;
      
      self.num_outputs = self.first_output_index[-1];
      self.num_states = self.first_state_index[-1];
      
      self.initial_condition = np.zeros(self.num_states);
      for block,block_index in self.block_index.items():
         first_state = self.first_state_index[block_index];
         self.initial_condition[first_state:first_state+block.num_states] = block.initial_condition;
   
   """
   Combined output function of the compiled system.
   
   This calculates the output vector for all the leaf blocks contained in the system.
   """
   def output_function(self,t,state):
      outputs = np.zeros(self.num_outputs);
      # Calculate the output vector
      for block in self.blocks:
         # We also process blocks without outputs. These may be sinks.
         block_index = self.block_index[block];
         
         first_input = self.first_input_index[block_index];
         first_state = self.first_state_index[block_index];
         first_output = self.first_output_index[block_index];
         input_indices = self.input_index[first_input:first_input+block.num_inputs];
         
         block_inputs = outputs[input_indices];
         block_states = state[first_state:first_state+block.num_states];
         
         if block.num_inputs>0 and block.num_states>0:
            # This system has inputs and states
            block_outputs = block.output_function(t,block_states,block_inputs);
         elif block.num_states>0:
            # This system has states, but no inputs
            block_outputs = block.output_function(t,block_states);
         elif block.num_inputs>0:
            # This system has inputs, but no states
            block_outputs = block.output_function(t,block_inputs);
         else:
            # This system neither has inputs nor states
            block_outputs = block.output_function(t);
         
         outputs[first_output:first_output+block.num_outputs] = block_outputs;
      
      return outputs;

   """
   Combined state update function of the compiled system.
   
   This calculates the state update vector for all the leaf blocks contained in the system.
   """
   def state_update_function(self,t,state,outputs):
      state_derivative = np.zeros(self.num_states);
      for block,block_index in self.block_index.items():
         if block.num_states>0:
            # We only consider blocks that have state
            first_input = self.first_input_index[block_index];
            first_state = self.first_state_index[block_index];
            input_indices = self.input_index[first_input:first_input+block.num_inputs];
            
            block_inputs = outputs[input_indices];
            block_states = state[first_state:first_state+block.num_states];
            
            if block.num_inputs>0:
               # This system has inputs
               block_state_derivative = block.state_update_function(t,block_states,block_inputs);
            else:
               # This system has no inputs
               block_state_derivative = block.state_update_function(t,block_states);
         
            state_derivative[first_state:first_state+block.num_states] = block_state_derivative;
      
      return state_derivative;
      

"""
Compiler for block trees.

- Determine leaf blocks.
- Determines the size and mapping of state and output vectors for leaf blocks.
- Determine mapping of inputs to output vector items for all blocks.
- Determine execution sequence for leaf blocks.
"""
class Compiler:
   def __init__(self,root):
      self.root = root;
   
   """
   Compile the block graph given as `root`.
   
   Compilation consists of collection of all leaf blocks, establishment of the input-output connections and
   determination of an execution order consistent with the inter-block dependencies.
   """
   def compile(self):
      # Collect all blocks in the graph
      self.blocks = list(Compiler.enumerate_blocks_pre_order(self.root));
      self.block_index = {block:index for index,block in zip(itertools.count(),self.blocks)};
      
      # Allocate input and output indices for all blocks
      # The first_input_index and first_output_index arrays give the index of the first input or output associated
      # with the block identified by the index, respectively.
      self.first_input_index = [0]+list(itertools.accumulate([block.num_outputs for block in self.blocks]));
      self.first_output_index = [0]+list(itertools.accumulate([block.num_outputs for block in self.blocks]));
      
      # Set up input and output vector maps for all blocks
      # For each input input_idx of block block_idx, self.input_vector_index[self.first_input_index[block_idx]+input_idx]
      # gives the index of the corresponding entry in the output vector.
      self.input_vector_index = [None]*self.first_input_index[-1];
      # For each output output_idx of block block_idx, self.output_vector_index[self.first_output_index[block_idx]+output_idx]
      # gives the index of the corresponding entry in the output vector.
      self.output_vector_index = [None]*self.first_output_index[-1];
      
      # Collect all leaf blocks in the graph
      self.leaf_blocks = list(self.root.enumerate_leaf_blocks());
      self.leaf_block_index = {block:index for index,block in zip(itertools.count(),self.leaf_blocks)};
      
      # Allocate the input, output and state indices for the leaf blocks
      self.leaf_first_input_index = [0]+list(itertools.accumulate([block.num_inputs for block in self.leaf_blocks]));
      self.leaf_first_output_index = [0]+list(itertools.accumulate([block.num_outputs for block in self.leaf_blocks]));
      self.leaf_first_state_index = [0]+list(itertools.accumulate([block.num_states for block in self.leaf_blocks]));
      
      # Set up the input vector maps for leaf blocks
      # self.leaf_input_vector_index[self.leaf_first_input_index[block_idx]+input_index] gives the offset of the
      # entry associated with input input_index of leaf block block_idx in the output vector.
      self.leaf_input_vector_index = [None]*self.leaf_first_input_index[-1];
      
      # Establish input-to-output mapping
      self.map_inputs_to_outputs();
      # Establish execution order
      self.build_execution_order();
      
      return CompiledSystem(blocks=self.execution_sequence,
                               block_index=self.leaf_block_index,
                               first_input_index=self.leaf_first_input_index,
                               first_state_index=self.leaf_first_state_index,
                               first_output_index=self.leaf_first_output_index,
                               input_index=self.leaf_input_vector_index,
                               global_block_index=self.block_index,
                               global_first_input_index=self.first_input_index,
                               global_first_output_index=self.first_output_index,
                               global_input_index=self.input_vector_index,
                               global_output_index=self.output_vector_index);

   """
   Fill output_vector_index with the output vector indices for leaf blocks.
   """
   def map_leaf_block_outputs(self):
      for block,leaf_block_index in self.leaf_block_index.items():
         block_index = self.block_index[block];
         output_vector_offset = self.leaf_first_output_index[leaf_block_index];
         output_offset = self.first_output_index[block_index];
         self.output_vector_index[output_offset:output_offset+block.num_outputs]=range(output_vector_offset,output_vector_offset+block.num_outputs);

   """
   Fill output_vector_index for outputs of non-leaf blocks.
   """
   def map_nonleaf_block_outputs(self):
      for block in Compiler.enumerate_blocks_post_order(self.root):
         if block not in self.leaf_block_index:
            block_index = self.block_index[block];
            dest_port_offset = self.first_output_index[block_index];
            for src_block,src_port_index,output_index in block.enumerate_output_connections():
               src_block_index = self.block_index[src_block];
               src_port_offset = self.first_output_index[src_block_index];
               self.output_vector_index[dest_port_offset+output_index]=self.output_vector_index[src_port_offset+src_port_index];

   """
   Fill input_vector_index for the destinations of internal connections.
   """
   def map_internal_connections(self):
      for block in Compiler.enumerate_blocks_post_order(self.root):
         if block not in self.leaf_block_index:
            for src_block,src_port_index,dst_block,dest_port_index in block.enumerate_internal_connections():
               src_block_index = self.block_index[src_block];
               src_port_offset = self.first_output_index[src_block_index];
               dest_block_index = self.block_index[dst_block];
               dest_port_offset = self.first_input_index[dest_block_index];
               self.input_vector_index[dest_port_offset+dest_port_index]=self.output_vector_index[src_port_offset+src_port_index];

   """
   Fill input_vector_index for the destinations of non-leaf block inputs.
   """
   def map_nonleaf_block_inputs(self):
      for block in Compiler.enumerate_blocks_pre_order(self.root):
         if block not in self.leaf_block_index:
            block_index = self.block_index[block];
            src_port_offset = self.first_input_index[block_index];
            for input_index,dst_block,dest_port_index in block.enumerate_input_connections():
               dest_block_index = self.block_index[dst_block];
               dest_port_offset = self.first_input_index[dest_block_index];
               self.input_vector_index[dest_port_offset+dest_port_index]=self.input_vector_index[src_port_offset+input_index];

   """
   Build the mapping of inputs and outputs to entries of the output vector.
   """
   def map_inputs_to_outputs(self):
      # Pre-populate for leaf blocks
      self.map_leaf_block_outputs();
      
      # Iterate over all non-leaf blocks and process outgoing connections
      self.map_nonleaf_block_outputs();
      
      # Iterate over all non-leaf blocks and process internal connections
      self.map_internal_connections();
      
      # Iterate over all non-leaf blocks and process incoming connections
      self.map_nonleaf_block_inputs();
      
      # Establish the leaf input vector map
      for leaf_block,leaf_block_idx in self.leaf_block_index.items():
         block_idx = self.block_index[leaf_block];
         leaf_input_start = self.leaf_first_input_index[leaf_block_idx];
         global_input_start = self.first_input_index[block_idx];
         self.leaf_input_vector_index[leaf_input_start:leaf_input_start+leaf_block.num_inputs] = \
            self.input_vector_index[global_input_start:global_input_start+leaf_block.num_inputs];
   
   """
   Establish an execution order of leaf blocks compatible with inter-block dependencies.
   """
   def build_execution_order(self):
      # We use Kahn's algorithm here
      # Initialize the number of incoming connections by leaf block
      num_incoming = [len(block.feedthrough_inputs) for block in self.leaf_blocks];

      # Initialize the list of leaf blocks that do not require any inputs
      S = [block for block,num_in in zip(self.leaf_blocks,num_incoming) if num_in==0];
      # Initialize the execution sequence
      L = [];
      while len(S)>0:
         src = S.pop();
         L.append(src);
         # Consider the targets of all outgoing connections of src
         src_index = self.leaf_block_index[src];
         src_output_index_start = self.leaf_first_output_index[src_index];
         src_output_index_end = src_output_index_start+src.num_outputs;
         for dest,dest_leaf_index in self.leaf_block_index.items():
            dest_port_offset = self.leaf_first_input_index[dest_leaf_index];
            # We consider only ports that feed through
            for dest_port in dest.feedthrough_inputs:
               dest_port_index = dest_port_offset+dest_port;
               src_port_index = self.leaf_input_vector_index[dest_port_index];
               if src_output_index_start <= src_port_index and src_port_index < src_output_index_end:
                  # The destination port is connected to the source system
                  num_incoming[dest_leaf_index]=num_incoming[dest_leaf_index]-1;
                  if num_incoming[dest_leaf_index]==0:
                     # All feedthrough input ports of the destination are satisfied
                     S.append(dest);
      
      if sum(num_incoming)>0:
         # There are unsatisfied inputs, which is probably due to cycles in the graph
         unsatisfied_blocks = [block.name for block,inputs in zip(self.leaf_blocks,num_incoming) if inputs>0];
         raise ValueError("Unsatisfied inputs for blocks %s" % unsatisfied_blocks);
      
      self.execution_sequence = L;
   
   """
   Enumerate all blocks in the graph with the given root in pre-order.
   """
   @staticmethod
   def enumerate_blocks_pre_order(root):
      yield root;
      try:
         for child in root.children:
            yield from Compiler.enumerate_blocks_pre_order(child);
      except AttributeError:
         # Ignore this, the root is not a non-leaf node
         pass;
   
   """
   Enumerate all blocks in the graph with the given root in post-order.
   """
   @staticmethod
   def enumerate_blocks_post_order(root):
      try:
         for child in root.children:
            yield from Compiler.enumerate_blocks_post_order(child);
      except AttributeError:
         # Ignore this, the root is not a non-leaf node
         pass;
      yield root;
