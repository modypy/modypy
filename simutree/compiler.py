import numpy as np;
import itertools;

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
