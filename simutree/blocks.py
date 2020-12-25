import numpy as np;
import bisect;
import itertools;

class Block(object):
   def __init__(self,
                num_inputs=0,
                num_outputs=0,
                name=None,
                feedthrough_inputs=None):
      self.num_inputs = num_inputs;
      self.num_outputs = num_outputs;
      self.name = name;
      if feedthrough_inputs is None:
         feedthrough_inputs = range(self.num_inputs);
      self.feedthrough_inputs = feedthrough_inputs;
   
   """
   Yield all non-virtual blocks contained in this block.
   """
   def enumerate_leaf_blocks(self):
      yield self;

class NonLeafBlock(Block):
   def __init__(self,
                children=[],
                **kwargs):
      Block.__init__(self,**kwargs);
      
      # List of children contained in this block
      self.children = [];
      # Map of child indices by block object
      self.child_index = {};
      
      # Start indices for connection sources
      # Indices 0...num_inputs-1 are reserved for parent block inputs.
      # Indices >=num_inputs are used for outputs of contained blocks.
      self.first_source_index = [self.num_inputs];
      
      # Start indices for connection destinations
      # Indices 0...num_outputs-1 are reserved for parent block outputs.
      # Indices >= num_outputs are used for inputs of contained blocks.
      self.first_destination_index = [self.num_outputs];
      
      # Connection sources by connection destination
      self.connection_source = [None]*self.num_outputs;
      
      for child in children:
         self.add_block(child);
   
   """
   Add a child block to this non-leaf block.
   """
   def add_block(self, child):
      if child in self.child_index:
         raise ValueError("Child is already contained in this non-leaf block");
      
      # Add the child to the list of children
      index = len(self.children);
      self.children.append(child);
      self.child_index[child] = index;
      
      # Set up source and destination indices
      self.first_source_index.append(self.first_source_index[-1]+child.num_outputs);
      self.first_destination_index.append(self.first_destination_index[-1]+child.num_inputs);
      self.connection_source.extend([None]*child.num_inputs);
   
   """
   Connect the output of one block to an input of another block.
   """
   def connect(self,src,src_output_index,dest,dest_input_index):
      src_block_index = self.child_index[src];
      dest_block_index = self.child_index[dest];
      
      if not(0<=src_output_index and src_output_index<src.num_outputs):
         raise ValueError("Invalid source output index");
      if not(0<=dest_input_index and dest_input_index<dest.num_inputs):
         raise ValueError("Invalid destination input index");
      
      src_port_index = self.first_source_index[src_block_index]+src_output_index;
      dest_port_index = self.first_destination_index[dest_block_index]+dest_input_index;
      
      self.connection_source[dest_port_index]=src_port_index;
   
   """
   Connect an input of this virtual block to an input of a contained block.
   """
   def connect_input(self,input_index,dest,dest_input_index):
      dest_block_index = self.child_index[dest];
      
      if not(0<=input_index and input_index<self.num_inputs):
         raise ValueError("Invalid source input index");
      if not(0<=dest_input_index and dest_input_index<dest.num_inputs):
         raise ValueError("Invalid destination input index");
      
      dest_port_index = self.first_destination_index[dest_block_index]+dest_input_index;
      
      self.connection_source[dest_port_index]=input_index;
   
   """
   Connect an output of a contained block to an output of this virtual block.
   """
   def connect_output(self,src,src_output_index,output_index):
      src_block_index = self.child_index[src];
      
      if not(0<=src_output_index and src_output_index<src.num_outputs):
         raise ValueError("Invalid source output index");
      if not(0<=output_index and output_index<self.num_outputs):
         raise ValueError("Invalid destination output index");
      
      src_port_index = self.first_source_index[src_block_index]+src_output_index;
      
      self.connection_source[output_index]=src_port_index;
   
   """
   Yield all internal connections.
   
   Each connection is represented by a tuple (src_block,src_port,dest_block,dest_port), with
   
    - src_block being the block providing the source port,
    - src_port being the index of the output on the source block,
    - dest_block being the block receiving the signal, and
    - dest_port being the index of the input on the destination block.
   """
   def enumerate_internal_connections(self):
      for dest_block_index, dest_block, first_dest_index in zip(itertools.count(),self.children, self.first_destination_index):
         for dest_port_index, src_index in zip(range(dest_block.num_inputs),self.connection_source[first_dest_index:]):
            if src_index is not None and src_index>=self.num_inputs:
               # This is a connection from an internal block
               src_block_index = bisect.bisect_right(self.first_source_index,src_index)-1;
               src_block = self.children[src_block_index];
               src_port_index = src_index-self.first_source_index[src_block_index];
               yield (src_block,src_port_index,dest_block,dest_port_index);
   
   """
   Yield all input connections.
   
   Each connection is represented by a tuple (src_port,dest_block,dest_port), with
   
    - src_port being the index of the input on the containing block,
    - dest_block being the block receiving the signal, and
    - dest_port being the index of the input on the destination block.
   """
   def enumerate_input_connections(self):
      for dest_block_index, dest_block, first_dest_index in zip(itertools.count(),self.children, self.first_destination_index):
         for dest_port_index, src_index in zip(range(dest_block.num_inputs),self.connection_source[first_dest_index:]):
            if src_index is not None and src_index<self.num_inputs:
               # This is a connection from an external input
               yield (src_index,dest_block,dest_port_index);
   
   """
   Yield all output connections.
   
   Each connection is represented by a tuple (src_block,src_port,dest_port), with
   
    - src_block being the block providing the source port,
    - src_port being the index of the output on the source block,
    - dest_port being the index of the output on the containing block.
   """
   def enumerate_output_connections(self):
      for dest_port_index, src_index in zip(range(self.num_outputs),self.connection_source):
         src_block_index = bisect.bisect_right(self.first_source_index,src_index)-1;
         src_block = self.children[src_block_index];
         src_port_index = src_index-self.first_source_index[src_block_index];
         yield (src_block,src_port_index,dest_port_index);
      
   def enumerate_leaf_blocks(self):
      for child_block in self.children:
         yield from child_block.enumerate_leaf_blocks();

class LeafBlock(Block):
   def __init__(self,
                num_states=0,
                initial_condition=None,
                **kwargs):
      Block.__init__(self,**kwargs);
      self.num_states = num_states;
      if initial_condition is None:
         initial_condition = np.zeros(self.num_states);
      self.initial_condition = initial_condition;
