from simtree.blocks import LeafBlock;
import numpy as np;

"""
A constant source block.

Provides the given constant value as output.
"""
class Constant(LeafBlock):
   def __init__(self,value,**kwargs):
      LeafBlock.__init__(self,**kwargs);
      self.value = np.asarray(value).flatten();
      self.num_outputs = self.value.size;
   
   def output_function(self,t):
      return self.value;

class SourceFromCallable(LeafBlock):
   def __init__(self,callable,**kwargs):
      LeafBlock.__init__(self,**kwargs);
      self.callable = callable;
   
   def output_function(self,*args):
      return self.callable(*args);