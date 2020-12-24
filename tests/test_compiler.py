import pytest;
from simutree.blocks import LeafBlock, NonLeafBlock;
from simutree.compiler import Compiler;
from fixtures.models import dcmotor_model;

def test_enumerate_blocks_pre_order(dcmotor_model):
   blocks_pre_order = list(Compiler.enumerate_blocks_pre_order(dcmotor_model.system));
   assert blocks_pre_order.index(dcmotor_model.system)<blocks_pre_order.index(dcmotor_model.voltage);
   assert blocks_pre_order.index(dcmotor_model.system)<blocks_pre_order.index(dcmotor_model.density);
   assert blocks_pre_order.index(dcmotor_model.system)<blocks_pre_order.index(dcmotor_model.engine);
   assert blocks_pre_order.index(dcmotor_model.engine)<blocks_pre_order.index(dcmotor_model.dcmotor);
   assert blocks_pre_order.index(dcmotor_model.engine)<blocks_pre_order.index(dcmotor_model.static_propeller);

def test_enumerate_blocks_post_order(dcmotor_model):
   blocks_post_order = list(Compiler.enumerate_blocks_post_order(dcmotor_model.system));
   assert blocks_post_order.index(dcmotor_model.system)>blocks_post_order.index(dcmotor_model.voltage);
   assert blocks_post_order.index(dcmotor_model.system)>blocks_post_order.index(dcmotor_model.density);
   assert blocks_post_order.index(dcmotor_model.system)>blocks_post_order.index(dcmotor_model.engine);
   assert blocks_post_order.index(dcmotor_model.engine)>blocks_post_order.index(dcmotor_model.dcmotor);
   assert blocks_post_order.index(dcmotor_model.engine)>blocks_post_order.index(dcmotor_model.static_propeller);

def test_compile(dcmotor_model):
   compiler = Compiler(dcmotor_model.system);
   compiler.compile();
   
   # Check counts of internal states and outputs
   assert compiler.leaf_first_state_index[-1]==2;
   assert compiler.leaf_first_output_index[-1]==6;
   
   # Check leaf input connections
   sp_idx = compiler.block_index[dcmotor_model.static_propeller];
   dcm_idx = compiler.block_index[dcmotor_model.dcmotor];
   voltage_idx = compiler.block_index[dcmotor_model.voltage];
   rho_idx = compiler.block_index[dcmotor_model.density];
   engine_idx = compiler.block_index[dcmotor_model.engine];
   
   # Check connection from dcmotor:0 to static_propeller:0
   assert compiler.output_vector_index[compiler.first_output_index[dcm_idx]+0]==compiler.input_vector_index[compiler.first_input_index[sp_idx]+0];
   # Check connection from static_propeller:1 to dcmotor:1
   assert compiler.output_vector_index[compiler.first_output_index[sp_idx]+1]==compiler.input_vector_index[compiler.first_input_index[dcm_idx]+1];
   # Check connection from voltage:0 to dcmotor:0
   assert compiler.output_vector_index[compiler.first_output_index[voltage_idx]+0]==compiler.input_vector_index[compiler.first_input_index[dcm_idx]+0];
   # Check connection from density:0 to static_propeller:1
   assert compiler.output_vector_index[compiler.first_output_index[voltage_idx]+0]==compiler.input_vector_index[compiler.first_input_index[dcm_idx]+0];
   # Check connection from static_propeller:0 to engine:0 (output)
   assert compiler.output_vector_index[compiler.first_output_index[sp_idx]+0]==compiler.output_vector_index[compiler.first_output_index[engine_idx]+0];
   # Check connection from dcmotor:1 to engine:1 (output)
   assert compiler.output_vector_index[compiler.first_output_index[dcm_idx]+1]==compiler.output_vector_index[compiler.first_output_index[engine_idx]+1];
   
   # Check execution order
   exec_seq = compiler.execution_sequence;
   assert exec_seq.index(dcmotor_model.dcmotor)<exec_seq.index(dcmotor_model.static_propeller);
   assert exec_seq.index(dcmotor_model.density)<exec_seq.index(dcmotor_model.static_propeller);

def test_compile_cyclic(dcmotor_model):
   # Enforce a cyclic dependency
   dcmotor_model.dcmotor.feedthrough_inputs=[0,1];
   compiler = Compiler(dcmotor_model.system);
   with pytest.raises(ValueError):
      compiler.compile();
