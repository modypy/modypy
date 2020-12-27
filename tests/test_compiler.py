import pytest
from simtree.blocks import LeafBlock, NonLeafBlock
from simtree.compiler import Compiler
from fixtures.models import dcmotor_model, decay_model


def test_enumerate_blocks_pre_order(dcmotor_model):
    blocks_pre_order = list(
        Compiler.enumerate_blocks_pre_order(dcmotor_model.system))
    assert blocks_pre_order.index(
        dcmotor_model.system) < blocks_pre_order.index(dcmotor_model.voltage)
    assert blocks_pre_order.index(
        dcmotor_model.system) < blocks_pre_order.index(dcmotor_model.density)
    assert blocks_pre_order.index(
        dcmotor_model.system) < blocks_pre_order.index(dcmotor_model.engine)
    assert blocks_pre_order.index(
        dcmotor_model.engine) < blocks_pre_order.index(dcmotor_model.dcmotor)
    assert blocks_pre_order.index(dcmotor_model.engine) < blocks_pre_order.index(
        dcmotor_model.static_propeller)


def test_enumerate_blocks_post_order(dcmotor_model):
    blocks_post_order = list(
        Compiler.enumerate_blocks_post_order(dcmotor_model.system))
    assert blocks_post_order.index(
        dcmotor_model.system) > blocks_post_order.index(dcmotor_model.voltage)
    assert blocks_post_order.index(
        dcmotor_model.system) > blocks_post_order.index(dcmotor_model.density)
    assert blocks_post_order.index(
        dcmotor_model.system) > blocks_post_order.index(dcmotor_model.engine)
    assert blocks_post_order.index(
        dcmotor_model.engine) > blocks_post_order.index(dcmotor_model.dcmotor)
    assert blocks_post_order.index(dcmotor_model.engine) > blocks_post_order.index(
        dcmotor_model.static_propeller)


def test_compile(dcmotor_model):
    compiler = Compiler(dcmotor_model.system)
    result = compiler.compile()

    # Check counts of internal states and external outputs
    assert result.num_states == 2
    assert result.num_outputs == 0

    # Check leaf input connections
    sp_idx = result.block_index[dcmotor_model.static_propeller]
    dcm_idx = result.block_index[dcmotor_model.dcmotor]
    voltage_idx = result.block_index[dcmotor_model.voltage]
    rho_idx = result.block_index[dcmotor_model.density]
    engine_idx = result.block_index[dcmotor_model.engine]

    # Check connection from dcmotor:0 to static_propeller:0
    assert result.output_to_signal_map[result.first_output_by_block_index[dcm_idx] + 0] == \
        result.input_to_signal_map[result.first_input_by_block_index[sp_idx] + 0]
    # Check connection from dcmotor:1 to engine:1 (output)
    assert result.output_to_signal_map[result.first_output_by_block_index[dcm_idx] + 1] == \
        result.output_to_signal_map[result.first_output_by_block_index[engine_idx] + 1]
    # Check connection from static_propeller:1 to dcmotor:1
    assert result.output_to_signal_map[result.first_output_by_block_index[sp_idx] + 1] == \
        result.input_to_signal_map[result.first_input_by_block_index[dcm_idx] + 1]
    # Check connection from static_propeller:0 to engine:0 (output)
    assert result.output_to_signal_map[result.first_output_by_block_index[sp_idx] + 0] == \
        result.output_to_signal_map[result.first_output_by_block_index[engine_idx] + 0]
    # Check connection from voltage:0 to dcmotor:0
    assert result.output_to_signal_map[result.first_output_by_block_index[voltage_idx] + 0] == \
        result.input_to_signal_map[result.first_input_by_block_index[dcm_idx] + 0]
    # Check connection from density:0 to static_propeller:1
    assert result.output_to_signal_map[result.first_output_by_block_index[rho_idx] + 0] == \
        result.input_to_signal_map[result.first_input_by_block_index[sp_idx] + 1]

    # Check execution order
    exec_seq = result.leaf_blocks_in_order
    assert exec_seq.index(dcmotor_model.dcmotor) < exec_seq.index(
        dcmotor_model.static_propeller)
    assert exec_seq.index(dcmotor_model.density) < exec_seq.index(
        dcmotor_model.static_propeller)


def test_compile_cyclic(dcmotor_model):
    # Enforce a cyclic dependency
    dcmotor_model.dcmotor.feedthrough_inputs = [0, 1]
    compiler = Compiler(dcmotor_model.system)
    with pytest.raises(ValueError):
        compiler.compile()


def test_compile_no_inputs(decay_model):
    compiler = Compiler(decay_model)
    result = compiler.compile()
    assert result.num_states == 1
    assert result.num_outputs == 1
    assert result.initial_condition == pytest.approx([10.0])

    initial_output = result.output_function(0, result.initial_condition)
    assert initial_output == pytest.approx([10.0])

    initial_state_update = result.state_update_function(
        0, result.initial_condition, initial_output)
    assert initial_state_update == pytest.approx([-10.0])


def test_compiled_system(dcmotor_model):
    compiler = Compiler(dcmotor_model.system)
    result = compiler.compile()

    initial_signals = result.signal_function(0, result.initial_condition)
    assert initial_signals == pytest.approx(
        [4.44, 1.29, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1.E-2)

    initial_state_update = result.state_update_function(
        0, result.initial_condition)
    assert initial_state_update == pytest.approx([0.0, 2336.84], abs=1.E-2)
