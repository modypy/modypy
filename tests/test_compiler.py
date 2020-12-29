import itertools
import numpy.testing as npt
import numpy.random as np_rand
import pytest
from fixtures.models import \
    propeller_model, engine_model, dcmotor_model, dcmotor_model_cyclic, \
    bouncing_ball_model
from simtree.blocks import NonLeafBlock, LeafBlock
from simtree.blocks.sources import Constant
from simtree.compiler import Compiler, CompiledSystem


def assert_input_connections_correct(block: NonLeafBlock,
                                     result: CompiledSystem):
    """
    Raises an AssertionError if an input connection in the given block is
    not correctly mapped in the given compilation result.

    A connection from a source port to a destination port is correctly mapped
    only if the source and the destination port are mapped to the same signal.
    """
    block_index = result.block_index[block]
    input_port_offset = result.first_input_by_block_index[block_index]
    for src_port, dest_block, dest_port in block.enumerate_input_connections():
        dest_block_index = result.block_index[dest_block]
        dest_port_offset = result.first_input_by_block_index[dest_block_index]
        src_signal = result.input_to_signal_map[input_port_offset + src_port]
        dest_signal = result.input_to_signal_map[dest_port_offset + dest_port]
        assert src_signal == dest_signal, \
            "Input %s(%d):%d is not properly connected to input %s(%d):%d" % (
                block.name, block_index, src_port,
                dest_block.name, dest_block_index, dest_port
            )


def assert_output_connections_correct(block: NonLeafBlock,
                                      result: CompiledSystem):
    """
    Raises an AssertionError if an output connection in the given block is
    not correctly mapped in the given compilation result.

    A connection from a source port to a destination port is correctly mapped
    only if the source and the destination port are mapped to the same signal.
    """
    block_index = result.block_index[block]
    output_port_offset = result.first_output_by_block_index[block_index]
    for src_block, src_port, dest_port in block.enumerate_output_connections():
        src_block_index = result.block_index[src_block]
        src_port_offset = result.first_output_by_block_index[src_block_index]
        src_signal = result.output_to_signal_map[src_port_offset + src_port]
        dest_signal = result.output_to_signal_map[output_port_offset + dest_port]
        assert src_signal == dest_signal, \
            "Output %s(%d):%d is not properly connected to output %s(%d):%d" % (
                src_block.name, src_block_index, src_port,
                block.name, block_index, dest_port
            )


def assert_internal_connections_correct(block: NonLeafBlock,
                                        result: CompiledSystem):
    """
    Raises an AssertionError if an internal connection in the given block is
    not correctly mapped in the given compilation result.

    A connection from a source port to a destination port is correctly mapped
    only if the source and the destination port are mapped to the same signal.
    """
    for src_block, src_port, dest_block, dest_port \
            in block.enumerate_internal_connections():
        src_block_index = result.block_index[src_block]
        dest_block_index = result.block_index[dest_block]
        src_port_offset = result.first_output_by_block_index[src_block_index]
        dest_port_offset = result.first_input_by_block_index[dest_block_index]
        src_signal = result.output_to_signal_map[src_port_offset + src_port]
        dest_signal = result.input_to_signal_map[dest_port_offset + dest_port]
        assert src_signal == dest_signal, \
            "Output %s(%d):%d is not properly connected to input %s(%d):%d" % (
                src_block.name, src_block_index, src_port,
                dest_block.name, dest_block_index, dest_port
            )


def assert_connections_correct(system: NonLeafBlock,
                               compilation_result: CompiledSystem):
    """
    Raises an AssertionError if a connection in the given block is
    not correctly mapped in the given compilation result.

    A connection from a source port to a destination port is correctly mapped
    only if the source and the destination port are mapped to the same signal.
    """
    assert_input_connections_correct(system, compilation_result)
    assert_output_connections_correct(system, compilation_result)
    assert_internal_connections_correct(system, compilation_result)


def assert_execution_order_correct(root: NonLeafBlock,
                                   result: CompiledSystem):
    """
    Raises an AssertionError if the execution order in the compiled system
    is not correct.

    The execution order is not correct if there is a pair of leaf blocks A and B
    in the sequence with A being listed before B and at least one connection
    from an output of B to a fed-through input of A.
    """

    # Determine the execution index for each block
    execution_index = dict(zip(result.leaf_blocks_in_order, itertools.count()))

    # Determine the source block for each signal
    signal_source = result.num_signals * [None]
    signal_source[0:root.num_inputs] = root.num_inputs * [root]
    for block, block_index in result.leaf_block_index.items():
        first_signal = result.first_signal_by_leaf_index[block_index]
        signal_source[first_signal:first_signal + block.num_outputs] = \
            block.num_outputs * [block]

    # Go through all leaf blocks and their fed-through inputs and verify that
    # the source of these inputs is either the root block or a block that
    # comes earlier in the execution sequence.
    for dest_block, dest_block_order in execution_index.items():
        dest_block_index = result.block_index[dest_block]
        input_signal_offset = result.first_input_by_block_index[dest_block_index]
        for input_port in dest_block.feedthrough_inputs:
            input_signal = \
                result.input_to_signal_map[input_signal_offset + input_port]
            src_block = signal_source[input_signal]
            src_block_index = result.block_index[src_block]
            output_signal_offset = \
                result.first_output_by_block_index[src_block_index]
            if src_block is not None and src_block is not root:
                src_block_order = execution_index[src_block]
                assert src_block_order < dest_block_order, \
                    "Input %s(%d):%d is connected to output %s(%d):%d, but " \
                    "the source block is not executed before the " \
                    "destination block" % (
                        dest_block.name, dest_block_index, input_port,
                        src_block.name, src_block_index, input_signal - output_signal_offset
                    )


def assert_initial_condition_correct(result):
    """
    Raises an AssertionError if the initial condition as reported by the
    compilation result does not conform to the initial conditions of the
    individual leaf blocks
    """
    initial_condition = result.initial_condition
    for block, block_index in result.leaf_block_index.items():
        if block.num_states > 0:
            first_state = result.first_state_by_leaf_index[block_index]
            npt.assert_almost_equal(initial_condition[first_state:first_state + block.num_states],
                                    block.initial_condition)


def assert_signal_vector_correct(result, t, state, inputs, signal_vector):
    """
    Raises an AssertionError if the signal vector as reported by the compilation
    result does not conform to the signal vector as expected given the time,
    state and input vector.
    """

    # Check that the original inputs are correctly represented
    npt.assert_almost_equal(signal_vector[0:result.num_inputs],
                            inputs)

    # Check whether the outputs of the leaf blocks are correctly represented
    for block in result.leaf_blocks_in_order:
        if block.num_outputs > 0:
            leaf_block_index = result.leaf_block_index[block]
            block_index = result.block_index[block]

            # Get the state vector for the block
            first_state_index = result.first_state_by_leaf_index[leaf_block_index]
            block_state = state[first_state_index:first_state_index + block.num_states]

            # Get the input vector for the block
            # We use the signal vector as reported by the compiled system
            first_input_index = result.first_input_by_block_index[block_index]
            input_signals = \
                result.input_to_signal_map[first_input_index:
                                           first_input_index + block.num_inputs]
            block_inputs = signal_vector[input_signals]

            # Determine the block outputs
            if block.num_states > 0 and block.num_inputs > 0:
                block_outputs = block.output_function(t,
                                                      block_state,
                                                      block_inputs)
            elif block.num_inputs > 0:
                block_outputs = block.output_function(t,
                                                      block_inputs)
            elif block.num_states > 0:
                block_outputs = block.output_function(t,
                                                      block_state)
            else:
                block_outputs = block.output_function(t)

            # Check whether the outputs of the block correspond to the
            # signals in the signals vector
            first_output_index = result.first_output_by_block_index[block_index]
            output_signals = \
                result.output_to_signal_map[first_output_index:
                                            first_output_index + block.num_outputs]
            npt.assert_almost_equal(block_outputs, signal_vector[output_signals])


def assert_state_update_vector_correct(result,
                                       t,
                                       state,
                                       signal_vector,
                                       state_update_vector):
    """
    Raises an AssertionError if the state update vector as reported by the
    compilation result does not conform to the state update vector as expected
    given the time, state and signal vector.
    """

    # Check whether the results from the leaf blocks are correctly represented
    for block in result.leaf_blocks_in_order:
        if block.num_states > 0:
            leaf_block_index = result.leaf_block_index[block]
            block_index = result.block_index[block]

            # Get the state vector for the block
            first_state_index = result.first_state_by_leaf_index[leaf_block_index]
            block_state = state[first_state_index:first_state_index + block.num_states]

            # Get the input vector for the block
            # We use the signal vector as reported by the compiled system
            first_input_index = result.first_input_by_block_index[block_index]
            input_signals = \
                result.input_to_signal_map[first_input_index:
                                           first_input_index + block.num_inputs]
            block_inputs = signal_vector[input_signals]

            # Determine the block state update vectors
            if block.num_inputs > 0:
                block_state_derivative = block.state_update_function(t,
                                                                     block_state,
                                                                     block_inputs)
            else:
                block_state_derivative = block.state_update_function(t,
                                                                     block_state)

            # Check whether the results of the block correspond to the
            # values in the value vector
            first_state_index = result.first_state_by_leaf_index[leaf_block_index]
            states = range(first_state_index, first_state_index + block.num_states)
            npt.assert_almost_equal(block_state_derivative,
                                    state_update_vector[states])


def assert_output_vector_correct(result,
                                 signal_vector,
                                 output_vector):
    """
    Raises an AssertionError if the output vector as reported by the compilation
    result does not conform to the respective elements of the signal vector as
    expected.
    """

    root_idx = result.block_index[result.root]
    first_output_index = result.first_output_by_block_index[root_idx]
    output_signals = \
        result.output_to_signal_map[first_output_index:
                                    first_output_index + result.root.num_outputs]
    npt.assert_almost_equal(output_vector,
                            signal_vector[output_signals])


def assert_event_vector_correct(result, t, state, signal_vector, event_vector):
    """
    Raises an AssertionError if the event vector as reported by the compilation
    result does not conform to the respective elements of the event vector as
    expected given the state and signal vector.
    """

    # Check whether the event values of the leaf blocks are correctly represented
    for block in result.leaf_blocks_in_order:
        if block.num_events > 0:
            leaf_block_index = result.leaf_block_index[block]
            block_index = result.block_index[block]

            # Get the state vector for the block
            first_state_index = result.first_state_by_leaf_index[leaf_block_index]
            block_state = state[first_state_index:first_state_index + block.num_states]

            # Get the input vector for the block
            # We use the signal vector as reported by the compiled system
            first_input_index = result.first_input_by_block_index[block_index]
            input_signals = \
                result.input_to_signal_map[first_input_index:
                                           first_input_index + block.num_inputs]
            block_inputs = signal_vector[input_signals]

            # Determine the block event values
            if block.num_states > 0 and block.num_inputs > 0:
                block_events = block.event_function(t,
                                                    block_state,
                                                    block_inputs)
            elif block.num_inputs > 0:
                block_events = block.event_function(t,
                                                    block_inputs)
            elif block.num_states > 0:
                block_events = block.event_function(t,
                                                    block_state)
            else:
                block_events = block.event_function(t)

            # Check whether the event of the block correspond to the
            # values in the global event vector
            first_event_index = \
                result.first_event_by_leaf_index[leaf_block_index]
            npt.assert_almost_equal(block_events,
                                    event_vector[first_event_index:
                                                 first_event_index + block.num_events])


def assert_event_update_correct(result, t, state, signal_vector, new_state_vector):
    """
    Raises an AssertionError if the updated state vector as reported by the
    compilation result does not conform to the respective elements of the
    updated state vectors expected given the state and input vector.
    """

    # Check whether the event values of the leaf blocks are correctly represented
    for block in result.leaf_blocks_in_order:
        if block.num_events > 0 and block.num_states > 0:
            leaf_block_index = result.leaf_block_index[block]
            block_index = result.block_index[block]

            # Get the state vector for the block
            first_state_index = result.first_state_by_leaf_index[leaf_block_index]
            block_state = state[first_state_index:first_state_index + block.num_states]

            # Get the input vector for the block
            # We use the signal vector as reported by the compiled system
            first_input_index = result.first_input_by_block_index[block_index]
            input_signals = \
                result.input_to_signal_map[first_input_index:
                                           first_input_index + block.num_inputs]
            block_inputs = signal_vector[input_signals]

            # Determine the block event values
            if block.num_states > 0 and block.num_inputs > 0:
                new_block_state = block.update_state_function(t,
                                                              block_state,
                                                              block_inputs)
            elif block.num_inputs > 0:
                new_block_state = block.update_state_function(t,
                                                              block_inputs)
            elif block.num_states > 0:
                new_block_state = block.update_state_function(t,
                                                              block_state)
            else:
                new_block_state = block.update_state_function(t)

            # Check whether the event of the block correspond to the
            # values in the global event vector
            first_state_index = \
                result.first_state_by_leaf_index[leaf_block_index]
            npt.assert_almost_equal(new_block_state,
                                    new_state_vector[first_state_index:
                                                     first_state_index + block.num_states])

def assert_vectors_correct(result, t, state, inputs):
    """
    Raises an AssertionError if and of the vectors as reported by the
    compilation result does not conform to the respective elements of the
    respective vector of the the individual leaf blocks.
    """

    if result.num_inputs > 0 and result.num_states > 0:
        signal_vector = result.signal_function(t, state, inputs)
        state_update_vector = result.state_update_function(t, state, inputs)
        output_vector = result.output_function(t, state, inputs)
        event_vector = result.event_function(t, state, inputs)
        new_state_vector = result.update_state_function(t, state, inputs)
    elif result.num_inputs > 0:
        signal_vector = result.signal_function(t, inputs)
        state_update_vector = result.state_update_function(t, inputs)
        output_vector = result.output_function(t, inputs)
        event_vector = result.event_function(t, inputs)
        new_state_vector = result.update_state_function(t, inputs)
    elif result.num_states > 0:
        signal_vector = result.signal_function(t, state)
        state_update_vector = result.state_update_function(t, state)
        output_vector = result.output_function(t, state)
        event_vector = result.event_function(t, state)
        new_state_vector = result.update_state_function(t, state)
    else:
        signal_vector = result.signal_function(t)
        state_update_vector = result.state_update_function(t)
        output_vector = result.output_function(t)
        event_vector = result.event_function(t)
        new_state_vector = result.update_state_function(t)

    assert_signal_vector_correct(result,
                                 t,
                                 state,
                                 inputs,
                                 signal_vector)
    assert_state_update_vector_correct(result,
                                       t,
                                       state,
                                       signal_vector,
                                       state_update_vector)
    assert_output_vector_correct(result,
                                 signal_vector,
                                 output_vector)
    assert_event_vector_correct(result,
                                t,
                                state,
                                signal_vector,
                                event_vector)
    assert_event_update_correct(result,
                                t,
                                state,
                                signal_vector,
                                new_state_vector)


@pytest.mark.parametrize(
    "model",
    [
        Constant(value=10.0),   # a model with no inputs and states
        propeller_model(),      # a model with inputs and no states
        engine_model(),         # a model with inputs, outputs and states
        dcmotor_model(),        # a model with no inputs, but outputs and states
        bouncing_ball_model()   # a model with events
    ],
    ids=[
        'Constant',
        'propeller_model',
        'engine_model',
        'dcmotor_model',
        'bouncing_ball_model'
    ]
)
def test_compile(model):
    compiler = Compiler(model)
    result = compiler.compile()

    # Check counts of internal states, events and signals
    total_states = \
        sum(block.num_states
            for block in model.enumerate_leaf_blocks())
    total_signals = model.num_inputs + \
                    sum(block.num_outputs
                        for block in model.enumerate_leaf_blocks())
    total_events = \
        sum(block.num_events
            for block in model.enumerate_leaf_blocks())
    assert result.num_states == total_states
    assert result.num_signals == total_signals
    assert result.num_events == total_events

    # Check number of inputs and outputs
    assert result.num_inputs == model.num_inputs
    assert result.num_outputs == model.num_outputs

    # Check connections
    for block in result.block_index.keys():
        if block not in result.leaf_block_index:
            assert_connections_correct(block, result)

    # Check execution order
    assert_execution_order_correct(model, result)

    # Check for correct composition of the initial state vector
    assert_initial_condition_correct(result)

    # Check for correct composition of the different vectors reported by the
    # compilation result.
    # We use a number of random samples to probabilistically ensure the
    # equality.
    rng = np_rand.default_rng()
    for test_idx in range(100):
        t = rng.standard_normal()
        state = rng.standard_normal(result.num_states)
        inputs = rng.standard_normal(result.num_inputs)

        assert_vectors_correct(result, t, state, inputs)


@pytest.mark.parametrize(
    "model",
    [dcmotor_model_cyclic()]
)
def test_compile_cyclic(model):
    # Enforce a cyclic dependency
    compiler = Compiler(model)
    with pytest.raises(ValueError):
        compiler.compile()
