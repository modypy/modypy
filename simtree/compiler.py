import itertools
import numpy as np


class CompiledSystem:
    """
    A block provided by the compiler.

    The block provides implementations of `output_function` and `state_update_function`
    as well properties `num_outputs` and `num_states` for the compiled block.
    It can be used as input for `simtree.simulator.Simulator`.
    """

    def __init__(self,
                 root,
                 leaf_blocks_in_order,
                 leaf_block_index,
                 block_index,

                 first_state_by_leaf_index,
                 first_event_by_leaf_index,
                 first_signal_by_leaf_index,

                 first_input_by_block_index,
                 first_output_by_block_index,

                 input_to_signal_map,
                 output_to_signal_map):
        """
        Constructor for the compiled block:.

        root: block
            Root block of this block.
        leaf_blocks_in_order: list
            List of the leaf blocks contained in this block, in the order they need to be executed.
        leaf_block_index: dictionary
            Dictionary mapping leaf blocks to their leaf block index.
        block_index: dictionary
            Dictionary mapping blocks to their block index.
        first_state_by_leaf_index: list-like of integer
            List of the respective first state index for each of the leaf systems
            based on the leaf block index.
        first_event_by_leaf_index: list-like of integer
            List of the respective first event index for each of the leaf systems
            based on the leaf block index.
        first_signal_by_leaf_index: list-like of integer
            List of the respective first (output) signal index for each of the
            leaf systems based on the leaf block index.
        first_input_by_block_index: list-like of integer
            List of the respective first input index for each of the leaf systems
            based on the block index.
        first_output_by_block_index: list-like of integer
            List of the respective first output index for each of the leaf systems
            based on the block index.
        input_to_signal_map: list-like of integer
            List of the signals mapped to the inputs.
        output_to_signal_map: list-like of integer
            List of the signals mapped to the outputs.
        """

        self.root = root
        self.leaf_blocks_in_order = leaf_blocks_in_order
        self.leaf_block_index = leaf_block_index
        self.block_index = block_index

        self.first_state_by_leaf_index = first_state_by_leaf_index
        self.first_event_by_leaf_index = first_event_by_leaf_index
        self.first_signal_by_leaf_index = first_signal_by_leaf_index

        self.first_input_by_block_index = first_input_by_block_index
        self.first_output_by_block_index = first_output_by_block_index

        self.input_to_signal_map = input_to_signal_map
        self.output_to_signal_map = output_to_signal_map

        # This combined block exposes the interface of the root block
        self.num_inputs = self.root.num_inputs
        self.num_outputs = self.root.num_outputs

        # Set up information about the states and events
        self.num_states = self.first_state_by_leaf_index[-1]
        self.num_events = self.first_event_by_leaf_index[-1]
        self.num_signals = self.first_signal_by_leaf_index[-1]

        # Initialise the global initial condition
        self.initial_condition = np.zeros(self.num_states)
        for block, leaf_block_index in self.leaf_block_index.items():
            first_state_index = self.first_state_by_leaf_index[leaf_block_index]
            self.initial_condition[first_state_index:first_state_index + block.num_states] = \
                block.initial_condition

    def signal_function(self, t, *args):
        """
        Calculate the value of all signals in the block.

        If the block has a state or inputs, accepts a state vector and/or an
        input vector as additional parameters - in that order.
        """

        if self.num_inputs > 0 and self.num_states > 0:
            states, inputs = args
        elif self.num_states > 0:
            states = args[0]
            inputs = []
        elif self.num_inputs > 0:
            states = []
            inputs = args[0]
        else:
            states = []
            inputs = []

        signals = np.zeros(self.num_signals)

        # The inputs are mapped to the signals representing the inputs to
        # the root block
        root_index = self.block_index[self.root]
        first_root_input = self.first_input_by_block_index[root_index]
        root_input_signals = \
            self.input_to_signal_map[first_root_input:first_root_input + self.root.num_inputs]
        signals[root_input_signals] = inputs

        # Calculate the signal vector
        for block in self.leaf_blocks_in_order:
            # We also process blocks without outputs. These may be sinks.
            leaf_block_index = self.leaf_block_index[block]
            block_index = self.block_index[block]

            # Gather the input vector for the block
            first_input = self.first_input_by_block_index[block_index]
            input_signals = \
                self.input_to_signal_map[first_input:first_input + block.num_inputs]
            block_inputs = signals[input_signals]

            # Gather the state vector for the block
            first_state = self.first_state_by_leaf_index[leaf_block_index]
            block_states = states[first_state:first_state + block.num_states]

            if block.num_inputs > 0 and block.num_states > 0:
                # This block has inputs and states
                block_outputs = block.output_function(t, block_states, block_inputs)
            elif block.num_states > 0:
                # This block has states, but no inputs
                block_outputs = block.output_function(t, block_states)
            elif block.num_inputs > 0:
                # This block has inputs, but no states
                block_outputs = block.output_function(t, block_inputs)
            else:
                # This block neither has inputs nor states
                block_outputs = block.output_function(t)

            # Write the output of the block into the respective signals
            first_signal = self.first_signal_by_leaf_index[leaf_block_index]
            signals[first_signal:first_signal + block.num_outputs] = block_outputs

        return signals

    def output_function(self, t, *args):
        """
        Output function for the root block.

        This calculates the output vector for the root block only.
        """

        if self.num_outputs == 0:
            return []

        signals = self.signal_function(t, *args)

        # Determine the output signals for the root block
        root_idx = self.block_index[self.root]
        first_output_signal = self.first_output_by_block_index[root_idx]
        output_signals = \
            self.output_to_signal_map[first_output_signal:
                                      first_output_signal + self.root.num_outputs]
        outputs = signals[output_signals]

        return outputs

    def state_update_function(self, t, *args):
        """
        Combined state update function of the compiled block.

        This calculates the state update vector for all the leaf blocks contained in the block.
        """

        if self.num_states > 0:
            states = args[0]
        else:
            states = []

        # Calculate the values of all signals
        signals = self.signal_function(t, *args)
        # Calculate the value of the state derivatives
        state_derivative = np.zeros(self.num_states)
        for block, leaf_block_index in self.leaf_block_index.items():
            # We only consider blocks that have state
            if block.num_states > 0:
                block_index = self.block_index[block]

                # Gather the input vector for the block
                first_input = self.first_input_by_block_index[block_index]
                input_signals = \
                    self.input_to_signal_map[first_input:first_input + block.num_inputs]
                block_inputs = signals[input_signals]

                # Gather the state vector for the block
                first_state = self.first_state_by_leaf_index[leaf_block_index]
                block_states = states[first_state:first_state + block.num_states]

                if block.num_inputs > 0:
                    # This block has inputs
                    block_state_derivative = \
                        block.state_update_function(t, block_states, block_inputs)
                else:
                    # This block has no inputs
                    block_state_derivative = \
                        block.state_update_function(t, block_states)

                # Write the state derivative of the block into the respective
                # elements of the state derivative vector
                state_derivative[first_state:first_state + block.num_states] = \
                    block_state_derivative

        return state_derivative

    def event_function(self, t, *args):
        """
        Combined event function for the compiled block.

        This calculates the event vector for all the leaf blocks contained in the block.
        """

        if self.num_states > 0:
            states = args[0]
        else:
            states = []

        # Calculate the values of all signals
        signals = self.signal_function(t, *args)

        # Determine the value of the event vector
        event_values = np.zeros(self.num_events)
        for block, leaf_block_index in self.leaf_block_index.items():
            # We only consider blocks that have events
            if block.num_events > 0:
                block_index = self.block_index[block]

                # Gather the input vector for the block
                first_input = self.first_input_by_block_index[block_index]
                input_signals = \
                    self.input_to_signal_map[first_input:first_input + block.num_inputs]
                block_inputs = signals[input_signals]

                # Gather the state vector for the block
                first_state = self.first_state_by_leaf_index[leaf_block_index]
                block_states = states[first_state:first_state + block.num_states]

                if block.num_inputs > 0 and block.num_states > 0:
                    # This block has inputs and states
                    block_event_values = \
                        block.event_function(t, block_states, block_inputs)
                elif block.num_states > 0:
                    # This block has states, but no inputs
                    block_event_values = block.event_function(t, block_states)
                elif block.num_inputs > 0:
                    # This block has inputs, but no states
                    block_event_values = block.event_function(t, block_inputs)
                else:
                    # This block neither has inputs nor states
                    block_event_values = block.event_function(t)

                # Write the event values of the block to the global event vector
                first_event = self.first_event_by_leaf_index[leaf_block_index]
                event_values[first_event:first_event + block.num_events] = \
                    block_event_values

        return event_values

    def update_state_function(self, t, *args):
        """
        Combined event handling function for the compiled block.

        This calculates the new state vector after a zero-crossing event.
        """

        if self.num_states > 0:
            states = args[0]
        else:
            states = []

        # Calculate the values of all signals
        signals = self.signal_function(t, *args)

        # Determine the new state vector
        new_state = np.zeros(self.num_states)
        for block, leaf_block_index in self.leaf_block_index.items():
            # We only consider blocks that have events and states
            if block.num_events > 0 and block.num_states > 0:
                block_index = self.block_index[block]

                # Gather the input vector for the block
                first_input = self.first_input_by_block_index[block_index]
                input_signals = \
                    self.input_to_signal_map[first_input:first_input + block.num_inputs]
                block_inputs = signals[input_signals]

                # Gather the state vector for the block
                first_state = self.first_state_by_leaf_index[leaf_block_index]
                block_states = states[first_state:first_state + block.num_states]

                if block.num_inputs > 0:
                    # This block has inputs and states
                    new_block_state = \
                        block.update_state_function(t, block_states, block_inputs)
                else:
                    # This block has states, but no inputs
                    new_block_state = \
                        block.update_state_function(t, block_states)

                # Write the new state into the global state vector
                new_state[first_state:first_state + block.num_states] = \
                    new_block_state

        return new_state


class Compiler:
    """
    Compiler for block trees.

    - Determine leaf blocks.
    - Determines the size and mapping of state, output and event vectors for leaf blocks.
    - Determine mapping of inputs to output vector items for all blocks.
    - Determine execution sequence for leaf blocks.
    """

    def __init__(self, root):
        self.root = root
        # Collect all blocks in the graph and assign block indices.
        # We do the enumeration in pre-order, so that the root block
        # has index 0.
        self.blocks = list(Compiler.enumerate_blocks_pre_order(self.root))
        self.block_index = {block: index for index,
                            block in zip(itertools.count(), self.blocks)}

        # Collect all leaf blocks in the graph and assign leaf block indices.
        self.leaf_blocks = list(self.root.enumerate_leaf_blocks())
        self.leaf_block_index = {block: index for index, block in
                                 zip(itertools.count(), self.leaf_blocks)}

        # Allocate state, signal and event indices for the compiled block.
        # There is exactly one signal for each output of a leaf block.
        # All inputs and the outputs of non-leaf blocks are then mapped to
        # signals.
        self.first_state_by_leaf_index = \
            list(itertools.accumulate([block.num_states for block in self.leaf_blocks],
                                      initial=0))
        self.first_event_by_leaf_index = \
            list(itertools.accumulate([block.num_events for block in self.leaf_blocks],
                                      initial=0))
        # We need to allocate signals for the inputs of the root block
        self.first_signal_by_leaf_index = \
            list(itertools.accumulate([block.num_outputs for block in self.leaf_blocks],
                                      initial=self.root.num_inputs))

        # Allocate input- and output-to-signal mappings for all blocks.
        self.first_input_by_block_index = \
            list(itertools.accumulate([block.num_outputs for block in self.blocks],
                                      initial=0))
        self.first_output_by_block_index = \
            list(itertools.accumulate([block.num_outputs for block in self.blocks],
                                      initial=0))

        # Set up input and output vector maps for all blocks.
        # For each input (adjusted by self.first_input_by_block_index)
        # this gives the index of the signal connected to the input.
        self.input_to_signal_map = [None] * self.first_input_by_block_index[-1]
        # For each output (adjusted by self.first_output_by_block_index)
        # this gives the index of the signal connected to the input.
        self.output_to_signal_map = [None] * self.first_output_by_block_index[-1]

    def compile(self):
        """
        Compile the block graph given as `root`.

        Compilation consists
            - establishment of the input-output connections and
            - determination of an execution order consistent with the
              inter-block dependencies.
        """

        # Establish input-to-output mapping
        self.map_inputs_to_outputs()
        # Establish execution order
        self.build_execution_order()

        return CompiledSystem(root=self.root,
                              leaf_blocks_in_order=self.execution_sequence,
                              leaf_block_index=self.leaf_block_index,
                              block_index=self.block_index,
                              first_state_by_leaf_index=self.first_state_by_leaf_index,
                              first_event_by_leaf_index=self.first_event_by_leaf_index,
                              first_signal_by_leaf_index=self.first_signal_by_leaf_index,

                              first_input_by_block_index=self.first_input_by_block_index,
                              first_output_by_block_index=self.first_output_by_block_index,

                              input_to_signal_map=self.input_to_signal_map,
                              output_to_signal_map=self.output_to_signal_map)

    def map_leaf_block_outputs(self):
        """
        Map the outputs of leaf blocks to the signal vector.
        """

        for block, leaf_block_index in self.leaf_block_index.items():
            block_index = self.block_index[block]
            signal_offset = self.first_signal_by_leaf_index[leaf_block_index]
            output_offset = self.first_output_by_block_index[block_index]
            self.output_to_signal_map[output_offset:output_offset + block.num_outputs] = \
                range(signal_offset, signal_offset + block.num_outputs)

    def map_root_inputs(self):
        """
        Map the inputs of the root block to the signal vector.
        """

        # The first signals are reserved as inputs for the root block
        root_idx = self.block_index[self.root]
        root_input_start = self.first_input_by_block_index[root_idx]
        self.input_to_signal_map[root_input_start:
                                 root_input_start + self.root.num_inputs] = \
            range(self.root.num_inputs)

    def map_nonleaf_block_outputs(self):
        """
        Map the outputs of non-leaf blocks to the signal vector.
        """

        # We iterate over all blocks in the graph in post-order and go through
        # all their outgoing connections, propagating the signal numbers
        # assigned to the source port towards the destination port.
        # Iterating post-order ensures that we encounter leaf blocks first.
        # The output-to-signal-map for the outputs of leaf blocks should
        # already have been populated.
        for block in Compiler.enumerate_blocks_post_order(self.root):
            if block not in self.leaf_block_index:
                block_index = self.block_index[block]
                dest_port_offset = self.first_output_by_block_index[block_index]
                for src_block, src_port_index, output_index in \
                        block.enumerate_output_connections():
                    src_block_index = self.block_index[src_block]
                    src_port_offset = self.first_output_by_block_index[src_block_index]
                    self.output_to_signal_map[dest_port_offset + output_index] = \
                        self.output_to_signal_map[src_port_offset + src_port_index]

    def map_internal_connections(self):
        """
        Map the inputs of blocks based on internal connections.
        """

        # We go over all blocks in the graph and go through all their internal
        # connections, propagating the signal numbers assigned to the source
        # port towards the destination port.
        # The output-to-signal map should have been completely filled already
        # by map_leaf_block_outputs and map_nonleaf_block_outputs.
        for block in Compiler.enumerate_blocks_post_order(self.root):
            if block not in self.leaf_block_index:
                for src_block, src_port_index, dst_block, dest_port_index in \
                        block.enumerate_internal_connections():
                    src_block_index = self.block_index[src_block]
                    src_port_offset = self.first_output_by_block_index[src_block_index]
                    dest_block_index = self.block_index[dst_block]
                    dest_port_offset = self.first_input_by_block_index[dest_block_index]
                    self.input_to_signal_map[dest_port_offset + dest_port_index] = \
                        self.output_to_signal_map[src_port_offset + src_port_index]

    def map_nonleaf_block_inputs(self):
        """
        Map the inputs of blocks based on input connections.
        """

        # We go over all non-leaf blocks and go through all input connections,
        # propagating the signal numbers assigned to the input port towards
        # the destination port.
        for block in Compiler.enumerate_blocks_pre_order(self.root):
            if block not in self.leaf_block_index:
                block_index = self.block_index[block]
                src_port_offset = self.first_input_by_block_index[block_index]
                for input_index, dst_block, dest_port_index in block.enumerate_input_connections():
                    dest_block_index = self.block_index[dst_block]
                    dest_port_offset = self.first_input_by_block_index[dest_block_index]
                    self.input_to_signal_map[dest_port_offset + dest_port_index] = \
                        self.input_to_signal_map[src_port_offset + input_index]

    def map_inputs_to_outputs(self):
        """
        Build the mapping of inputs and outputs to entries of the output vector.
        """

        # Pre-populate for leaf blocks
        self.map_leaf_block_outputs()

        # Pre-populate for inputs of the root block
        self.map_root_inputs()

        # Iterate over all non-leaf blocks and process outgoing connections
        self.map_nonleaf_block_outputs()

        # Iterate over all non-leaf blocks and process internal connections
        self.map_internal_connections()

        # Iterate over all non-leaf blocks and process incoming connections
        self.map_nonleaf_block_inputs()

    def build_execution_order(self):
        """
        Establish an execution order of leaf blocks compatible with inter-block dependencies.
        """

        # We use Kahn's algorithm here
        # Initialize the number of incoming connections by leaf block
        num_incoming = [len(block.feedthrough_inputs) for block in self.leaf_blocks]

        # Initialize the list of leaf blocks that do not require any inputs
        blocks_ready_to_fire = [block
                                for block, num_in in zip(self.leaf_blocks, num_incoming)
                                if num_in == 0]

        def satisfy_signal_users(signal_index_start, signal_index_end):
            for dest, dest_leaf_index in self.leaf_block_index.items():
                dest_index = self.block_index[dest]
                dest_port_offset = self.first_input_by_block_index[dest_index]
                # We consider only ports that feed through
                for dest_port in dest.feedthrough_inputs:
                    dest_port_index = dest_port_offset + dest_port
                    src_signal_index = self.input_to_signal_map[dest_port_index]
                    if signal_index_start <= src_signal_index < signal_index_end:
                        # The destination port is connected to the source signal
                        # There is one less input to be fulfilled on the
                        # destination block
                        num_incoming[dest_leaf_index] = num_incoming[dest_leaf_index] - 1

                        # If all feed-through inputs on the destination block
                        # are satisfied, that block is ready to fire.
                        if num_incoming[dest_leaf_index] == 0:
                            blocks_ready_to_fire.append(dest)

        # The inputs of the root block are considered to be connected
        satisfy_signal_users(0, self.root.num_inputs)

        # Initialize the execution sequence
        execution_sequence = []
        while len(blocks_ready_to_fire) > 0:
            # Take one block that is ready to fire
            src = blocks_ready_to_fire.pop()
            # As it can fire now, we can append it to the execution sequence
            execution_sequence.append(src)
            # Consider the targets of all outgoing connections of src
            src_leaf_index = self.leaf_block_index[src]
            signal_index_start = self.first_signal_by_leaf_index[src_leaf_index]
            signal_index_end = signal_index_start + src.num_outputs
            satisfy_signal_users(signal_index_start, signal_index_end)

        if sum(num_incoming) > 0:
            # There are unsatisfied inputs, which is probably due to cycles in
            # the graph
            unsatisfied_blocks = [block.name for block, inputs in zip(
                self.leaf_blocks, num_incoming) if inputs > 0]
            raise ValueError("Unsatisfied inputs for blocks %s" %
                             unsatisfied_blocks)

        self.execution_sequence = execution_sequence

    @staticmethod
    def enumerate_blocks_pre_order(root):
        """
        Enumerate all blocks in the graph with the given root in pre-order.
        """

        yield root
        try:
            for child in root.children:
                yield from Compiler.enumerate_blocks_pre_order(child)
        except AttributeError:
            # Ignore this, the root is not a non-leaf node
            pass

    @staticmethod
    def enumerate_blocks_post_order(root):
        """
        Enumerate all blocks in the graph with the given root in post-order.
        """

        try:
            for child in root.children:
                yield from Compiler.enumerate_blocks_post_order(child)
        except AttributeError:
            # Ignore this, the root is not a non-leaf node
            pass
        yield root
