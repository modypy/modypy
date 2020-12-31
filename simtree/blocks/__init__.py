import bisect
import itertools

import numpy as np


class Block:
    def __init__(self,
                 num_inputs=0,
                 num_outputs=0,
                 name=None,
                 feedthrough_inputs=None):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.name = name
        if feedthrough_inputs is None:
            feedthrough_inputs = range(self.num_inputs)
        self.feedthrough_inputs = feedthrough_inputs

    def enumerate_leaf_blocks(self):
        """
        Yield all non-virtual blocks contained in this block.
        """
        yield self


class NonLeafBlock(Block):
    def __init__(self,
                 children=None,
                 **kwargs):
        Block.__init__(self, **kwargs)

        # List of children contained in this block
        self.children = []
        # Map of child indices by block object
        self.child_index = {}

        # Start indices for connection sources
        # Indices 0...num_inputs-1 are reserved for parent block inputs.
        # Indices >=num_inputs are used for outputs of contained blocks.
        self.first_source_index = [self.num_inputs]

        # Start indices for connection destinations
        # Indices 0...num_outputs-1 are reserved for parent block outputs.
        # Indices >= num_outputs are used for inputs of contained blocks.
        self.first_destination_index = [self.num_outputs]

        # Connection sources by connection destination
        self.connection_source = [None]*self.num_outputs

        if children is not None:
            for child in children:
                self.add_block(child)

    def add_block(self, child):
        """
        Add a child block to this non-leaf block.
        """

        if child in self.child_index:
            raise ValueError("Child is already contained in this non-leaf block")

        # Add the child to the list of children
        index = len(self.children)
        self.children.append(child)
        self.child_index[child] = index

        # Set up source and destination indices
        self.first_source_index.append(
            self.first_source_index[-1]+child.num_outputs)
        self.first_destination_index.append(
            self.first_destination_index[-1]+child.num_inputs)
        self.connection_source.extend([None]*child.num_inputs)

    def connect(self, src, src_ports, dest, dest_ports):
        """
        Connect ports of internal blocks to each other.

        :param src: The source block.
        :param src_ports: The source port index or iterable of source port indices.
        :param dest: The destination block.
        :param dest_ports: The destination port index or iterable of destination
            port indices.
        """

        if isinstance(src_ports, int):
            src_ports = [src_ports]
        if isinstance(dest_ports, int):
            dest_ports = [dest_ports]

        src_block_index = self.child_index[src]
        dest_block_index = self.child_index[dest]

        for src_port, dest_port in zip(src_ports, dest_ports):
            if not 0 <= src_port < src.num_outputs:
                raise ValueError("Invalid source output index %d" % src_port)
            if not 0 <= dest_port < dest.num_inputs:
                raise ValueError("Invalid destination input index %d" % dest_port)

            src_port_index = self.first_source_index[src_block_index] + \
                src_port
            dest_port_index = self.first_destination_index[dest_block_index] + \
                dest_port

            self.connection_source[dest_port_index] = src_port_index

    def connect_input(self, input_ports, dest, dest_ports):
        """
        Connect inputs of this block to inputs of a contained block.

        :param input_ports: The input port index or iterable of input port indices.
        :param dest: The destination block.
        :param dest_ports: The destination port index or iterable of destination
            port indices.
        """

        if isinstance(input_ports, int):
            input_ports = [input_ports]
        if isinstance(dest_ports, int):
            dest_ports = [dest_ports]

        dest_block_index = self.child_index[dest]

        for input_port, dest_port in zip(input_ports, dest_ports):
            if not 0 <= input_port < self.num_inputs:
                raise ValueError("Invalid source input index %d" % input_port)
            if not 0 <= dest_port < dest.num_inputs:
                raise ValueError("Invalid destination input index %d" % dest_port)

            dest_port_index = self.first_destination_index[dest_block_index] + \
                dest_port

            self.connection_source[dest_port_index] = input_port

    def connect_output(self, src, src_ports, output_ports):
        """
        Connect outputs of a contained block to outputs of this block.

        :param src: The source block.
        :param src_ports: The source port index or iterable of source port indices.
        :param output_ports: The output port index or iterable of output port indices.
        """

        if isinstance(src_ports, int):
            src_ports = [src_ports]
        if isinstance(output_ports, int):
            output_ports = [output_ports]

        src_block_index = self.child_index[src]

        for src_port, output_port in zip(src_ports, output_ports):
            if not 0 <= src_port < src.num_outputs:
                raise ValueError("Invalid source output index %d" % src_port)
            if not 0 <= output_port < self.num_outputs:
                raise ValueError("Invalid destination output index %d" % output_port)

            src_port_index = self.first_source_index[src_block_index] + \
                src_port

            self.connection_source[output_port] = src_port_index

    def enumerate_internal_connections(self):
        """
        Yield all internal connections.

        Each connection is represented by a tuple (src_block,src_port,dest_block,dest_port), with

         - src_block being the block providing the source port,
         - src_port being the index of the output on the source block,
         - dest_block being the block receiving the signal, and
         - dest_port being the index of the input on the destination block.
        """

        for dest_block, first_dest_index in \
                zip(self.children,
                    self.first_destination_index):
            for dest_port_index, src_index in \
                    zip(range(dest_block.num_inputs),
                        self.connection_source[first_dest_index:]):
                if src_index is not None and src_index >= self.num_inputs:
                    # This is a connection from an internal block
                    src_block_index = bisect.bisect_right(
                        self.first_source_index, src_index)-1
                    src_block = self.children[src_block_index]
                    src_port_index = src_index - \
                        self.first_source_index[src_block_index]
                    yield src_block, src_port_index, dest_block, dest_port_index

    def enumerate_input_connections(self):
        """
        Yield all input connections.

        Each connection is represented by a tuple (src_port,dest_block,dest_port), with

         - src_port being the index of the input on the containing block,
         - dest_block being the block receiving the signal, and
         - dest_port being the index of the input on the destination block.
        """

        for dest_block, first_dest_index in \
                zip(self.children,
                    self.first_destination_index):
            for dest_port_index, src_index in \
                    zip(range(dest_block.num_inputs),
                        self.connection_source[first_dest_index:]):
                if src_index is not None and src_index < self.num_inputs:
                    # This is a connection from an external input
                    yield src_index, dest_block, dest_port_index

    def enumerate_output_connections(self):
        """
        Yield all output connections.

        Each connection is represented by a tuple (src_block,src_port,dest_port), with

         - src_block being the block providing the source port,
         - src_port being the index of the output on the source block,
         - dest_port being the index of the output on the containing block.
        """

        for dest_port_index, src_index in zip(range(self.num_outputs), self.connection_source):
            if src_index is not None:
                src_block_index = bisect.bisect_right(
                    self.first_source_index, src_index)-1
                src_block = self.children[src_block_index]
                src_port_index = src_index-self.first_source_index[src_block_index]
                yield src_block, src_port_index, dest_port_index

    def enumerate_leaf_blocks(self):
        for child_block in self.children:
            yield from child_block.enumerate_leaf_blocks()


class LeafBlock(Block):
    def __init__(self,
                 num_states=0,
                 num_events=0,
                 initial_condition=None,
                 **kwargs):
        Block.__init__(self, **kwargs)
        self.num_states = num_states
        self.num_events = num_events
        if initial_condition is None:
            initial_condition = np.zeros(self.num_states)
        self.initial_condition = initial_condition
