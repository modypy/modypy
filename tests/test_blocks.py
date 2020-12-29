import pytest
from simtree.blocks import LeafBlock, NonLeafBlock


def test_block_interface():
    child_a = LeafBlock(num_inputs=2,
                        num_states=1,
                        num_outputs=2,
                        initial_condition=[10.])
    child_b = LeafBlock(num_inputs=2, num_outputs=2, feedthrough_inputs=[0])
    parent = NonLeafBlock(children=[child_a, child_b],
                          num_inputs=1,
                          num_outputs=1)
    root = NonLeafBlock()
    root.add_block(parent)

    child_a_index = parent.child_index[child_a]
    child_b_index = parent.child_index[child_b]

    # Try to add a block a second time
    with pytest.raises(ValueError):
        parent.add_block(child_a)

    # Check enumeration of leaf blocks
    assert set(root.enumerate_leaf_blocks()) == set([child_a, child_b])

    # Internal connections
    # Add a proper internal connection and check for execution
    parent.connect(child_a, 0, child_b, 0)
    assert parent.connection_source[parent.first_destination_index[child_b_index] + 0] == \
           parent.first_source_index[child_a_index] + 0

    # Try to connect an invalid source to a valid destination
    with pytest.raises(ValueError):
        parent.connect(child_a, 2, child_b, 0)

    # Try to connect an invalid destination
    with pytest.raises(ValueError):
        parent.connect(child_a, 0, child_b, 2)

    # Check the list of internal connections
    assert list(parent.enumerate_internal_connections()) == \
           [(child_a, 0, child_b, 0)]

    # External input connections
    # Add a proper input connection and check for execution
    parent.connect_input(0, child_a, 1)
    assert parent.connection_source[parent.first_destination_index[child_a_index] + 1] == 0

    # Try to connect an invalid source to a valid destination
    with pytest.raises(ValueError):
        parent.connect_input(2, child_a, 0)

    # Try to connect a valid source to an invalid destination
    with pytest.raises(ValueError):
        parent.connect_input(0, child_a, 2)

    # Check the list of input connections
    assert list(parent.enumerate_input_connections()) == \
           [(0, child_a, 1)]

    # External output connections
    # Add a proper output connection and check for execution
    parent.connect_output(child_b, 1, 0)
    assert parent.connection_source[0] == \
           parent.first_source_index[child_b_index] + 1

    # Try to connect an invalid source to a valid destination
    with pytest.raises(ValueError):
        parent.connect_output(child_a, 2, 0)

    # Try to connect a valid source to an invalid destination
    with pytest.raises(ValueError):
        parent.connect_output(child_a, 0, 2)

    # Check the list of output connections
    assert list(parent.enumerate_output_connections()) == \
           [(child_b, 1, 0)]
