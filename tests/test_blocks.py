import pytest
from simtree.blocks import LeafBlock, NonLeafBlock
from fixtures.models import dcmotor_model


def test_multiple_block_addition():
    child = LeafBlock()
    sys = NonLeafBlock()
    sys.add_block(child)
    with pytest.raises(ValueError):
        sys.add_block(child)


def test_int_conn_invalid_source(dcmotor_model):
    with pytest.raises(ValueError):
        dcmotor_model.system.connect(
            dcmotor_model.voltage, 2, dcmotor_model.engine, 0)


def test_int_conn_invalid_destination(dcmotor_model):
    with pytest.raises(ValueError):
        dcmotor_model.system.connect(
            dcmotor_model.voltage, 0, dcmotor_model.engine, 2)


def test_input_conn_invalid_source(dcmotor_model):
    with pytest.raises(ValueError):
        dcmotor_model.engine.connect_input(2, dcmotor_model.dcmotor, 0)


def test_input_conn_invalid_destination(dcmotor_model):
    with pytest.raises(ValueError):
        dcmotor_model.engine.connect_input(0, dcmotor_model.dcmotor, 2)


def test_output_conn_invalid_source(dcmotor_model):
    with pytest.raises(ValueError):
        dcmotor_model.engine.connect_output(dcmotor_model.dcmotor, 2, 0)


def test_output_conn_invalid_destination(dcmotor_model):
    with pytest.raises(ValueError):
        dcmotor_model.engine.connect_output(dcmotor_model.dcmotor, 1, 2)


def test_enumerate_leaf_blocks(dcmotor_model):
    assert dcmotor_model.leaf_blocks == set(
        dcmotor_model.system.enumerate_leaf_blocks())


def test_enumerate_internal_connections(dcmotor_model):
    int_conns = set(dcmotor_model.engine.enumerate_internal_connections())
    assert int_conns == set([(dcmotor_model.dcmotor, 0, dcmotor_model.static_propeller, 0),
                             (dcmotor_model.static_propeller, 1, dcmotor_model.dcmotor, 1)])


def test_enumerate_input_connections(dcmotor_model):
    input_conns = set(dcmotor_model.engine.enumerate_input_connections())
    assert input_conns == set(
        [(0, dcmotor_model.dcmotor, 0), (1, dcmotor_model.static_propeller, 1)])


def test_enumerate_output_connections(dcmotor_model):
    output_conns = set(dcmotor_model.engine.enumerate_output_connections())
    assert output_conns == set(
        [(dcmotor_model.static_propeller, 0, 0), (dcmotor_model.dcmotor, 1, 1)])
