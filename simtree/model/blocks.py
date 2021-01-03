"""
Provides classes ``Block`` and ``MetaProperty`` for implementation of the block
graph.
"""
import inspect
from abc import ABC, abstractmethod

from .model_context import BlockContainer


class Block(BlockContainer):
    """
    This class represents a block in a model.

    Each block has a parent, which is either another ``Block`` instance or an
    instance of ``ModelContext``.
    """
    def __init__(self, parent):
        self.parent = parent
        self.context = parent.context

        self.parent.register_block(self)

        self.create_meta_property_instances()

    def create_meta_property_instances(self):
        """Create meta-property instances for this block"""
        for name, value in \
                inspect.getmembers(type(self),
                                   (lambda obj: isinstance(obj, MetaProperty))):

            value.instantiate_for_block(self)


class MetaProperty(ABC):
    """
    This class represents a meta-property of a block class. It is the base class
    for all meta-properties provided in the model.
    """
    def __init__(self):
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    @abstractmethod
    def instantiate_for_block(self, block):
        """Instantiate this meta-property for the given block."""
