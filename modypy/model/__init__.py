"""Declarations for creation of models"""
from .events import AbstractEventSource, EventPort, Clock, ZeroCrossEventSource
from .ports import (
    ShapeType,
    AbstractSignal,
    InputSignal,
    MultipleSignalsError,
    Port,
    PortNotConnectedError,
    ShapeMismatchError,
    Signal,
    signal_function,
    signal_method,
)
from .states import State
from .system import Block, System, SystemState
