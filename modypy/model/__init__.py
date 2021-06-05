"""Declarations for creation of models"""
from .events import Clock, EventPort, ZeroCrossEventSource
from .ports import (
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
