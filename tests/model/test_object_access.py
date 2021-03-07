"""Test the object access protocol"""
from unittest.mock import Mock

import pytest

from modypy.model import System, ZeroCrossEventSource, State, Signal, Port, PortNotConnectedError
