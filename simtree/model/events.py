"""
Provides classes for defining events.
"""


class Event:
    """
    An event defines the occurrence of specific circumstances during execution
    which are of special importance and shall be explicitly recorded.

    Events are detected by the simulator by monitoring a special event function
    for a change in sign. When this change in sign happens the event is said to
    have occurred. The simulator will specifically mark that point in the
    trajectory.

    Events can also be reacted upon by updating the state of the system. In that
    case, a handler function can be specified which will modify the state. After
    that, simulation can continue with the updated state.
    """
    def __init__(self, owner, event_function, update_function=None):
        self.owner = owner
        self.event_function = event_function
        self.update_function = update_function
        self.event_index = self.owner.system.register_event(self)
