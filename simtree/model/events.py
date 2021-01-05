"""
Provides classes for defining events.
"""


class Event:
    def __init__(self, owner, event_function, update_function=None):
        self.owner = owner
        self.event_function = event_function
        self.update_function = update_function
        self.event_index = self.owner.system.register_event(self)
