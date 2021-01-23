"""
Provides classes for defining events and event ports.
"""
from abc import ABC
from math import ceil


class MultipleEventSourcesError(RuntimeError):
    """An exception raised when two ports connected to different clocks
    shall be connected to each other."""


class EventPort:
    """An event port is a port that can be connected to other event ports."""

    def __init__(self, owner):
        self.owner = owner
        self._reference = self
        self._listeners = set()

    @property
    def reference(self):
        """The event port that is referenced by connection to this event
        port."""
        if self._reference is not self:
            self._reference = self._reference.reference
        return self._reference

    @reference.setter
    def reference(self, new_reference):
        self._reference = new_reference

    @property
    def source(self):
        """The event source this port is connected to or ``None`` if it is
        not connected to any event source"""

        if self.reference == self:
            return None
        return self.reference.source

    @property
    def listeners(self):
        """The listeners registered on this event port"""
        if self.reference == self:
            return self._listeners
        return self.reference.listeners

    def connect(self, other):
        """Connect an event port to another event port.

        Args:
            other: The other event port

        Raises:
            MultipleEventSourcesError: raised when two ports that are already
                connected to two different sources shall be connected to each
                other
        """

        if self.source is not None and other.source is not None:
            # Both ports are already connected to an event.
            # It is an error if it's not the same event.
            if self.source != other.source:
                raise MultipleEventSourcesError()
        else:
            # At least one of the ports is not yet connected to an event.
            # We select a common reference and join all the listeners connected
            # to both ports in one set.
            if other.source is not None:
                # The other port is already connected to an event,
                # so we choose the other port as reference and add
                # our listeners to its listeners.
                other.listeners.update(self.listeners)
                self.reference.reference = other.reference
            else:
                # The other part is not yet connected to a event,
                # so we make ourselves the reference and add
                # its listeners to our listeners.
                self.listeners.update(other.listeners)
                other.reference.reference = self.reference

    def register_listener(self, listener):
        """Register a listener for this event port.

        Args:
          listener: The listener to register
        """
        self.listeners.add(listener)


class AbstractEventSource(EventPort, ABC):
    """An event source defines the circumstances under which an event occurs.
    Events are occurrences of special occurrence that may require a reaction.

    Events can be reacted upon by updating the state of the system. For this
    purpose, event listeners can be registered which are called upon occurrence
    of the event.

    ``AbstractEventSource`` is the abstract base class for all event sources."""

    @property
    def source(self):
        """An event source always has itself as source"""
        return self


class ZeroCrossEventSource(AbstractEventSource):
    """A ``ZeroCrossEventSource`` defines an event source by the change of sign
    of a special event function. Such zero-cross events are specifically
    monitored and the values of event functions are recorded by the simulator.
    """

    def __init__(self, owner, event_function, direction=0):
        """
        Create a new zero-crossing event-source.

        Args:
            owner: The system or block this event belongs to
            event_function: The callable used to calculate the value of the
                event function
            direction: The direction of the sign change to consider.
                Possible values:

                ``1``
                    Consider only changes from negative to positive

                ``-1``
                    Consider only changes from positive to negative

                ``0`` (default)
                    Consider all changes
        """

        AbstractEventSource.__init__(self, owner)
        self.event_function = event_function
        self.direction = direction
        self.event_index = self.owner.system.register_event(self)


class Clock(AbstractEventSource):
    """A clock is an event source that generates a periodic event."""

    def __init__(self,
                 owner,
                 period,
                 start_time=0.0,
                 end_time=None,
                 run_before_start=False):
        """
        Construct a clock.

        The clock generates a periodic tick occurring at multiples of
        ``period`` offset by ``start_time``. If ``end_time`` is set to
        a value other than ``None``, no ticks will be generated after
        ``end_time``. If ``run_before_start`` is set to ``True``, the
        clock will also generate ticks before the time defined by
        ``start_time``.

        Args:
            owner: The owner object of this clock (a system or a block)
            period: The period of the clock
            start_time: The start time of the clock (default: 0)
            end_time: The end time of the clock (default: ``None``)
            run_before_start: Flag indicating whether the clock shall already
                run before the start time (default: ``False``)
        """

        AbstractEventSource.__init__(self, owner)
        self.period = period
        self.start_time = start_time
        self.end_time = end_time
        self.run_before_start = run_before_start

        self.owner.system.register_clock(self)

    def tick_generator(self, not_before):
        """Return a generate that will yield the times of the ticks of
        this clock.

        Args:
          not_before: The ticks shown shall not be before the given time

        Returns:
          A generator for ticks
        """

        k = ceil((not_before - self.start_time) / self.period)
        if k < 0 and not self.run_before_start:
            # No ticks before the start
            k = 0

        tick_time = self.start_time + k * self.period
        while self.end_time is None or tick_time <= self.end_time:
            yield tick_time
            k += 1
            tick_time = self.start_time + k * self.period
