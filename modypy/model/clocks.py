"""
Provides classes for defining clocks.
"""
from math import ceil


class MultipleClocksError(RuntimeError):
    """An exception raised when two ports connected to different clocks
    shall be connected to each other."""


class ClockPort:
    """
    A clock port is a port that can be connected to other clock ports.
    """

    def __init__(self, owner):
        self.owner = owner
        self._reference = self
        self._listeners = set()

    @property
    def reference(self):
        """
        The clock-port that is referenced by connection by this clock port.
        """
        if self._reference is not self:
            self._reference = self._reference.reference
        return self._reference

    @reference.setter
    def reference(self, new_reference):
        self._reference = new_reference

    @property
    def clock(self):
        """The clock this port is connected to or ``None`` if it is
        not connected to any clock"""

        if self.reference == self:
            return None
        return self.reference.clock

    @property
    def listeners(self):
        """The listeners registered on this clock port"""
        if self.reference == self:
            return self._listeners
        return self.reference.listeners

    def connect(self, other):
        """
        Connect a clock port to another clock port.

        :param other: The other clock port
        """

        if self.clock is not None and other.clock is not None:
            # Both ports are already connected to a clock.
            # It is an error if it's not the same port.
            if self.clock != other.clock:
                raise MultipleClocksError()
        else:
            # At least one of the ports is not yet connected to
            # a clock.
            # We select a common reference and join all the
            # listeners connected to both ports in one set.
            if other.clock is not None:
                # The other port is already connected to a clock,
                # so we choose the other port as reference and add
                # our listeners to its listeners.
                other.listeners.update(self.listeners)
                self.reference.reference = other.reference
            else:
                # The other part is not yet connected to a clock,
                # so we make ourselves the reference and add
                # its listeners to our listeners.
                self.listeners.update(other.listeners)
                other.reference.reference = self.reference

    def register_listener(self, listener):
        """
        Register a listener for this clock port.

        :param listener: The listener to register
        """
        self.listeners.add(listener)


class Clock(ClockPort):
    """
    A clock generates a periodic tick event.
    """

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

        :param owner: The owner object of this clock (a system or a block)
        :param period:  The period of the clock
        :param start_time: The start time of the clock (default: 0)
        :param end_time: The end time of the clock (default: ``None``)
        :param run_before_start: Flag indicating whether the clock shall already
            run before the start time (default: ``False``)
        """

        ClockPort.__init__(self, owner)
        self.period = period
        self.start_time = start_time
        self.end_time = end_time
        self.run_before_start = run_before_start

        self.owner.system.register_clock(self)

    @property
    def clock(self):
        """A clock always references itself"""
        return self

    def tick_generator(self, not_before):
        """
        Return a generate that will yield the times of the ticks of
        this clock.

        :param not_before: The ticks shown shall not be before the given
            time
        :return: A generator for ticks
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
