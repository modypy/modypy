"""
Events are instances in time when some specific condition about the system
changes. For example, clocks split the domain of the time variable into a
possibly infinite sequence of continuous and disjoint intervals, each as long
as a period of the clock. Whenever the continuous time variable leaves one of
these intervals - and thereby enters the succeeding interval - a clock tick
occurs.

Similarly, zero-crossing events occur when the sign of a specific event function
changes. That event function may depend on the time and on the value of signals
and states.

Listeners are special functions that may change the value of states in the
system. They may be bound to events, meaning that they are executed whenever the
respective event occurs. Note that any listener may be bound to multiple events.

As each event may have multiple listeners bound to it, each occurrence of an
event may lead to multiple listeners being executed. Similarly, multiple events
may occur at any point in time, also possibly leading to multiple listeners
being executed.

The order of execution of listeners in this situation is undefined. Thus,
model developers should make sure that listeners acting on the same parts of the
state are confluent, i.e., that the final states resulting from different orders
of execution of listeners are equivalent to each other. What is considered
`equivalent` may depend on the application.

Further, listeners may change the state in such a way that the sign of the
event function of a zero-crossing event changes. Thus, one event may lead to the
occurrence of another, and it is possible that a single event results in an
endless event loop. Thus, event listeners need to be expressed carefully so that
they do not trigger any unwanted events.
"""
from math import ceil

import heapq
from abc import ABC
from typing import Optional


class MultipleEventSourcesError(RuntimeError):
    """An exception raised when two ports connected to different clocks
    shall be connected to each other."""


class AbstractEventSource(ABC):
    """An event source defines the circumstances under which an event occurs.
    Events are occurrences of special occurrence that may require a reaction.

    Events can be reacted upon by updating the state of the system. For this
    purpose, event listeners can be registered which are called upon occurrence
    of the event.

    ``AbstractEventSource`` is the abstract base class for all event sources."""

    def __init__(self, owner):
        self.owner = owner
        self._listeners = set()

    def connect(self, other):
        """Connect an event to another event.

        Args:
            other: The other event port

        Raises:
            MultipleEventSourcesError: raised if both sides of the connection
                are already connected to different event sources
        """
        other.reference = self

    @property
    def reference(self) -> "AbstractEventSource":
        return self

    @reference.setter
    def reference(self, other: "AbstractEventSource"):
        if other.reference is not self:
            raise MultipleEventSourcesError

    @property
    def source(self) -> "AbstractEventSource":
        """The event source this of this event"""
        return self

    @property
    def listeners(self):
        """The listeners registered on this event source"""
        return self._listeners

    def register_listener(self, listener):
        """Register a listener for this event port.

        Args:
          listener: The listener to register
        """
        self.listeners.add(listener)


class EventPort(AbstractEventSource):
    """An event port is a placeholder for an event port that can be connected to
    other event ports or an event source."""

    def __init__(self, owner):
        AbstractEventSource.__init__(self, owner)
        self._reference = self

    def connect(self, other):
        """Connect an event port to another event port.

        Args:
            other: The other event port

        Raises:
            MultipleEventSourcesError: raised when two ports that are already
                connected to two different sources shall be connected to each
                other
        """

        self.reference = other

    @property
    def reference(self) -> AbstractEventSource:
        """The event port referenced by this port"""
        if self._reference is not self:
            # Try to further shorten the reference path
            self._reference = self._reference.reference
        return self._reference

    @reference.setter
    def reference(self, other: AbstractEventSource):
        if self._reference is self:
            other.listeners.update(self._listeners)
            self._reference = other.reference
        elif other.reference is other:
            self.listeners.update(other.listeners)
            other.reference = self
        else:
            # Both sides are bound already, so we defer this to one of the
            # references
            self._reference.reference = other

    @property
    def source(self) -> Optional[AbstractEventSource]:
        """The event source this port is connected to or ``None`` if it is
        not connected to any event source"""

        if self.reference is self:
            return None
        return self.reference.source

    @property
    def listeners(self):
        """The listeners registered on this event source"""
        if self.reference is not self:
            return self.reference.listeners
        return self._listeners


class ZeroCrossEventSource(AbstractEventSource):
    """A ``ZeroCrossEventSource`` defines an event source by the change of sign
    of a special event function. Such zero-cross events are specifically
    monitored and the values of event functions are recorded by the simulator.
    """

    def __init__(self, owner, event_function, direction=0, tolerance=1e-12):
        """
        Create a new zero-crossing event-source.

        Args:
            owner: The system or block this event belongs to
            event_function: The callable used to calculate the value of the
                event function
            direction: The direction of the sign change to consider
                Possible values:

                ``1``
                    Consider only changes from negative to positive

                ``-1``
                    Consider only changes from positive to negative

                ``0`` (default)
                    Consider all changes
            tolerance: The tolerance around zero
                Values with an absolute value less than or equal to
                ``tolerance`` are considered to be zero
        """

        AbstractEventSource.__init__(self, owner)
        self.event_function = event_function
        self.direction = direction
        self.event_index = self.owner.system.register_event(self)
        self.tolerance = tolerance

    def __call__(self, system_state):
        return self.event_function(system_state)


class Clock(AbstractEventSource):
    """A clock is an event source that generates a periodic event with optional
    jitter.

    The events are generated off of an ideal periodic event source and shifted
    by a random amount of jitter.

    The ideal periodic event source generates ticks separated by a given period.
    The ticks are shifted so that one tick will occur at the given start time.
    The interval from which the ideal event source will generate ticks can be
    limited at either side.
    """

    def __init__(
        self,
        owner,
        period,
        start_time=0.0,
        end_time=None,
        run_before_start=False,
        jitter_generator=None,
    ):
        """
        Construct a clock.

        Args:
            owner: The owner object of this clock (a system or a block)
            period: The period of the clock
            start_time: The start time of the clock (default: 0)
            end_time: The end time of the clock (default: ``None``)
            run_before_start: Flag indicating whether the clock shall already
                run before the start time (default: ``False``)
            jitter_generator: A callable which will return a single random value
                to be used as jitter, or ``None`` if no jitter shall be
                generated (default: ``None``)

        Generation of jitter values with an absolute value larger than or equal
        to the period may lead to generation of non-monotonously increasing tick
        times.
        """

        AbstractEventSource.__init__(self, owner)
        self.period = period
        self.start_time = start_time
        self.end_time = end_time
        self.run_before_start = run_before_start
        self.jitter_generator = jitter_generator

        self.owner.system.register_clock(self)

    def tick_generator(self, not_before):
        """Return a generator that will yield the times of the ticks of this
        clock.

        Args:
          not_before: The ticks shown shall not be before the given time

        Returns:
          A generator for ticks
        """

        # Note that we must not generate any ticks before the given time.
        # This means that even with jitter, the time must not be before that
        # time. Thus, we select the jitter amount first and then define
        # our index such that the tick with the given jitter will not be
        # before the start time.
        j = self.jitter_generator() if self.jitter_generator is not None else 0
        k = ceil((not_before - self.start_time - j) / self.period)
        if k < 0 and not self.run_before_start:
            # No ticks before the start
            k = 0

        tick_time = self.start_time + k * self.period + j
        while self.end_time is None or tick_time <= self.end_time:
            yield tick_time
            k += 1
            j = (
                self.jitter_generator()
                if self.jitter_generator is not None
                else 0
            )
            tick_time = self.start_time + k * self.period + j


class ClockQueue:
    """Queue of clock events"""

    def __init__(self, start_time, clocks):
        self.clock_queue = []

        # Fill the queue
        for clock in clocks:
            # Get a tick generator, started at the current time
            tick_generator = clock.tick_generator(start_time)
            try:
                first_tick = next(tick_generator)
                entry = _TickEntry(first_tick, clock, tick_generator)
                heapq.heappush(self.clock_queue, entry)
            except StopIteration:
                # The block did not produce any ticks at all,
                # so we just ignore it
                pass

    @property
    def next_clock_tick(self):
        """The time at which the next clock tick will occur or `None` if there
        are no further clock ticks"""
        if len(self.clock_queue) > 0:
            return self.clock_queue[0].tick_time
        return None

    def tick(self, current_time):
        """Advance all the clocks until the current time"""
        # We collect the clocks to tick here and executed all their listeners
        # later.
        clocks_to_tick = list()

        while (
            len(self.clock_queue) > 0
            and self.clock_queue[0].tick_time <= current_time
        ):
            tick_entry = heapq.heappop(self.clock_queue)
            clock = tick_entry.clock

            clocks_to_tick.append(clock)

            try:
                # Get the next tick for the clock
                next_tick_time = next(tick_entry.tick_generator)
                next_tick_entry = _TickEntry(
                    next_tick_time, clock, tick_entry.tick_generator
                )
                # Add the clock tick to the queue
                heapq.heappush(self.clock_queue, next_tick_entry)
            except StopIteration:
                # This clock does not deliver any more ticks, so we simply
                # ignore it from now on.
                pass

        return clocks_to_tick


class _TickEntry:
    """A ``_TickEntry`` holds information about the next tick of a given clock.
    An order over ``_TickEntry`` instances is defined by their time.
    """

    def __init__(self, tick_time, clock, tick_generator):
        self.tick_time = tick_time
        self.clock = clock
        self.tick_generator = tick_generator

    def __lt__(self, other):
        return self.tick_time < other.tick_time
