"""Blocks for discrete-time simulation"""
from modypy import model
from scipy import stats


class ZeroOrderHold(model.Block):
    """A zero-order-hold block which samples an input signal when the connected
    event occurs.

    The block provides an event port ``event_input`` that should be connected
    to the event source that shall trigger the sampling.
    """

    def __init__(self, owner, shape=1, initial_condition=None):
        """
        Constructor for ``ZeroOrderHold``

        Args:
            owner: The owner of the block (system or block)
            shape: The shape of the input and output signal
            initial_condition: The initial state of the sampling output
                (before the first tick of the block)
        """
        model.Block.__init__(self, owner)

        self.event_input = model.EventPort(self)
        self.event_input.register_listener(self.update_state)
        self.input = model.Port(shape=shape)
        self.output = model.State(
            self,
            shape=shape,
            initial_condition=initial_condition,
            derivative_function=None,
        )

    def update_state(self, data):
        """Update the state on a clock event

        Args:
          data: The time, states and signals of the system
        """
        self.output.set_value(data, self.input(data))


def zero_order_hold(system, input_port, event_port, initial_condition=None):
    """Create a ``ZeroOrderHold`` instance that samples the given input port.
    This is a convenience function that returns the single output port of the
    zero-order-hold block.

    Args:
      system: The owner of the ``ZeroOrderHold`` block.
      input_port: The input port to sample.
      event_port: The event port to use as a sampling signal
      initial_condition: The initial condition of the ``ZeroOrderHold`` block.
        (Default value = None)

    Returns:
        The output signal of the zero-order hold
    """

    hold = ZeroOrderHold(
        system, shape=input_port.shape, initial_condition=initial_condition
    )
    hold.input.connect(input_port)
    hold.event_input.connect(event_port)
    return hold.output


class NoiseSource(model.State):
    """A discrete-time noise source

    This noise source provides random outputs sourced from a given
    random source.
    The noise is sampled when the event given by the trigger occurs.

    To generate band-limited white noise, use an uncorrelated random source and
    a time constant that is sufficiently small in relation to the smallest
    time constant in the system.

    Args:
        owner: The owner block or system
        trigger: The event source triggering the update of the noise source.
        shape: The shape of the output (default: ``()``)
        random_source: A callable accepting a ``size`` argument giving the shape
            of data to be generated

    Example:
        Collect and plot noise data:

        >>> from modypy import model, simulation
        >>> from modypy.blocks import filters, discrete
        >>> import matplotlib.pyplot as plt
        >>>
        >>> system = model.System()
        >>> noise_clock = model.Clock(owner=system, period=1/10)
        >>> noise = discrete.NoiseSource(owner=system, trigger=noise_clock)
        >>>
        >>> simulator = simulation.Simulator(system=system,
        >>>                                  start_time=0.0)
        >>> result = simulation.SimulationResult(
        >>>     system=system,
        >>>     source=simulator.run_until(time_boundary=10.0))
        >>>
        >>> plt.plot(result.time, noise(result))
        >>> plt.show()
    """

    def __init__(
        self,
        owner,
        trigger: model.AbstractEventSource,
        shape: model.ShapeType = (),
        random_source=stats.norm.rvs,
    ):
        model.State.__init__(self, owner, shape=shape)
        trigger.register_listener(self._update_state)
        self.random_source = random_source

    def _update_state(self, system_state):
        """Fetch new random data"""
        self.set_value(system_state, self.random_source(size=self.shape))
