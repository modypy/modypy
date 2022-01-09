"""Blocks for filtering"""
import numpy as np

from modypy.model import EventPort, Block, State
from modypy.model.ports import AbstractSignal


class IIRFilter(Block, AbstractSignal):
    """Infinite Impulse Response (IIR) filter

    Args:
        parent: Parent block or system
        source: The source signal to filter
        filter_spec: The filter specification, e.g. as returned by
            `scipy.signal.butter`.
        filter_format: The filter format. Allowed values are 'ba' or 'tf' for
            transfer function format, 'zpk' for zero-pole-gain format or 'sos'
            for second-order section format.

    The filter format is as follows:

        'ba', 'tf'
            Transfer function format, given as a tuple `(b, a)` of coefficients,
            with `b` being the coefficients for the nominator and `a` being the
            coefficients for the denominator.
            Coefficients with the highest order are listed first, i.e. the
            polynomial `z^2+3z+5` would be represented as `(1,3,5)`.
        'zpk'
            Zero-pole-gain format, given as a tuple `(z, p, k)` with `z` giving
            the zeroes, `p` giving the poles and `k` being the system gain.
        'sos'
            Second-order-section format, given as an array with shape
            `(n_sections, 6)`, with each row (first index) corresponding to a
            second-order section.
            See `scipy.signal.sosfilt` for details on the second-order section
            format.

    Raises:
        ValueError: The filter format is not valid.

    Example:
        Generate a signal of a 10Hz and a 20Hz sine wave and apply an order-10
        high-pass Butterworth filter to it.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from scipy import signal
        >>> from modypy.blocks import linear, sources, filters, discrete
        >>> from modypy import model, simulation
        >>>
        >>> sin1_sig = sources.FunctionSignal(np.sin,
        >>>         linear.gain(2*10*np.pi, sources.time))
        >>> sin2_sig = sources.FunctionSignal(np.sin,
        >>>         linear.gain(2*20*np.pi, sources.time))
        >>> sum_sig = linear.sum_signal((sin1_sig, sin2_sig))
        >>>
        >>> system = model.System()
        >>> clock = model.Clock(owner=system, period=1/1000.0)
        >>> zoh = discrete.zero_order_hold(system, sum_sig, clock)
        >>> filter_spec = signal.butter(10, 15, 'hp', fs=1000, output='sos')
        >>> filter_block = filter.IIRFilter(
        >>>     parent=system,
        >>>     source=zoh,
        >>>     filter_spec=filter_spec,
        >>>     filter_format='sos')
        >>> filter_block.trigger.connect(clock)
        >>>
        >>> sim = simulation.Simulator(system=system, start_time=0)
        >>> result = simulation.SimulationResult(system,
        >>>     sim.run_until(time_boundary=1.0))
        >>>
        >>> fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        >>> ax1.plot(result.time, sum_sig(result))
        >>> ax1.set_title('10 Hz and 20 Hz sinusoids')
        >>> ax1.axis([0, 1, -2, 2])
        >>> ax2.plot(result.time, filter_block(result))
        >>> ax2.set_title('After 15 Hz high-pass filter')
        >>> ax2.axis([0, 1, -2, 2])
        >>> ax2.set_xlabel('Time [seconds]')
        >>> plt.tight_layout()
        >>> plt.show()

    .. versionadded:: 4.0.0
    """

    def __init__(self, parent, source, filter_spec, filter_format="ba"):
        Block.__init__(self, parent=parent)
        AbstractSignal.__init__(self, shape=source.shape)

        # Convert filter specification to second-order section format
        # These only work for one-dimensional filter specifications
        if filter_format in ("ba", "tf"):
            # pylint: disable=import-outside-toplevel
            from scipy.signal import tf2sos

            filter_spec = tf2sos(*filter_spec)
        elif filter_format == "zpk":
            # pylint: disable=import-outside-toplevel
            from scipy.signal import zpk2sos

            filter_spec = zpk2sos(*filter_spec)
        elif filter_format != "sos":
            raise ValueError(
                f"Invalid filter format '{filter_format}'. "
                f"Allowed formats are 'ba', 'tf', 'zpk' or 'sos'"
            )

        self.source = source
        self.filter_spec = np.asarray(filter_spec)

        # For each of the filter elements we will need two state variables
        self.n_sections = self.filter_spec.shape[0]

        self.state = State(
            owner=self, shape=(self.n_sections, 2) + source.shape
        )
        self.trigger = EventPort(self)
        self.trigger.register_listener(self._update_filter)

    def _update_filter(self, system_state):
        """Update the filter state"""

        # Use the input as the input to the zeroth section
        u_old = self.source(system_state.prev)
        x_old = self.state(system_state.prev)
        x_new = self.state(system_state)

        for section in range(self.n_sections):
            # Calculate the old output
            y_old = (
                self.filter_spec[section, 0] * u_old + x_old[section, 0]
            ) / self.filter_spec[section, 3]

            # Determine the new states
            x_new[section, 0] = (
                self.filter_spec[section, 1] * u_old
                - self.filter_spec[section, 4] * y_old
                + x_old[section, 1]
            )
            x_new[section, 1] = (
                self.filter_spec[section, 2] * u_old
                - self.filter_spec[section, 5] * y_old
            )

            # Use the section output as the input to the next section
            u_old = y_old

    def __call__(self, system_state):
        """Calculate the output of the filter"""
        # Use the input as the input to the zeroth section
        u = self.source(system_state)
        x = self.state(system_state)

        for section in range(self.n_sections):
            # Calculate the output of this section
            u = (
                self.filter_spec[section, 0] * u + x[section, 0]
            ) / self.filter_spec[section, 3]

        return u
