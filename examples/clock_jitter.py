"""Example of a clock with jitter"""

from modypy import model, simulation
from scipy import stats
import matplotlib.pyplot as plt


class Counter(model.State):
    def __init__(self, owner, trigger: model.EventPort):
        model.State.__init__(self, owner)
        trigger.register_listener(self._update_counter)

    def _update_counter(self, system_state):
        self.set_value(system_state, self(system_state.prev) + 1)


system = model.System()
clock_periodic = model.Clock(owner=system, period=1.0)
counter_periodic = Counter(owner=system, trigger=clock_periodic)
clock_with_jitter = model.Clock(
    owner=system,
    period=1.0,
    jitter_generator=stats.uniform(scale=0.1, loc=-0.05).rvs,
)
counter_jitter = Counter(owner=system, trigger=clock_with_jitter)

simulator = simulation.Simulator(system, start_time=0.0)
result = simulation.SimulationResult(
    system, source=simulator.run_until(time_boundary=10.0)
)

plt.step(result.time, counter_periodic(result), label="periodic", where="post")
plt.step(result.time, counter_jitter(result), label="jitter", where="post")
plt.grid()
plt.legend()
plt.show()
