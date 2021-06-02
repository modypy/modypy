# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- Added a schematic and made some information more explicit in the integrator
  tutorial.

## [3.0.0rc2 - 2021-05-31]
### Changed
- Updated the documentation regarding the changes in the default shape of
  signals and states, as well as the new coding style and the new github and
  documentation URLs.
## Fixed
- Removed _version.py from coverage measurements, as it is actually not our own
  code, is replaced by a very simple piece of code in releases and is not well
  covered by tests.

## [3.0.0rc1 - 2021-05-30]
### Added
- Added a __version__ property in the modypy package (now used in setup.py
  and sphinx conf.py)
### Changed
- Major reorganisation of the Simulator, in preparation for an extensible
  event detection concept. With optimisiations on the Simulator, the ODE solver
  is now only re-initialized if events lead to state changes. This also means
  that control of step size using `max_step` may be necessary.
- States and ports may now be scalar instead of just of shape `(1)`.
- Major changes to the evaluation interface: The dictionary-based access has
  been deprecated, and the evaluation class has been removed. Both simulation
  and the simulator results support the same way of accessing system states and
  inputs by call. This also leads to signals/ports not being explicitly
  registered in the system anymore (#20/#27).
- The results in ``SimulatorResult`` are now shaped differently: The last index
  represents the sample number (#21)
- The ``run_until`` method now returns a generator for the simulation sample
  points. The values returned are system states. They can be cached using a
  ``SimulatorResult``.
  This allows for more flexibility on working with simulation data (e.g., 
  when doing continuous simulation).

## [2.1.0] - 2021-03-07
### Added
- Added a new dictionary interface for accessing state, signal and event values
- Made states, ports and event ports callable for evaluation

## [2.0.0] - 2021-02-28
### Added
- Added this changelog
- Added clocks as event generators for discrete-time components
- Added a set of new blocks, including zero-order-hold, integrator and saturation
- Added a programmer's guide to the documentation
- Added detection of excessive zero-crossing events
- Added minimization objective for steady-state determination
- Event listeners can now trigger other zero-crossing events
- The direction of change for zero-crossing-events can now be specified
- Zero-crossing-events now have a configurable tolerance for the sign-change
- Introduced ``sum_signal`` and ``gain`` in ``modypy.linear`` to replace
  ``Gain`` and ``Sum``
### Changed
- Improved API documentation
- Generalized the interfaces for steady-state finding and linearization
- Modified the event system to include clocks and event ports, similar to signal
  ports.
- States can now be defined without a derivative function. The values of these
  states stay constant if not otherwise modified (e.g., by event handlers)
- The ``t_bounds`` parameter of ``run_until`` is now called ``time_boundary``
- For discrete-time-only systems the integration step of state derivatives is
  skipped.
### Deprecated
- The ``inputs`` property on the ``DataProvider`` object passed to derivative,
  signal, event and event handler functions is deprecated.
- The ``Gain`` and ``Sum`` blocks in ``modypy.linear`` are deprecated.
### Removed
- The linearization algorithm does not consider ``OutputPort`` instances
  anymore.
### Fixed
- Fixed several issues with the packaging (content type of long description,
  versioning, inclusion of documentation, tests and examples in source package).

## [1.0.1] - 2021-01-07 - Brownbag Release
### Fixed
- Added missing utils module to the package and added tests for it.

## [1.0] - 2021-01-06 - Initial Release
