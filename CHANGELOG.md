# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added a __version__ property in the modypy package (now used in setup.py
  and sphinx conf.py)
### Changed
- Major changes to the evaluation interface: The dictionary-based access has
  been dropped, as has the evaluation class. Both simulation and the simulator
  results support the same way of accessing system states and inputs. This also
  leads to signals/ports not being explicitly registered in the system anymore.
- The results in ``SimulatorResult`` are now shaped differently: The last index
  represents the sample number (#21)

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
