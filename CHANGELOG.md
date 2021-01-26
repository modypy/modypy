# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added this changelog.
- Added clocks as event generators for discrete-time components.
- Added discrete-time block library, including a zero-order-hold block.
- Added a user's guide to the documentation.
- The direction of change for zero-crossing-events can be specified now.
### Changed
- Improved documentation
- Modified the event system to include clocks and event ports, similar to signal
  ports.
- States can now be defined without a derivative function. The values of these
  states stay constant if not otherwise modified (e.g., by event handlers)
- The ``t_bounds`` parameter of ``run_until`` is now called ``time_boundary``  
### Deprecated
- The ``inputs`` property on the ``DataProvider`` object passed to derivative,
  signal, event and event handler functions is deprecated and will be removed
  in a future release.
### Fixed
- Fixed several issues with the packaging (content type of long description,
  versioning, inclusion of documentation, tests and examples in source package).

## [1.0.1] - 2021-01-07 - Brownbag Release
### Fixed
- Added missing utils module to the package and added tests for it.

## [1.0] - 2021-01-06 - Initial Release