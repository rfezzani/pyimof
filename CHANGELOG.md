# Changelog

All notable changes to the Pyimof package will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to
[Semantic Versioning](http://semver.org/spec/v2.0.0.html).

[Unreleased]
------------

### Added

- image registration example to the doc,
- Gaussian kernel support to `iLK`.

### Changed

- `TV` regularization method in `tvl1` solver.

### Fixed

- `pyimof.io.floread` complyance with middlebury files,
- image loading as `np.float32`.

### Removed

- `div` and discrete gradient schemes.

v1.0.0 - 2019-06-04
------------

### Added

- `iLK` optical flow solver,
- `TV-L1` optical flow solver,
- the two-frame grayscale Middlebury training dataset in the
  `pyimof.data` submodule,
- read/write vector field to `.flo` format in the `pyimof.io`
  submodule,
- optical flow color coding and quiver plot in the `pyimof.display`
  submodule.
