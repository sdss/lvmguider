# Changelog

## Next version

### 🚀 New

* [#5](https://github.com/sdss/lvmguider/pull/5) Major refactor of the guider code.
  * Extracted sources are always matched to Gaia DR3 regardless of whether the solution was obtained using astrometry.net or kd-tree. `lmag` zero points are calculated for each source.
  * Once acquisition is complete the WCS of the individual cameras is determined from the Gaia cross-match, and not by tranlating the reference WCS using the measured offset. This allows each new WCS to also reflect changes in PA.
  * Average zero points and PAs are output.
  * All the metadata is carried out in two dataclasess, `CameraSolution` and `GuiderSolution`.
  * The output data model has changed. Former `proc-` files are now named `lvm.{telescope}.guider`, and sources are saved as `parquet` tables. The latter allows to preserve the column types in the presence of missing data.


## 0.3.0 - September 1, 2023

### 🚀 New

* [#3](https://github.com/sdss/lvmguider/pull/3) Plotting of focus sweep and fit data using a cubic spline.
* [#4](https://github.com/sdss/lvmguider/pull/4) Added code to generate co-added frames and QA metrics. This code will probably change significantly before it's used routinely.

### ✨ Improved

* Command telescope offsets with more relaxed thresholds for axis errors. Timeout and continue guiding if the offset takes longer than a certain amount of time.
* Command PlaneWave offsets with more relaxed `axis_error` and a reasonable timeout to prevent offsets taking long to complete under windy conditions.


## 0.2.1 - August 20, 2023

### ✨ Improved

* Use actor model for `lvm.TEL.foc` to reduce polling.
* Round up `focus_position`.
* Prevent `astropy` from downloading data from the internet.

### 🔧 Fixed

* Set guider to `IDLE`` after` `stop --now`.


## 0.2.0 - August 14, 2023

### ✨ Improved

* [#2](https://github.com/sdss/lvmguider/pull/2) Improvements to `proc-` file and WCS:
  * Set `WCSMODE` in header.
  * Solve both cameras at the same time.
  * Store WCS for individual images in PROC extension (only for acquisition frames).
  * Add reference frames to `proc-` file.
  * Update `proc-` file WCS during guiding using reference WCS and offsets.
  * Fix offsets in header.
* Take into account the offset of the reference image when guiding.
* Handle compressed AG images.

### 🔧 Fixed

* Fix a cosine of the declination in the calculation of the offsets during guiding, which caused guiding to drift away and revert to acquisition.
* Fix cases in which the measured position would not be reported.
* Fix repeated `-t` flag in `expose` command.
* Fix some cases in which the sequence number for the next AG exposure would be off or would use a sequence number for which exposures already existed.

### ⚙️ Engineering

* Lint using `ruff`.


## 0.1.0 - July 26, 2023

### 🚀 New

* Initial version of the guider with functioning focus, acquisition, and guiding routines.
