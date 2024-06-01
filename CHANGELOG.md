# Changelog

## 0.5.2 - June 1, 2024

### 🔧 Fixed

* Fix comment for keyword `REFFILE` being too long when the keyword is populated.


## 0.5.1 - February 12, 2024

### ✨ Improved

* Added `AIRMASS` and `GUIDERV` keywords to co-add FITS files and data frames.


## 0.5.0 - February 7, 2024

### 🚀 New

* Added `RA`, `DEC`, `FWHM`, `ISFSWEEP` to the `PROC` headers, `FrameData`, and `lvmops.agcam_frame`. All AG frames are now loaded to the database, not only those associated with an exposure.
* `Focuser` now uses a temperature model to estimate the initial best focus.
* Added `adjust-focus` and `focus-info` commands. If a focus sweep has been previously executed, `adjust-focus` can be used to adjust the focuser position based on the delta temperature between the bench temperature during the focus sweep and the current temperature.

### ✨ Improved

* Use inverse standard deviation as weights for the focus spline fit.
* `Focuser` now has an option to require the best resulting focus to be in the range of focuser positions tested or it will automatically repeat the focus sweep with a larger step size.


## 0.4.2 - January 31, 2024

### ✨ Improved

* Upgraded `astropy` to 6.0.0. The docker image now tries (but continues if it fails) to update the `astropy-iers-data` package before running the actor and internally `lvmguider` should never try to download IERS data over the internet.

### 🔧 Fixed

* Fixed `pandas` deprecation in `lvmguider.extraction`.


## 0.4.1 - November 5, 2023

### 🔧 Fixed

* Actually import the `corrections` command.
* Use WCS to determine `RAMEAS`, `DECMEAS`, `PAMEAS` in co-added images.
* Wrap `offset_pa` in the -180 to 180 range.


## 0.4.0 - September 28, 2023

In addition to the changes listed under `0.4.0b2`, this version provides:

### 🚀 New

* [#8](https://github.com/sdss/lvmguider/pull/8) Added `corrections` command.

### 🔧 Fixed

* Reported FWHM were actually Gaussian sigmas. This has not been fixed.
* Improved matching of extracted sources with Gaia DR3.


## 0.4.0b2 - September 14, 2023

In addition to the changes listed under `0.4.0b1`, this version provides:

### 🚀 New

* [#7](https://github.com/sdss/lvmguider/pull/7) `stop` command now cancels the guider task for faster performance.

### ✨ Improved

* If a single camera fails to match with Gaia during guiding, retry that camera using astrometry.net.
* Use 25% percentile to calculate the FWHM.

### 🔧 Fixed

* The co-add file watcher seemed to skip files. Apparently this happens because some events are detected as a file being moved (from a temporary file to the final `.fits.gz` file) instead of a file creation. This version handles both kinds of events.
* Various fixes for the co-added code when dealing with missing data.


## 0.4.0b1 - September 12, 2023

### 🚀 New

* [#5](https://github.com/sdss/lvmguider/pull/5) Major refactor of the guider code.
  * Extracted sources are always matched to Gaia DR3 regardless of whether the solution was obtained using astrometry.net or kd-tree. `lmag` zero points are calculated for each source.
  * Once acquisition is complete the WCS of the individual cameras is determined from the Gaia cross-match, and not by tranlating the reference WCS using the measured offset. This allows each new WCS to also reflect changes in PA.
  * Average zero points and PAs are output.
  * All the metadata is carried out in two dataclasess, `CameraSolution` and `GuiderSolution`.
  * The output data model has changed. Former `proc-` files are now named `lvm.{telescope}.guider`, and sources are saved as `parquet` tables. The latter allows to preserve the column types in the presence of missing data.
* [#6](https://github.com/sdss/lvmguider/pull/6) Major improvements to co-adding code. QA plots generated during the co-adding process and the results are loaded to the database.
* Implemented guiding in rotation/PA.
* Added script to reprocess all MJDs at LCO and generate co-added products.
* Added file watcher to generate co-adds during the night. Updated the container image to support running the file watcher as a Kubernetes service.

### ✨ Improved

* Improved the sigma clipping of FWHM outliers.

### 🏷️ Changed

* Renamed references to "master frame" to "full frame". Coordinates are now referred to as `x_ff` and `y_ff`.


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
