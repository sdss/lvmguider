# Changelog

## Next version

### ✨ Improved

* [#2](https://github.com/sdss/lvmguider/pull/2) Improvements to `proc-` file and WCS:
    - Set `WCSMODE` in header.
    - Solve both cameras at the same time.
    - Store WCS for individual images in PROC extension (only for acquisition frames).
    - Add reference frames to `proc-` file.
    - Update `proc-` file WCS during guiding using reference WCS and offsets.
    - Fix offsets in header.


## 0.1.0 - July 26, 2023

### 🚀 New

* Initial version of the guider with functioning focus, acquisition, and guiding routines.
