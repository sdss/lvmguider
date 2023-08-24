#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-08-23
# @Filename: coadd.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass

import nptyping as npt
import numpy
import sep
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.time import Time

from sdsstools import get_logger
from sdsstools.time import get_sjd

from lvmguider.tools import get_frameno, get_proc_path


ARRAY_2D_UINT = npt.NDArray[npt.Shape["*, *"], npt.UInt16]
ARRAY_2D = npt.NDArray[npt.Shape["*, *"], npt.Float32]


@dataclass
class FrameData:
    """Data associated with a frame."""

    file: pathlib.Path
    proc_header: fits.Header
    frameno: int
    date_obs: Time
    camera: str
    telescope: str
    exptime: float
    image_type: str
    kmirror_drot: float
    focusdt: float
    fwhm_median: float | None = None
    fwhm_std: float | None = None
    guide_mode: str = "guide"
    stacked: bool = False


def create_coadded_frame_header(frame_data: dict[int, FrameData]):
    """Creates the header object for a co-added frame."""

    # Create a list with only the stacked frames and sort by frame number.
    frames = list(
        sorted(
            [fd for fd in frame_data.values() if fd.stacked],
            key=lambda x: x.frameno,
        )
    )

    header = fits.Header()

    sjd = get_sjd("LCO", frames[0].date_obs.to_datetime())

    header["TELESCOP"] = (frames[0].telescope, " Telescope that took the image")
    header["CAMNAME"] = (frames[0].camera, "Camera name")
    header["INSTRUME"] = ("LVM", "SDSS-V Local Volume Mapper")
    header["OBSERVAT"] = ("LCO", "Observatory")
    header["MJD"] = (sjd, "SDSS MJD (MJD+0.4)")
    header["EXPTIME"] = (1.0, "Exposure time [s]")
    header["PIXSIZE"] = (9.0, "Pixel size [um]")
    header["PIXSCALE"] = (1.009, "Scaled of unbinned pixel [arcsec/pix]")
    header["FRAME0"] = (frames[0].frameno, "First co-added frame")
    header["FRAME1"] = (frames[-1].frameno, "Last co-added frame")
    header["NFRAMES"] = (len(frames), "Number of frames stacked")
    header["OBSTIME0"] = (frames[0].date_obs.isot, "DATE-OBS of FRAME0")
    header["OBSTIME1"] = (frames[-1].date_obs.isot, "DATE-OBS of FRAME1")
    header.insert("TELESCOP", ("", "/*** BASIC DATA ***/"))

    return header


def get_frame_range(
    path: str | pathlib.Path,
    telescope: str,
    frameno0: int,
    frameno1: int,
    camera: str | None = None,
) -> list[pathlib.Path]:
    """Returns a list of AG frames in a frame number range.

    Parameters
    ----------
    path
        The directory where the files are written to.
    telescope
        The telescope for which to select frames.
    frameno0
        The initial frame number.
    frameno1
        The final frame number.
    camera
        The camera, east or west, for which to select frames. If not provided
        selects both cameras.

    Returns
    -------
    frames
        A sorted list of frames in the desired range.

    """

    if camera is None:
        files = pathlib.Path(path).glob(f"lvm.{telescope}.agcam.*")
    else:
        files = pathlib.Path(path).glob(f"lvm.{telescope}.agcam.{camera}_*")

    selected: list[pathlib.Path] = []
    for file in files:
        frameno = get_frameno(file)
        if frameno >= frameno0 and frameno <= frameno1:
            selected.append(file)

    return list(sorted(selected))


def coadd_camera_frames(
    files: list[str | pathlib.Path],
    outpath: str = "coadds/lvm.{telescope}.agcam.{camname}.coadd_{frameno0:08d}_{frameno1:08d}.fits",  # noqa: E501
    use_sigmaclip: bool = False,
    sigma: float = 3.0,
    skip_acquisition_after_guiding: bool = False,
    log: logging.Logger | None = None,
    verbose: bool = False,
    quiet: bool = False,
):
    """Co-adds a series of AG camera frames.

    This routine does the following:

    - Selects all the frames from the end of the initial acquisition.
    - For each frame in the range to co-add, substracts the dark current frame and
      derives counts per second.
    - Co-adds the frames using a sigma-clipped median.
    - Extracts sources from the image using SExtractor. Calculates PSF parameters
      by fitting Gaussian profiles to the x and y-axis marginal distributions.
    - Uses the first guide WCS to determine the centre of the image and query
      Gaia DR3 sources.
    - Uses a kd-tree to match Gaia objects to image sources.
    - Calculates image transparency as the median zero-point of an LVM-defined
      magnitude for each source. This AB magnitude is derived by integrating the
      Gaia XP mean spectrum over a bandpass with the effective response of the
      optical system and AG sensor.
    - Derives a new WCS solution for the co-added image by matching pixel positions
      for the sources in the co-added image to Gaia coordinates.
    - Writes the co-added image and metadata to disk.

    Parameters
    ----------
    files
        The list of files to be co-added.
    outpath
        The path to the co-added frame. If a relative path, written relative to
        the path of the first file to be co-added.
    use_sigmaclip
        Whether to use sigma clipping when combining the stack of data. Disabled
        by default as it uses significant CPU and memory.
    sigma
        The sigma value for co-addition sigma clipping.
    skip_acquisition_after_guiding
        Images are co-added starting on the first frame marked as ``'guide'``
        and until the last frame. If ``skip_acquisition_after_guiding=True``,
        ``'acquisition'`` images found after the initial acquisition are
        discarded.
    log
        An instance of a logger to be used to output messages. If not provided
        a custom logger will be created.
    verbose
        Increase verbosity of the logging.
    quiet
        Do not output log messages.

    Returns
    -------
    hdul
        An ``HDUList`` object with the co-added frame written to disk.

    """

    # Create log and set verbosity
    log = log or get_logger("lvmguider.coadd_camera_frames", use_rich_handler=True)
    for handler in log.handlers:
        if verbose:
            handler.setLevel(logging.DEBUG)
        elif quiet:
            handler.setLevel(logging.CRITICAL)

    # Get the list of frame numbers from the files.
    files = list(sorted(files))
    frame_nos = sorted([get_frameno(file) for file in files])

    # Use the first file to get some common data (or at least it should be common!)
    sample_raw_header = fits.getheader(files[0], "RAW")

    camname: str = sample_raw_header["CAMNAME"]
    telescope: str = sample_raw_header["TELESCOP"]
    dateobs0: str = sample_raw_header["DATE-OBS"]
    sjd = get_sjd("LCO", date=Time(dateobs0, format="isot").to_datetime())

    log.info(
        f"Co-adding frames {min(frame_nos)}-{max(frame_nos)} for MJD={sjd}, "
        f"camera={camname!r}, telescope={telescope!r}."
    )

    frame_data: dict[int, FrameData] = {}
    dark: ARRAY_2D | None = None
    data_stack: list[ARRAY_2D] = []

    # Loop over each file, add the dark-subtracked data to the stack, and collect
    # frame metadata.
    for ii, file in enumerate(files):
        frameno = frame_nos[ii]

        with fits.open(str(file)) as hdul:
            if "PROC" not in hdul:
                log.warning(f"Frame {frameno}: PROC extension not found.")
                continue

            # PROC header of the raw AG frame.
            proc_header = hdul["PROC"].header

            # Get the proc- file. Just used to determine
            # if we were acquiring or guiding.
            proc_file = get_proc_path(file)
            if not proc_file.exists():
                log.warning(f"Frame {frameno}: cannot find associated proc- image.")
                continue

            proc_astrometry = fits.getheader(str(proc_file), "ASTROMETRY")

            # If we have not yet loaded the dark frame, get it and get the
            # normalised dark.
            if dark is None and proc_header.get("DARKFILE", None):
                hdul_dark = fits.open(str(proc_header["DARKFILE"]))

                dark_data: ARRAY_2D_UINT = hdul_dark["RAW"].data.astype(numpy.float32)
                dark_exptime: float = hdul_dark["RAW"].header["EXPTIME"]

                dark = dark_data / dark_exptime

                hdul_dark.close()

            # Get the frame data and exposure time. Calculate the counts per second.
            exptime = float(hdul["RAW"].header["EXPTIME"])
            data: ARRAY_2D = hdul["RAW"].data.astype(numpy.float32) / exptime

            # If we have a dark frame, subtract it now. If not fit a
            # background model and subtract that.
            if dark is not None:
                data -= dark
            else:
                back = sep.Background(data)
                data -= back.back()

            # Decide whether this frame should be stacked.
            guide_mode = "acquisition" if proc_astrometry["ACQUISIT"] else "guide"
            skip_acquisition = len(data_stack) == 0 or skip_acquisition_after_guiding
            if guide_mode == "acquisition" and skip_acquisition:
                stacked = False
            else:
                stacked = True
                data_stack.append(data)

            # Determine the median FWHM and FWHM deviation for this frame.
            sources = hdul["SOURCES"] if "SOURCES" in hdul else None
            if sources is None:
                log.warning(f"Frame {frameno}: SOURCES extension not found.")
                fwhm_median = None
                fwhm_std = None
            else:
                fwhm = 0.5 * (sources.data["xstd"] + sources.data["ystd"])
                fwhm_median = float(numpy.median(fwhm))
                fwhm_std = float(numpy.std(fwhm))

            # Add information as a FrameData. We do not include the data itself
            # because it's in data_stack and we don't need it beyond that.
            frame_data[frameno] = FrameData(
                file=pathlib.Path(file),
                frameno=frameno,
                proc_header=proc_header,
                camera=proc_header["CAMNAME"],
                telescope=proc_header["TELESCOP"],
                date_obs=Time(proc_header["DATE-OBS"], format="isot"),
                exptime=exptime,
                image_type=proc_header["IMAGETYP"],
                kmirror_drot=proc_header["KMIRDROT"],
                focusdt=proc_header["FOCUSDT"],
                fwhm_median=fwhm_median,
                fwhm_std=fwhm_std,
                guide_mode=guide_mode,
                stacked=stacked,
            )

            del data

    if dark is None:
        log.critical("No dark frame found in range. Co-added frame may be unreliable.")

    # Combine the stack of data frames using the median of each pixel.
    # Optionally sigma-clip each pixel (this is computationally intensive).
    if use_sigmaclip:
        log.debug(f"Sigma-clipping stack with sigma={sigma}")
        stack_masked = sigma_clip(
            numpy.array(data_stack),
            sigma=int(sigma),
            maxiters=3,
            masked=True,
            axis=0,
            copy=False,
        )

        log.debug("Creating median-combined co-added frame.")
        coadd: ARRAY_2D = numpy.ma.median(stack_masked, axis=0).data

    else:
        log.debug("Creating median-combined co-added frame.")
        coadd: ARRAY_2D = numpy.median(numpy.array(data_stack), axis=0)

    del data_stack

    # Create the path for the output file.
    frameno0 = min(frame_nos)
    frameno1 = max(frame_nos)
    outpath = outpath.format(
        telescope=telescope,
        camname=camname,
        frameno0=frameno0,
        frameno1=frameno1,
    )

    if pathlib.Path(outpath).is_absolute():
        outpath_full = pathlib.Path(outpath)
    else:
        outpath_full = pathlib.Path(files[0]).parent / outpath

    if outpath_full.parent.exists() is False:
        outpath_full.parent.mkdir(parents=True, exist_ok=True)

    # Construct the header.
    header = create_coadded_frame_header(frame_data)

    # Write the file.
    log.debug(f"Writing co-added frame to {outpath_full.absolute()!s}")
    hdul = fits.HDUList([fits.PrimaryHDU()])
    hdul.append(fits.CompImageHDU(data=coadd, header=header, name="COADD"))
    hdul.writeto(str(outpath_full), overwrite=True)
    hdul.close()

    return hdul
