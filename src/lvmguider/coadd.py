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
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from sqlite3 import OperationalError

from typing import Any, cast

import nptyping as npt
import numpy
import pandas
import peewee
import sep
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
from scipy.spatial import KDTree

from sdsstools.time import get_sjd

from lvmguider import config
from lvmguider import log as llog
from lvmguider.extraction import extract_marginal
from lvmguider.tools import get_db_connection, get_frameno, get_proc_path
from lvmguider.transformations import (
    XZ_AG_FRAME,
    ag_to_master_frame,
    get_crota2,
    solve_locs,
)


ARRAY_2D_UINT = npt.NDArray[npt.Shape["*, *"], npt.UInt16]
ARRAY_2D = npt.NDArray[npt.Shape["*, *"], npt.Float32]
ARRAY_1D = npt.NDArray[npt.Shape["*"], npt.Float32]

COADD_CAMERA_DEFAULT_PATH = (
    "coadds/lvm.{telescope}.agcam.{camname}.coadd_{frameno0:08d}_{frameno1:08d}.fits"
)
MASTER_COADD_DEFAULT_PATH = (
    "coadds/lvm.{telescope}.agcam.coadd_{frameno0:08d}_{frameno1:08d}.fits"
)


@dataclass
class FrameData:
    """Data associated with a frame."""

    file: pathlib.Path
    raw_header: fits.Header
    frameno: int
    date_obs: Time
    camera: str
    telescope: str
    exptime: float
    image_type: str
    kmirror_drot: float
    focusdt: float
    proc_header: fits.Header | None = None
    proc_file: pathlib.Path | None = None
    guide_error: float | None = None
    fwhm_median: float | None = None
    fwhm_std: float | None = None
    guide_mode: str = "guide"
    stacked: bool = False
    wcs: WCS | None = None


def create_coadded_frame_header(
    frame_data: list[FrameData],
    sources: pandas.DataFrame,
    use_sigmaclip: bool = False,
    sigma: int | None = None,
    wcs: WCS | None = None,
    is_master: bool = False,
):
    """Creates the header object for a co-added frame."""

    # Create a list with only the stacked frames and sort by frame number.
    frames = list(sorted(frame_data, key=lambda x: x.frameno))
    stacked = list(
        sorted(
            [fd for fd in frame_data if fd.stacked],
            key=lambda x: x.frameno,
        )
    )

    frame0 = frames[0].frameno
    framen = frames[-1].frameno
    stack0 = stacked[0].frameno
    stackn = stacked[-1].frameno

    sjd = get_sjd("LCO", frames[0].date_obs.to_datetime())

    fwhm_medians = [ff.fwhm_median for ff in stacked if ff.fwhm_median is not None]
    fwhm0 = numpy.round(fwhm_medians[0], 2)
    fwhmn = numpy.round(fwhm_medians[-1], 2)
    fwhm_median = numpy.round(numpy.median(fwhm_medians), 2)
    cofwhm = numpy.round(sources.fwhm.median(), 2)
    cofwhmst = numpy.round(sources.fwhm.std(), 2)

    zp = round(sources.zp.dropna().median(), 3)

    # Determine the PA drift due to k-mirror tracking.
    frame_wcs = [frame.wcs for frame in stacked if frame.wcs is not None]
    crota2 = numpy.array(list(map(get_crota2, frame_wcs)))

    # Do some sigma clipping to remove big outliers (usually due to WCS errors).
    if len(crota2) > 0:
        crota2_masked = sigma_clip(crota2, 5, masked=True)
        crota2_min = round(numpy.ma.min(crota2_masked), 6)
        crota2_max = round(numpy.ma.max(crota2_masked), 6)
        crota2_drift = round(abs(crota2_min - crota2_max), 6)
        drift_warning = crota2_drift > 0.1
    else:
        crota2_min = crota2_max = crota2_drift = None
        drift_warning = True

    wcs_header = wcs.to_header() if wcs is not None else []

    header = fits.Header()

    # Basic info
    header["TELESCOP"] = (frames[0].telescope, " Telescope that took the image")

    if not is_master:
        header["CAMNAME"] = (frames[0].camera, "Camera name")

    header["INSTRUME"] = ("LVM", "SDSS-V Local Volume Mapper")
    header["OBSERVAT"] = ("LCO", "Observatory")
    header["MJD"] = (sjd, "SDSS MJD (MJD+0.4)")

    if is_master:
        header["EXPTIME"] = (1.0, "[s] Exposure time")
        header["PIXSIZE"] = (9.0, "[um] Pixel size")
        header["PIXSCALE"] = (1.009, "[arcsec/pix]Scaled of unbinned pixel")

    header.insert("TELESCOP", ("", "/*** BASIC DATA ***/"))

    # Frame info
    header["FRAME0"] = (frame0, "First frame in guide sequence")
    header["FRAMEN"] = (framen, "Last frame in guide sequence")
    header["NFRAMES"] = (framen - frame0 + 1, "Number of frames in sequence")

    if not is_master:
        header["STACK0"] = (stack0, "First stacked frame")
        header["STACKN"] = (stackn, "Last stacked frame")
        header["NSTACKED"] = (stackn - stack0 + 1, "Number of frames stacked")

    if not is_master:
        header["COESTIM"] = ("median", "Estimator used to stack data")
        header["SIGCLIP"] = (use_sigmaclip, "Was the stack sigma-clipped?")
        header["SIGMA"] = (sigma, "Sigma used for sigma-clipping")

    header["OBSTIME0"] = (frames[0].date_obs.isot, "DATE-OBS of first frame")
    header["OBSTIMEN"] = (frames[-1].date_obs.isot, "DATE-OBS of last frame")
    header["FWHM0"] = (fwhm0, "[arcsec] FWHM of sources in first frame")
    header["FWHMN"] = (fwhmn, "[arcsec] FWHM of sources in last frame")
    header["FHHMMED"] = (fwhm_median, "[arcsec] Median of the FHWM of all frames")
    header["COFWHM"] = (cofwhm, "[arcsec] Co-added median FWHM")
    header["COFWHMST"] = (cofwhmst, "[arcsec] Co-added FWHM standard deviation")

    if not is_master:
        header["PAMIN"] = (crota2_min, "[deg] Minimum PA from WCS")
        header["PAMAX"] = (crota2_max, "[deg] Maximum PA from WCS")
        header["PADRIFT"] = (crota2_drift, "[deg] PA drift in frame range")

    header["ZEROPT"] = (zp, "[mag] Instrumental zero-point")
    header.insert("FRAME0", ("", "/*** CO-ADDED PARAMETERS ***/"))

    # Warnings
    header["WARNPADR"] = (drift_warning, "PA drift > 0.1 degrees")
    header["WARNTRAN"] = (False, "Transparency N magnitudes above photometric")
    header.insert("WARNPADR", ("", "/*** WARNINGS ***/"))

    header.extend(wcs_header)
    if is_master:
        header.insert("WCSAXES", ("", "/*** MASTER FRAME WCS ***/"))
    else:
        header.insert("WCSAXES", ("", "/*** CO-ADDED WCS ***/"))

    return header


def estimate_zeropoint(
    coadd_image: ARRAY_2D,
    sources: pandas.DataFrame,
    wcs: WCS,
    gain: float = 5,
    max_separation: float = 2,
    ap_radius: float = 6,
    ap_bkgann: tuple[float, float] = (6, 10),
    log: logging.Logger | None = None,
    db_connection_params: dict[str, Any] = {},
):
    """Determines the ``lmag`` zeropoint for each source.

    Parameters
    ----------
    sources
        A data frame with extracted sources. The data frame is returned
        after adding the aperture photometry and estimated zero-points.
        If the database connection fails or it is otherwise not possible
        to calculate zero-points, the original data frame is returned.
    wcs
        A WCS associated with the image. Used to determine the centre of the
        image and perform a radial query in the database.
    gain
        The gain of the detector in e-/ADU.
    max_separation
        Maximum separation between detections and matched Gaia sources, in arcsec.
    ap_radius
        The radius to use for aperture photometry, in pixels.
    ap_bkgann
        The inner and outer radii of the annulus used to determine the background
        around each source.
    log
        An instance of a logger to be used to output messages. If not provided
        defaults to the package log.
    db_connection_params
        A dictionary of DB connection parameters to pass to `.get_db_connection`.

    """

    orig_sources = sources.copy()

    # Camera FOV in degrees. Approximate, just for initial radial query.
    CAM_FOV = numpy.max(XZ_AG_FRAME) / 3600
    PIXSCALE = 1.009  # arcsec/pix

    # Epoch difference with Gaia DR3.
    GAIA_EPOCH = 2016.0
    epoch = Time.now().jyear
    epoch_diff = epoch - GAIA_EPOCH

    log = log or llog

    conn = get_db_connection(**db_connection_params)

    try:
        conn.connect()
    except OperationalError as err:
        log.error(f"Failed connecting to DB: {err}")
        return orig_sources

    # Get RA/Dec of centre of frame.
    skyc = wcs.pixel_to_world(*XZ_AG_FRAME)
    ra: float = skyc.ra.deg
    dec: float = skyc.dec.deg

    # Query lvm_magnitude and gaia_dr3_source in a radial query around RA/Dec.

    gdr3_sch, gdr3_table = config["database"]["gaia_dr3_source_table"].split(".")
    lmag_sch, lmag_table = config["database"]["lvm_magnitude_table"].split(".")
    GDR3 = peewee.Table(gdr3_table, schema=gdr3_sch).bind(conn)
    LMAG = peewee.Table(lmag_table, schema=lmag_sch).bind(conn)

    cte = (
        LMAG.select(
            LMAG.c.source_id,
            LMAG.c.lmag_ab,
            LMAG.c.lflux,
        )
        .where(peewee.fn.q3c_radial_query(LMAG.c.ra, LMAG.c.dec, ra, dec, CAM_FOV))
        .cte("cte", materialized=True)
    )
    query = (
        cte.select(
            cte.star,
            GDR3.c.ra,
            GDR3.c.dec,
            GDR3.c.pmra,
            GDR3.c.pmdec,
            GDR3.c.phot_g_mean_mag,
        )
        .join(GDR3, on=(cte.c.source_id == GDR3.c.source_id))
        .with_cte(cte)
        .dicts()
    )

    with conn.atomic():
        conn.execute_sql("SET LOCAL work_mem='2GB'")
        conn.execute_sql("SET LOCAL enable_seqscan=false")
        data = query.execute(conn)

    gaia_df = pandas.DataFrame.from_records(data)

    # Match detections with Gaia sources. Take into account proper motions.
    ra_gaia: ARRAY_1D = gaia_df.ra.to_numpy(numpy.float32)
    dec_gaia: ARRAY_1D = gaia_df.dec.to_numpy(numpy.float32)
    pmra: ARRAY_1D = gaia_df.pmra.to_numpy(numpy.float32)
    pmdec: ARRAY_1D = gaia_df.pmdec.to_numpy(numpy.float32)

    pmra_gaia = numpy.nan_to_num(pmra) / 1000 / 3600  # deg/yr
    pmdec_gaia = numpy.nan_to_num(pmdec) / 1000 / 3600

    ra_epoch = ra_gaia + pmra_gaia / numpy.cos(numpy.radians(dec)) * epoch_diff
    dec_epoch = dec_gaia + pmdec_gaia * epoch_diff

    gaia_df["ra_epoch"] = ra_epoch
    gaia_df["dec_epoch"] = dec_epoch

    # Reset index of sources.
    sources = sources.reset_index(drop=True)

    # Calculate x/y pixels of the Gaia detections. We use origin 0 but the
    # sep/SExtractor x/y in sources assume that the centre of the lower left
    # pixel is (1,1) so we adjust the returned pixel values.
    xpix, ypix = wcs.wcs_world2pix(ra_epoch, dec_epoch, 0)
    gaia_df["xpix"] = xpix + 0.5
    gaia_df["ypix"] = ypix + 0.5

    tree = KDTree(gaia_df.loc[:, ["xpix", "ypix"]].to_numpy())
    dd, ii = tree.query(sources.loc[:, ["x", "y"]].to_numpy())
    valid = dd < max_separation

    # Get Gaia rows for the valid matches. Change their indices to those
    # of their matching sources (which are 0..len(sources)-1 since we reindexed).
    matches = gaia_df.iloc[ii[valid]]
    matches.index = numpy.arange(len(ii))[valid]

    # Concatenate frames.
    sources = pandas.concat([sources, matches], axis=1)

    # Calculate the separation between matched sources. Drop xpix/ypix.
    dx = sources.xpix - sources.x
    dy = sources.ypix - sources.y
    sources["match_sep"] = numpy.hypot(dx, dy) * PIXSCALE
    sources.drop(columns=["xpix", "ypix"], inplace=True)

    # Add master frame pixels.
    xy = sources.loc[:, ["x", "y"]].to_numpy()
    camera = sources.iloc[0]["camera"]
    telescope = sources.iloc[0]["telescope"]
    mf_locs, _ = ag_to_master_frame(f"{telescope}-{camera[0]}", xy)
    sources.loc[:, ["x_mf", "y_mf"]] = mf_locs

    # Do aperture photometry around the detections.
    flux_adu, fluxerr_adu, _ = sep.sum_circle(
        coadd_image,
        sources.x,
        sources.y,
        ap_radius,
        bkgann=ap_bkgann,
        gain=gain,
    )

    # Calculate zero point. By definition this is the magnitude
    # of an object that produces 1 count per second on the detector.
    # For an arbitrary object producing DT counts per second then
    # m = -2.5 x log10(DN) - ZP
    zp = -2.5 * numpy.log10(flux_adu * gain) - sources.lmag_ab

    sources["ap_flux"] = flux_adu
    sources["ap_fluxerr"] = fluxerr_adu
    sources["zp"] = zp

    return sources


def get_guide_dataframe(frames: list[FrameData]):
    """Creates a data frame with guide information from ``proc-`` files."""

    guide_records: list[dict] = []
    _processed_proc_files: list[pathlib.Path] = []

    for frame in frames:
        proc_file = frame.proc_file

        if proc_file is None or not proc_file.exists():
            continue

        if proc_file in _processed_proc_files:
            continue

        hdul = fits.open(proc_file)
        if "ASTROMETRY" not in hdul or "SOURCES" not in hdul:
            continue

        astrom = hdul["ASTROMETRY"].header
        astrom["NAXIS"] = 2

        sources = pandas.DataFrame(hdul["SOURCES"].data)

        wcs = WCS(astrom)

        record = dict(
            frameno=get_frameno(astrom["FILE0"]),
            date=astrom["DATE"] if "DATE" in astrom else numpy.nan,
            n_sources=len(sources),
            fwhm=frame.fwhm_median,
            telescope=frame.telescope,
            ra=astrom["RAMEAS"],
            dec=astrom["DECMEAS"],
            ra_offset=astrom["OFFRAMEA"],
            dec_offset=astrom["OFFDEMEA"],
            separation=numpy.hypot(astrom["OFFRAMEA"], astrom["OFFDEMEA"]),
            ra_corr=astrom["RACORR"],
            dec_corr=astrom["DECORR"],
            ax0_corr=astrom["AX0CORR"],
            ax1_corr=astrom["AX1CORR"],
            mode="acquisition" if astrom["ACQUISIT"] else "guide",
            pa=get_crota2(wcs),
        )

        guide_records.append(record)

    df = pandas.DataFrame.from_records(guide_records)
    df.sort_values("frameno", inplace=True)

    return df


def framedata_to_dataframe(frame_data: list[FrameData]):
    """Converts a list of frame data to a data frame (!)."""

    records: list[dict] = []
    for fd in frame_data:
        records.append(
            dict(
                frameno=fd.frameno,
                date_obs=fd.date_obs.isot,
                camera=fd.camera,
                telescope=fd.telescope,
                exptime=fd.exptime,
                kmirror_drot=fd.kmirror_drot,
                focusdt=fd.focusdt,
                guide_error=fd.guide_error,
                fwhm_median=fd.fwhm_median,
                fwhm_std=fd.fwhm_std,
                guide_mode=fd.guide_mode,
                stacked=int(fd.stacked),
            )
        )

    df = pandas.DataFrame.from_records(records)
    df = df.sort_values(["frameno", "camera"])

    return df


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
        files = pathlib.Path(path).glob(f"lvm.{telescope}.agcam.*.fits")
    else:
        files = pathlib.Path(path).glob(f"lvm.{telescope}.agcam.{camera}_*.fits")

    selected: list[pathlib.Path] = []
    for file in files:
        frameno = get_frameno(file)
        if frameno >= frameno0 and frameno <= frameno1:
            selected.append(file)

    return list(sorted(selected))


def coadd_camera_frames(
    files: list[str | pathlib.Path],
    outpath: str | None = COADD_CAMERA_DEFAULT_PATH,
    use_sigmaclip: bool = False,
    sigma: float = 3.0,
    skip_acquisition_after_guiding: bool = False,
    database_profile: str = "default",
    database_params: dict = {},
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
        The list of files to be co-added. The must all correspond to a single
        camera and telescope.
    outpath
        The path of the co-added frame. If a relative path, it is written relative
        to the path of the first file to be co-added. If `None`, the file is not
        written to disk but the ``HDUList` is returned.
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
    database_profile
        Profile name to use to connect to the database and query Gaia.
    database_params
        Additional database parameters used to override the profile.
    log
        An instance of a logger to be used to output messages. If not provided
        defaults to the package log.
    verbose
        Increase verbosity of the logging.
    quiet
        Do not output log messages.

    Returns
    -------
    hdul,frame_data
        A tuple with the ``HDUList`` object with the co-added frame, and
        the frame data.

    """

    # Create log and set verbosity
    log = log or llog
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

    gain: float = sample_raw_header["GAIN"]
    ra: float = sample_raw_header["RA"]
    dec: float = sample_raw_header["DEC"]
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
            # RAW header
            raw_header = hdul["RAW"].header

            # Do some sanity checks.
            if raw_header["CAMNAME"] != camname:
                raise ValueError("Multiple cameras found in the list of frames.")

            if raw_header["TELESCOP"] != telescope:
                raise ValueError("Multiple telescopes found in the list of frames.")

            # PROC header of the raw AG frame.
            if "PROC" not in hdul:
                log.warning(f"Frame {frameno}: PROC extension not found.")
                proc_header = None
            else:
                proc_header = hdul["PROC"].header

            if proc_header is None or proc_header["WCSMODE"] == "none":
                wcs = None
            else:
                proc_header["NAXIS"] = 2
                wcs = WCS(proc_header)

            # Get the proc- file. Just used to determine
            # if we were acquiring or guiding.
            proc_file = get_proc_path(file)
            if not proc_file.exists():
                log.warning(f"Frame {frameno}: cannot find associated proc- image.")
                proc_astrometry = None
                guide_error = None
                proc_file = None
            else:
                proc_astrometry = fits.getheader(str(proc_file), "ASTROMETRY")
                guide_error = numpy.hypot(
                    proc_astrometry["OFFRAMEA"],
                    proc_astrometry["OFFDEMEA"],
                )

            # If we have not yet loaded the dark frame, get it and get the
            # normalised dark.
            if dark is None and proc_header and proc_header.get("DARKFILE", None):
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
            if proc_astrometry:
                guide_mode = "acquisition" if proc_astrometry["ACQUISIT"] else "guide"
                skip_acquisition = (
                    len(data_stack) == 0 or skip_acquisition_after_guiding
                )
                if guide_mode == "acquisition" and skip_acquisition:
                    stacked = False
                else:
                    stacked = True
                    data_stack.append(data)
            else:
                guide_mode = "none"
                stacked = False

            # Determine the median FWHM and FWHM deviation for this frame.
            if "SOURCES" not in hdul:
                log.warning(f"Frame {frameno}: SOURCES extension not found.")
                fwhm_median = None
                fwhm_std = None
            else:
                sources = pandas.DataFrame(hdul["SOURCES"].data)
                valid = sources.loc[(sources.xfitvalid == 1) & (sources.yfitvalid == 1)]

                fwhm = 0.5 * (valid.xstd + valid.ystd)
                fwhm_median = float(numpy.median(fwhm))
                fwhm_std = float(numpy.std(fwhm))

            # Add information as a FrameData. We do not include the data itself
            # because it's in data_stack and we don't need it beyond that.
            frame_data[frameno] = FrameData(
                file=pathlib.Path(file),
                frameno=frameno,
                raw_header=raw_header,
                proc_header=proc_header,
                proc_file=proc_file,
                camera=raw_header["CAMNAME"],
                telescope=raw_header["TELESCOP"],
                date_obs=Time(raw_header["DATE-OBS"], format="isot"),
                exptime=exptime,
                image_type=raw_header["IMAGETYP"],
                kmirror_drot=raw_header["KMIRDROT"],
                focusdt=raw_header["FOCUSDT"],
                fwhm_median=fwhm_median,
                fwhm_std=fwhm_std,
                guide_error=guide_error,
                guide_mode=guide_mode,
                stacked=stacked,
                wcs=wcs,
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

    # Extract sources in the co-added frame.
    coadd_sources = extract_marginal(
        coadd,
        box_size=31,
        threshold=3.0,
        max_detections=50,
        sextractor_quick_options={"minarea": 5},
    )
    coadd_sources["telescope"] = telescope
    coadd_sources["camera"] = camname

    # Get astrometry.net solution.
    camera_solution = solve_locs(
        coadd_sources.loc[:, ["x", "y", "flux"]],
        ra,
        dec,
        full_frame=False,
        raise_on_unsolved=False,
    )
    if camera_solution.solved is None:
        log.warning("Cannot determine astrometric solution for co-added frame.")
        wcs = None
        # TODO: if this fails we could still use kd-tree and Gaia but we need
        #       a better estimate of the RA/Dec of the centre of the camera.
    else:
        wcs = camera_solution.wcs
        db_connection_params = {"profile": database_profile, **database_params}
        coadd_sources = estimate_zeropoint(
            coadd,
            coadd_sources,
            wcs,
            gain=gain,
            log=log,
            db_connection_params=db_connection_params,
        )

    # Construct the header.
    header = create_coadded_frame_header(
        list(frame_data.values()),
        coadd_sources,
        use_sigmaclip=use_sigmaclip,
        sigma=int(sigma) if use_sigmaclip else None,
        wcs=wcs,
    )

    # Get frame data table.
    frame_data_df = framedata_to_dataframe(list(frame_data.values()))

    # Create the co-added HDU list.
    hdul = fits.HDUList([fits.PrimaryHDU()])
    hdul.append(fits.CompImageHDU(data=coadd, header=header, name="COADD"))
    hdul.append(fits.BinTableHDU(Table.from_pandas(coadd_sources), name="SOURCES"))
    hdul.append(fits.BinTableHDU(Table.from_pandas(frame_data_df), name="FRAMEDATA"))

    if outpath is not None:
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

        log.info(f"Writing co-added frame to {outpath_full.absolute()!s}")
        hdul.writeto(str(outpath_full), overwrite=True)

    hdul.close()

    return hdul, frame_data


def create_master_coadd(
    files: list[str | pathlib.Path],
    outpath: str | None = MASTER_COADD_DEFAULT_PATH,
    save_camera_coadded_frames: bool = False,
    log: logging.Logger | None = None,
    verbose: bool = False,
    quiet: bool = False,
    **coadd_camera_frames_kwargs,
):
    """Produces a master co-added frame.

    Calls `.coadd_camera_frames` with the frames for each camera and produces
    a single co-added file with the extensions from the camera co-added frames
    and an extension ``MASTER`` with the global astrometric solution of the
    master frame.

    Parameters
    ----------
    files
        The list of files to co-add. Must include frames from all the cameras for
        a given guide sequences.
    outpath
        The path of the master co-added frame. If a relative path, it is written
        relative to the path of the first file in the list. If `None`, the file
        is not written to disk but the ``HDUList` is returned.
    save_camera_coadded_frames
        Whether to write to disk the individual co-added images for each camera.
    log
        An instance of a logger to be used to output messages. If not provided
        defaults to the package log.
    verbose
        Increase verbosity of the logging.
    quiet
        Do not output log messages.
    coadd_camera_frames_kwargs
        Arguments to pass to `.coadd_camera_frames` for each set of camera frames.

    Returns
    -------
    hdul
        An ``HDUList`` object with the maste co-added frame.

    """

    # Create log and set verbosity
    log = log or llog
    for handler in log.handlers:
        if verbose:
            handler.setLevel(logging.DEBUG)
        elif quiet:
            handler.setLevel(logging.CRITICAL)

    coadd_camera_frames_kwargs = coadd_camera_frames_kwargs.copy()
    if save_camera_coadded_frames is False:
        coadd_camera_frames_kwargs["outpath"] = None

    coadd_camera_frames_kwargs["log"] = log
    coadd_camera_frames_kwargs["quiet"] = False
    coadd_camera_frames_kwargs["verbose"] = False

    camera_to_files: defaultdict[str, list[pathlib.Path]] = defaultdict(list)
    for file in files:
        for camera in ["east", "west"]:
            if camera in str(file):
                camera_to_files[camera].append(pathlib.Path(file))
                break

    cameras = [camera for camera in camera_to_files if len(camera_to_files[camera]) > 0]
    camera_files = [camera_to_files[camera] for camera in cameras]

    # TODO: this could be async but I'm getting some errors in astropy when
    #       trying it, so for now let's do sync.
    ccf_partial = partial(coadd_camera_frames, **coadd_camera_frames_kwargs)
    hduls = map(ccf_partial, camera_files)

    # Create master HDU list and concatenate sources. Create MASTER
    # extension but leave empty for now.
    master_hdu = fits.HDUList([fits.PrimaryHDU()])
    source_dfs: list[pandas.DataFrame] = []
    frame_data_all: list[FrameData] = []
    for ii, (hdul, frame_data) in enumerate(hduls):
        assert isinstance(hdul, fits.HDUList)

        source_dfs.append(pandas.DataFrame(hdul["SOURCES"].data))
        coadd_hdu = hdul["COADD"]
        coadd_hdu.name = f"COADD_{cameras[ii].upper()}"
        master_hdu.append(coadd_hdu)

        frame_data_all += frame_data.values()

    sources_coadd = pandas.concat(source_dfs)
    master_hdu.append(
        fits.BinTableHDU(
            data=Table.from_pandas(sources_coadd),
            name="SOURCES",
        )
    )

    # Now let's create the master frame WCS. It's easy since we already matched
    # with Gaia.
    wcs_data = sources_coadd.loc[:, ["x_mf", "y_mf", "ra_epoch", "dec_epoch"]]
    wcs_data = wcs_data.dropna()

    skycoords = SkyCoord(
        ra=wcs_data.ra_epoch,
        dec=wcs_data.dec_epoch,
        unit="deg",
        frame="icrs",
    )
    wcs_mf: WCS = fit_wcs_from_points((wcs_data.x_mf, wcs_data.y_mf), skycoords)

    # Initial MF HDU
    mf_header = create_coadded_frame_header(
        frame_data_all,
        sources_coadd,
        wcs=wcs_mf,
        is_master=True,
    )
    mf_hdu = fits.ImageHDU(name="MASTER", header=mf_header)
    master_hdu.insert(1, mf_hdu)

    # Add PA derived from MF WCS.
    crota2 = get_crota2(wcs_mf)
    mf_hdu.header.insert(
        "ZEROPT",
        ("PAWCS", round(crota2, 3), "[deg] PA of the IFU derived from WCS"),
    )

    guide_data = get_guide_dataframe(frame_data_all)
    frame_data_t = Table.from_pandas(framedata_to_dataframe(frame_data_all))

    master_hdu.append(fits.BinTableHDU(Table.from_pandas(guide_data), name="GUIDEDATA"))
    master_hdu.append(fits.BinTableHDU(frame_data_t, name="FRAMEDATA"))

    # Calculate PA drift. Do some sigma clipping to remove big outliers.
    pa_values: ARRAY_1D = guide_data.pa.dropna().to_numpy(numpy.float32)
    pa_values = cast(ARRAY_1D, sigma_clip(pa_values, 10, masked=True))
    pa_min = round(pa_values.min(), 6)
    pa_max = round(pa_values.max(), 6)
    pa_drift = abs(pa_max - pa_min)

    mf_hdu.header.insert("ZEROPT", ("PAMIN", pa_min, "[deg] Minimum PA from WCS"))
    mf_hdu.header.insert("ZEROPT", ("PAMAX", pa_max, "[deg] Maximum PA from WCS"))
    mf_hdu.header.insert("ZEROPT", ("PADRIFT", pa_drift, "[deg] PA drift in sequence"))

    # If any of the cameras measured PA drift, set the warning.
    padrift_warn: bool = False
    for cam in ["EAST", "WEST"]:
        if master_hdu[f"COADD_{cam}"].header["WARNPADR"]:
            padrift_warn = True
            break
    mf_hdu.header["WARNPADR"] = padrift_warn

    if outpath is not None:
        # Create the path for the output file.
        sample_header = master_hdu[f"COADD_{cameras[0].upper()}"].header
        frameno0 = sample_header["FRAME0"]
        frameno1 = sample_header["FRAMEN"]
        telescope = sample_header["TELESCOP"]

        outpath = outpath.format(
            telescope=telescope,
            frameno0=frameno0,
            frameno1=frameno1,
        )

        if pathlib.Path(outpath).is_absolute():
            outpath_full = pathlib.Path(outpath)
        else:
            outpath_full = pathlib.Path(files[0]).parent / outpath

        if outpath_full.parent.exists() is False:
            outpath_full.parent.mkdir(parents=True, exist_ok=True)

        log.info(f"Writing co-added frame to {outpath_full.absolute()!s}")
        master_hdu.writeto(str(outpath_full), overwrite=True)

    return master_hdu


def get_guider_files_from_spec(
    spec_file: str | pathlib.Path,
    telescope: str | None = None,
    agcam_path: str | pathlib.Path = "/data/agcam",
    camera: str | None = None,
):
    """Returns the AG files taken during a spectrograph exposure.

    A convenience function that reads the header of the spectrograph frame,
    extracts the guide frames range and returns a list of those frames
    filenames (missing frames are skipped).

    Parameters
    ----------
    spec_file
        The path to the spectrograph frame to use to determine the range.
    telescope
        The telescope for which to extract the range of guide/AG frames.
        Defaults to the value in the configuration file.
    agcam_path
        The ``agcam`` path where AG frames are ordered by SJD.
    camera
        The camera, east or west, for which to select frames. If not provided
        selects both cameras.

    Returns
    -------
    frames
        A list of AG file names that match the range of guide frames
        taken during the spectrograph exposure. Missing frames in the
        range are not included.

    """

    spec_file = pathlib.Path(spec_file)
    if not spec_file.exists():
        raise FileNotFoundError(f"Cannot find file {spec_file!s}")

    header = fits.getheader(str(spec_file))

    if telescope is None:
        telescope = config["telescope"]

    assert isinstance(telescope, str)

    sjd = get_sjd("LCO", date=Time(header["OBSTIME"], format="isot").to_datetime())
    frame0 = header.get(f"G{telescope.upper()}FR0", None)
    frame1 = header.get(f"G{telescope.upper()}FRN", None)

    if frame0 is None or frame1 is None:
        raise ValueError(f"Cannot determine the guider frame range for {spec_file!s}")

    agcam_path = pathlib.Path(agcam_path) / str(sjd)
    if not agcam_path.exists():
        raise FileNotFoundError(f"Cannot find agcam path {agcam_path!s}")

    return get_frame_range(agcam_path, telescope, frame0, frame1, camera=camera)
