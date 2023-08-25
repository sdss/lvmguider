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
from sqlite3 import OperationalError

from typing import Any

import nptyping as npt
import numpy
import pandas
import peewee
import sep
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from scipy.spatial import KDTree

from sdsstools.time import get_sjd

from lvmguider import config
from lvmguider import log as llog
from lvmguider.extraction import extract_marginal
from lvmguider.tools import get_db_connection, get_frameno, get_proc_path
from lvmguider.transformations import XZ_AG_FRAME, get_crota2, solve_locs


ARRAY_2D_UINT = npt.NDArray[npt.Shape["*, *"], npt.UInt16]
ARRAY_2D = npt.NDArray[npt.Shape["*, *"], npt.Float32]


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
    guide_error: float | None = None
    fwhm_median: float | None = None
    fwhm_std: float | None = None
    guide_mode: str = "guide"
    stacked: bool = False
    wcs: WCS | None = None


def create_coadded_frame_header(
    frame_data: dict[int, FrameData],
    sources: pandas.DataFrame,
    use_sigmaclip: bool = False,
    sigma: int | None = None,
    wcs: WCS | None = None,
):
    """Creates the header object for a co-added frame."""

    # Create a list with only the stacked frames and sort by frame number.
    frames = list(
        sorted(
            [fd for fd in frame_data.values() if fd.stacked],
            key=lambda x: x.frameno,
        )
    )

    sjd = get_sjd("LCO", frames[0].date_obs.to_datetime())

    fwhm0 = numpy.round(frames[0].fwhm_median, 2) if frames[0].fwhm_median else None
    fwhmn = numpy.round(frames[-1].fwhm_median, 2) if frames[-1].fwhm_median else None
    cofwhm = numpy.round(sources.fwhm.median(), 2)
    cofwhmst = numpy.round(sources.fwhm.std(), 2)

    guide_errors = numpy.array([f.guide_error for f in frames if f.guide_error])
    guide_error_mean = round(float(guide_errors.mean()), 3)
    guide_error_std = round(float(guide_errors.std()), 3)

    zp = round(sources.zp.dropna().median(), 3)

    # Determine the PA drift due to k-mirror tracking.
    frame_wcs = [frame.wcs for frame in frames if frame.wcs is not None]
    crota2 = numpy.array(list(map(get_crota2, frame_wcs)))
    crota2[crota2 < 0] += 360

    # Do some sigma clipping to remove big outliers (usually due to WCS errors).
    crota2_masked = sigma_clip(crota2, 5, masked=True)
    crota2_min = numpy.ma.min(crota2_masked)
    crota2_max = numpy.ma.max(crota2_masked)
    crota2_drift = abs(crota2_min - crota2_max)

    wcs_header = wcs.to_header() if wcs is not None else []

    header = fits.Header()

    header["TELESCOP"] = (frames[0].telescope, " Telescope that took the image")
    header["CAMNAME"] = (frames[0].camera, "Camera name")
    header["INSTRUME"] = ("LVM", "SDSS-V Local Volume Mapper")
    header["OBSERVAT"] = ("LCO", "Observatory")
    header["MJD"] = (sjd, "SDSS MJD (MJD+0.4)")
    header["EXPTIME"] = (1.0, "[s] Exposure time")
    header["PIXSIZE"] = (9.0, "[um] Pixel size")
    header["PIXSCALE"] = (1.009, "[arcsec/pix]Scaled of unbinned pixel")
    header.insert("TELESCOP", ("", "/*** BASIC DATA ***/"))

    header["FRAME0"] = (frames[0].frameno, "First co-added frame")
    header["FRAMEN"] = (frames[-1].frameno, "Last co-added frame")
    header["NFRAMES"] = (len(frames), "Number of frames stacked")
    header["COESTIM"] = ("median", "Estimator used to stack data")
    header["SIGCLIP"] = (use_sigmaclip, "Was the stack sigma-clipped?")
    header["SIGMA"] = (sigma, "Sigma used for sigma-clipping")
    header["OBSTIME0"] = (frames[0].date_obs.isot, "DATE-OBS of first frame")
    header["OBSTIMEN"] = (frames[-1].date_obs.isot, "DATE-OBS of last frame")
    header["FWHM0"] = (fwhm0, "[arcsec] FWHM of sources in first frame")
    header["FWHMN"] = (fwhmn, "[arcsec] FWHM of sources in last frame")
    header["COFWHM"] = (cofwhm, "[arcsec] Co-added FWHM")
    header["COFWHMST"] = (cofwhmst, "[arcsec] Co-added FWHM standard deviation")
    header["GERRMEAN"] = (guide_error_mean, "[arcsec] Mean of guider errors")
    header["GERRSTD"] = (guide_error_std, "[arcsec] Deviation of guider errors")
    header["PAMIN"] = (round(crota2_min, 4), "[deg] Minimum PA from WCS")
    header["PAMAX"] = (round(crota2_max, 4), "[deg] Maximum PA from WCS")
    header["PADRIFT"] = (round(crota2_drift, 4), "[deg] PA drift in frame range")
    header["ZEROPT"] = (zp, "[mag] Instrumental zero-point")
    header.insert("FRAME0", ("", "/*** CO-ADDED PARAMETERS ***/"))

    header["WARNGUID"] = (guide_error_mean > 3, "Mean guide error > 3 arcsec")
    header["WARNPADR"] = (crota2_drift > 0.1, "PA drift > 0.1 degrees")
    header.insert("WARNGUID", ("", "/*** WARNINGS ***/"))

    header.extend(wcs_header)
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
    db_connection_params: dict[str, Any] | None = None,
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
        If `None`, uses the default configuration.

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

    db_connection_params = db_connection_params or {}
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
    ra_gaia = gaia_df.ra
    dec_gaia = gaia_df.dec
    pmra_gaia = gaia_df.pmra / 1000 / 3600  # deg/yr
    pmdec_gaia = gaia_df.pmdec / 1000 / 3600

    ra_epoch = ra_gaia + pmra_gaia / numpy.cos(numpy.radians(dec)) * epoch_diff
    dec_epoch = dec_gaia + pmdec_gaia * epoch_diff

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
        defaults to the package log.
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

            # PROC header of the raw AG frame.
            if "PROC" not in hdul:
                log.warning(f"Frame {frameno}: PROC extension not found.")
                proc_header = None
            else:
                proc_header = hdul["PROC"].header

            if proc_header is None or proc_header["WCSMODE"] == "none":
                wcs = None
            else:
                wcs = WCS(proc_header)

            # Get the proc- file. Just used to determine
            # if we were acquiring or guiding.
            proc_file = get_proc_path(file)
            if not proc_file.exists():
                log.warning(f"Frame {frameno}: cannot find associated proc- image.")
                continue

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
            guide_mode = "acquisition" if proc_astrometry["ACQUISIT"] else "guide"
            skip_acquisition = len(data_stack) == 0 or skip_acquisition_after_guiding
            if guide_mode == "acquisition" and skip_acquisition:
                stacked = False
            else:
                stacked = True
                data_stack.append(data)

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
        coadd_sources = estimate_zeropoint(
            coadd,
            coadd_sources,
            wcs,
            gain=gain,
            log=log,
        )

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
    header = create_coadded_frame_header(
        frame_data,
        coadd_sources,
        use_sigmaclip=use_sigmaclip,
        sigma=int(sigma) if use_sigmaclip else None,
        wcs=wcs,
    )

    # Write the file.
    log.debug(f"Writing co-added frame to {outpath_full.absolute()!s}")
    hdul = fits.HDUList([fits.PrimaryHDU()])
    hdul.append(fits.CompImageHDU(data=coadd, header=header, name="COADD"))
    hdul.append(fits.BinTableHDU(data=Table.from_pandas(coadd_sources), name="SOURCES"))
    hdul.writeto(str(outpath_full), overwrite=True)
    hdul.close()

    return hdul
