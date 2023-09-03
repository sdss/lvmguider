#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-08-23
# @Filename: coadd.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import multiprocessing
import os
import pathlib
from functools import partial

from typing import TYPE_CHECKING, Sequence

import numpy
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.time import Time

from sdsstools.time import get_sjd

from lvmguider import config, log
from lvmguider.dataclasses import (
    CameraSolution,
    CoAdd_CameraSolution,
    FrameData,
    GlobalSolution,
    GuiderSolution,
)
from lvmguider.extraction import extract_marginal
from lvmguider.tools import (
    angle_difference,
    estimate_zeropoint,
    get_dark_subtrcted_data,
    get_frameno,
    get_guider_files_from_spec,
    header_from_model,
    nan_or_none,
    polyfit_with_sigclip,
    sort_files_by_camera,
)
from lvmguider.transformations import ag_to_master_frame, match_with_gaia, wcs_from_gaia
from lvmguider.types import ARRAY_2D_F32


if TYPE_CHECKING:
    pass


AnyPath = str | os.PathLike


def process_all_spec_frames(path: AnyPath, **kwargs):
    """Processes all the spectrograph frames in a directory."""

    path = pathlib.Path(path)
    spec_files = sorted(path.glob("sdR-*-b1-*.fits.gz"))

    for file in spec_files:
        # Skip cals and frames without any guider information.
        header = fits.getheader(str(file))

        if header["IMAGETYP"] != "object":
            continue

        coadd_from_spec_frame(file, fail_silent=True, **kwargs)


def coadd_from_spec_frame(
    file: AnyPath,
    outpath: str | None = None,
    telescopes: list[str] = ["sci", "spec", "skye", "skyw"],
    fail_silent: bool = False,
    use_time_range: bool | None = None,
    **kwargs,
):
    """Processes the guider frames associated with an spectrograph file."""

    outpath = outpath or config["coadds"]["paths"]["coadd_spec_path"]

    file = pathlib.Path(file)
    if not file.exists():
        raise FileExistsError(f"File {file!s} not found.")

    specno = int(file.name.split("-")[-1].split(".")[0])
    outpath = outpath.format(specno=specno)

    for telescope in telescopes:
        try:
            frames = get_guider_files_from_spec(
                file,
                telescope=telescope,
                use_time_range=use_time_range,
            )
        except Exception as err:
            if fail_silent is False:
                log.error(f"Cannot process {telescope} for {file!s}: {err}")
            continue

        if len(frames) < 4:
            if fail_silent is False:
                log.warning(f"No guider frames found for {telescope!r} in {file!s}")
            continue

        log.info(
            f"Generating co-added frame for {telescope!r} for "
            f"spectrograph file {file!s}"
        )

        create_coadd(frames, telescope=telescope, outpath=outpath, **kwargs)


def create_coadd(
    files: Sequence[AnyPath],
    telescope: str,
    outpath: str | None = "default",
    save_camera_coadded: bool = False,
    **coadd_camera_kwargs,
):
    """Produces a master co-added frame.

    Calls `.coadd_camera` with the frames for each camera and produces
    a single co-added file with the extensions from the camera co-added frames
    and an extension ``COADD`` with the global astrometric solution of the
    master frame.

    Parameters
    ----------
    files
        The list of files to co-add. Must include frames from all the cameras for
        a given guider sequence.
    telescope
        The telescope associated with the images.
    outpath
        The path of the master co-added frame. If a relative path, it is written
        relative to the path of the first file in the list. If `False`, the file
        is not written to disk but the ``HDUList` is returned. With ``"default"``,
        the default output path is used.
    save_camera_coadded
        Whether to write to disk the individual co-added images for each camera.
    coadd_camera_kwargs
        Arguments to pass to `.coadd_camera` for each set of camera frames.

    Returns
    -------
    global_solution
        The `.GlobalSolution` instance with the information about the global
        frame and solution.

    """

    if outpath == "default":
        outpath = config["coadds"]["paths"]["coadd_path"]

    coadd_camera_kwargs = coadd_camera_kwargs.copy()
    if save_camera_coadded is False:
        coadd_camera_kwargs["outpath"] = None

    camera_to_files = sort_files_by_camera(files)
    cameras = [camera for camera in camera_to_files if len(camera_to_files[camera]) > 0]
    camera_files = [camera_to_files[camera] for camera in cameras]

    frame_nos = set([get_frameno(file) for file in files])

    # Co-add each camera independently.
    with multiprocessing.Pool(2) as pool:
        coadd_camera_partial = partial(coadd_camera, **coadd_camera_kwargs)
        coadd_solutions = list(pool.map(coadd_camera_partial, camera_files))

    # Now create a global solution.
    root = pathlib.Path(files[0]).parent
    guider_solutions = get_guider_solutions(root, list(frame_nos), telescope)

    gs = GlobalSolution(
        coadd_solutions=coadd_solutions,
        guider_solutions=guider_solutions,
        telescope=telescope,
    )

    # Fit a new WCS from the individual co-added solutions using the master frame.
    sources = gs.sources
    if len(sources.ra.dropna()) > 5:
        gs.wcs = wcs_from_gaia(sources, xy_cols=["x_mf", "y_mf"])
    else:
        log.warning("Unable to fit global WCS. Not enough matched sources.")

    if outpath is not None:
        # Create the path for the output file.
        date_obs = fits.getval(files[0], "DATE-OBS", "RAW")
        sjd = get_sjd("LCO", Time(date_obs, format="isot").to_datetime())

        frameno0 = min(frame_nos)
        frameno1 = max(frame_nos)
        outpath = outpath.format(
            telescope=telescope,
            frameno0=frameno0,
            frameno1=frameno1,
            mjd=sjd,
        )

        if pathlib.Path(outpath).is_absolute():
            path = pathlib.Path(outpath)
        else:
            path = root / outpath

        if path.parent.exists() is False:
            path.parent.mkdir(parents=True, exist_ok=True)

        gs.path = path

        log.info(f"Writing co-added frame to {path.absolute()!s}")

        # Write global header and co-added frames.
        hdul = fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(
                    data=None,
                    header=create_global_header(gs),
                    name="GLOBAL",
                ),
            ]
        )

        for coadd in gs.coadd_solutions:
            hdul.append(
                fits.CompImageHDU(
                    data=coadd.coadd_image,
                    header=create_coadd_header(coadd),
                )
            )

        hdul.writeto(str(path), overwrite=True)

        # Write sources to file.
        if gs.sources is not None:
            sources_path = path.with_name(path.stem + "_sources.parquet")
            log.debug(f"Writing co-added sources to {sources_path!s}")
            gs.sources.to_parquet(sources_path)

        # Write guider data.
        guider_data = gs.guider_data()
        frames_path = path.with_name(path.stem + "_guiderdata.parquet")
        log.debug(f"Writing guide data to {frames_path!s}")
        guider_data.to_parquet(frames_path)

        # Write frame data.
        frame_data = gs.frame_data()
        frames_path = path.with_name(path.stem + "_frames.parquet")
        log.debug(f"Writing frame data to {frames_path!s}")
        frame_data.to_parquet(frames_path)

    return gs


def coadd_camera(
    files: Sequence[AnyPath],
    outpath: str | None = "default",
    use_sigmaclip: bool = False,
    sigma: float = 3.0,
    database_profile: str = "default",
    database_params: dict = {},
):
    """Co-adds a series of AG camera frames.

    This routine does the following:

    - Receives a range of guider frames that usually correspond to a full guider
      sequence.
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
        written to disk but the ``HDUList` is returned. If the value is
        ``"default"``, a default path is used.
    use_sigmaclip
        Whether to use sigma clipping when combining the stack of data. Disabled
        by default as it uses significant CPU and memory.
    sigma
        The sigma value for co-addition sigma clipping.
    database_profile
        Profile name to use to connect to the database and query Gaia.
    database_params
        Additional database parameters used to override the profile.

    Returns
    -------
    camera_solution
        The `.CoAdd_CameraSolution` object holding the information about
        the co-added frame.

    """

    db_connection_params = {"profile": database_profile, **database_params}

    if outpath == "default":
        outpath = config["coadds"]["paths"]["coadd_camera_path"]

    # Get the list of frame numbers from the files.
    paths = sorted([pathlib.Path(file) for file in files])

    frame_nos = sorted([get_frameno(path) for path in paths])

    # Use the first file to get some common data (or at least it should be common!)
    sample_raw_header = fits.getheader(paths[0], "RAW")

    camname: str = sample_raw_header["CAMNAME"]
    telescope: str = sample_raw_header["TELESCOP"]
    dateobs0: str = sample_raw_header["DATE-OBS"]
    sjd = get_sjd("LCO", date=Time(dateobs0, format="isot").to_datetime())

    log.info(
        f"Co-adding frames {min(frame_nos)}-{max(frame_nos)} for MJD={sjd}, "
        f"camera={camname!r}, telescope={telescope!r}."
    )

    # Loop over each file, add the dark-subtracked data to the stack, and collect
    # frame metadata.
    _frames = list(map(create_framedata, paths))
    frames = [frame for frame in _frames if frame is not None]

    data_stack: list[ARRAY_2D_F32] = []
    for fd in frames:
        if fd.data is not None and fd.stacked is True:
            data_stack.append(fd.data)
        fd.data = None

    coadd_image: ARRAY_2D_F32 | None = None
    coadd_solution: CoAdd_CameraSolution

    if len(data_stack) == 0:
        if telescope == "spec":
            log.warning(f"Not stacking data for telecope {telescope!r}.")
        else:
            log.error(f"No data to stack for {telescope!r}.")

        coadd_solution = CoAdd_CameraSolution(
            frameno=-1,
            camera=camname,
            telescope=telescope,
        )

    else:
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
            coadd_image = numpy.ma.median(stack_masked, axis=0).data

        else:
            log.debug("Creating median-combined co-added frame.")
            coadd_image = numpy.median(numpy.array(data_stack), axis=0)

        del data_stack
        assert coadd_image is not None

        coadd_solution = process_coadd(
            coadd_image,
            frames,
            db_connection_params=db_connection_params,
        )

    coadd_solution.frames = frames
    coadd_solution.sigmaclip = use_sigmaclip
    coadd_solution.sigmaclip_sigma = sigma if use_sigmaclip else None

    if outpath is not None:
        # Create the path for the output file.
        frameno0 = min(frame_nos)
        frameno1 = max(frame_nos)
        outpath = outpath.format(
            telescope=telescope,
            camname=camname,
            frameno0=frameno0,
            frameno1=frameno1,
            mjd=sjd,
        )

        if pathlib.Path(outpath).is_absolute():
            path = pathlib.Path(outpath)
        else:
            path = pathlib.Path(paths[0]).parent / outpath

        if path.parent.exists() is False:
            path.parent.mkdir(parents=True, exist_ok=True)

        coadd_solution.path = path

        log.info(f"Writing co-added frame to {path.absolute()!s}")

        # Write co-addded image and header.
        hdul = fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.CompImageHDU(
                    data=coadd_image,
                    header=create_coadd_header(coadd_solution),
                    name="COADD",
                ),
            ]
        )
        hdul.writeto(str(path), overwrite=True)

        # Write sources to file.
        if coadd_solution.sources is not None:
            sources_path = path.with_name(path.stem + "_sources.parquet")
            log.debug(f"Writing co-added sources to {sources_path!s}")
            coadd_solution.sources.to_parquet(sources_path)

        # Write frame data.
        frame_data = coadd_solution.frame_data()
        frames_path = path.with_name(path.stem + "_frames.parquet")
        log.debug(f"Writing frame data to {frames_path!s}")
        frame_data.to_parquet(frames_path)

    return coadd_solution


def create_framedata(path: pathlib.Path):
    """Collects information from a guider frame into a `.FrameData` object."""

    if not path.exists():
        log.error(f"Cannot find frame {path!s}")
        return None

    with fits.open(str(path)) as hdul:
        # RAW header
        raw_header = hdul["RAW"].header

        frameno = get_frameno(path)
        telescope = raw_header["TELESCOP"]
        camname = raw_header["CAMNAME"]
        date_obs = Time(raw_header["DATE-OBS"], format="isot")

        log_h = f"Frame {telescope}-{camname}-{frameno}:"

        # PROC header of the raw AG frame.
        if "PROC" not in hdul:
            log.error(f"{log_h} cannot find PROC extension.")
            return None

        proc_header = hdul["PROC"].header

        data, dark_sub = get_dark_subtrcted_data(path)
        if dark_sub is False:
            log.debug(f"{log_h} missing dark frame. Fitting and removing background.")

        wcs_mode = proc_header["WCSMODE"]
        stacked = wcs_mode == "gaia"

        try:
            solution = CameraSolution.open(path)
        except ValueError as err:
            log.error(f"{log_h} failed creating CameraSolution ({err})")
            return None

        if solution.sources is None:
            log.warning(f"{log_h} no sources found.")

    # Add information as a FrameData.
    return FrameData(
        frameno=frameno,
        path=path,
        solution=solution,
        raw_header=dict(raw_header),
        camera=camname,
        telescope=telescope,
        date_obs=date_obs,
        sjd=get_sjd("LCO", date_obs.to_datetime()),
        exptime=raw_header["EXPTIME"],
        image_type=raw_header["IMAGETYP"],
        kmirror_drot=raw_header["KMIRDROT"],
        focusdt=raw_header["FOCUSDT"],
        stacked=stacked,
        data=data,
    )


def process_coadd(
    data: ARRAY_2D_F32,
    framedata: list[FrameData],
    db_connection_params={},
):
    """Processes co-added data, extracts sources, and determines the WCS."""

    frame: FrameData | None = None
    for fd in framedata:
        if fd.solution.solved and fd.solution.wcs_mode == "gaia":
            frame = fd
            break

    if frame is None or frame.solution.wcs is None:
        raise RuntimeError("Cannot find WCS solutions for coadded frame.")

    # Extract sources in the co-added frame.
    coadd_sources = extract_marginal(
        data,
        box_size=31,
        threshold=3.0,
        max_detections=50,
        sextractor_quick_options={"minarea": 5},
    )
    coadd_sources["telescope"] = frame.telescope
    coadd_sources["camera"] = frame.camera

    # Add master frame pixels.
    xy = coadd_sources.loc[:, ["x", "y"]].to_numpy()
    mf_locs, _ = ag_to_master_frame(f"{frame.telescope}-{frame.camera[0]}", xy)
    coadd_sources.loc[:, ["x_mf", "y_mf"]] = mf_locs

    # Match with Gaia sources.
    coadd_sources, n_matches = match_with_gaia(
        frame.solution.wcs,
        coadd_sources,
        max_separation=3,
        concat=True,
        db_connection_params=db_connection_params,
    )

    if n_matches < 5:
        log.warning("Insufficient number of Gaia matches. Cannot generate WCS.")
        wcs = None

    else:
        # Get WCS solution from Gaia.
        wcs = wcs_from_gaia(coadd_sources)

    zp = estimate_zeropoint(data, coadd_sources)
    coadd_sources.update(zp)

    return CoAdd_CameraSolution(
        frameno=-1,
        coadd_image=data,
        telescope=frame.telescope,
        camera=frame.camera,
        sources=coadd_sources,
        wcs=wcs,
        wcs_mode="gaia",
        matched=wcs is not None,
    )


def get_guider_solutions(root: pathlib.Path, framenos: list[int], telescope: str):
    """Collects guider solutions for a range of framenos."""

    solutions: list[GuiderSolution] = []

    for frameno in framenos:
        filename = f"lvm.{telescope}.guider_{frameno:08d}.fits"

        path = root / filename
        if not path.exists():
            log.error(f"Cannot find guider solution. File {path!s} does not exist.")

        guider_data = fits.getheader(path, "GUIDERDATA")

        solution = GuiderSolution.open(path)
        solution.guide_mode = guider_data["GUIDMODE"]
        solution.ra_field = guider_data["RAFIELD"]
        solution.dec_field = guider_data["DECFIELD"]
        solution.pa_field = guider_data.get("PAFIELD", numpy.nan)

        solutions.append(solution)

    return solutions


def create_coadd_header(solution: CoAdd_CameraSolution):
    """Creates the header object for a co-added frame."""

    assert solution.frames is not None
    assert solution.sources is not None

    telescope = solution.telescope

    frame_data = solution.frame_data()
    stacked = frame_data.loc[frame_data.stacked == 1, :]

    frame0 = frame_data.iloc[0].frameno
    framen = frame_data.iloc[-1].frameno
    stack0 = stacked.iloc[0].frameno
    stackn = stacked.iloc[-1].frameno

    isot = frame_data.iloc[0].date_obs
    sjd = get_sjd("LCO", Time(isot, format="isot").to_datetime())

    # Gets the range of FWHM in the sequence
    fwhm0: float | None = None
    fwhmn: float | None = None
    fwhm_median: float | None = None
    fwhm_medians = stacked.fwhm.dropna()
    if len(fwhm_medians) > 0:
        fwhm0 = fwhm_medians.iloc[0]
        fwhmn = fwhm_medians.iloc[-1]
        fwhm_median = float(numpy.median(fwhm_medians))

    # Gets the FWHM of the sources extracted from the stacked image.
    cofwhm = solution.fwhm
    cofwhmst = solution.sources.loc[solution.sources.valid == 1].fwhm.dropna().median()

    # Determine the PA drift due to k-mirror tracking.
    frame_pa = stacked.loc[:, ["frameno", "pa"]].dropna()
    pa_min: float = numpy.nan
    pa_max: float = numpy.nan
    pa_drift: float = numpy.nan

    pa_coeffs = numpy.array([numpy.nan, numpy.nan])
    drift_warn: bool = True

    if telescope == "spec":
        pa_min = frame_pa.pa.min()
        pa_max = frame_pa.pa.max()
        pa_drift = numpy.abs(pa_min - pa_max)
        drift_warn = False

    elif len(frame_pa) >= 2:
        pa_coeffs = polyfit_with_sigclip(
            frame_pa.frameno.to_numpy(numpy.float32),
            frame_pa.pa.to_numpy(numpy.float32),
            sigma=3,
            deg=1,
        )

        pa_vals = numpy.polyval(pa_coeffs, frame_pa.frameno)
        pa_min = numpy.round(pa_vals.min(), 4)
        pa_max = numpy.round(pa_vals.max(), 4)
        pa_drift = numpy.round(numpy.abs(pa_min - pa_max), 4)

        drift_warn = pa_drift > config["coadds"]["warnings"]["pa_drift"]

    wcs_header = solution.wcs.to_header() if solution.wcs is not None else []

    header = header_from_model("CAMERA_COADD")

    # Basic info
    header["TELESCOP"] = frame_data.iloc[0].telescope
    header["CAMNAME"] = frame_data.iloc[0].camera
    header["MJD"] = sjd
    header.insert("TELESCOP", ("", "/*** BASIC DATA ***/"))

    # Frame info
    header["FRAME0"] = frame0
    header["FRAMEN"] = framen
    header["NFRAMES"] = framen - frame0 + 1

    if telescope != "spec":
        header["STACK0"] = stack0
        header["STACKN"] = stackn
        header["NSTACKED"] = stackn - stack0 + 1
        header["COESTIM"] = "median"
        header["SIGCLIP"] = solution.sigmaclip
        header["SIGMA"] = solution.sigmaclip_sigma

    header["OBSTIME0"] = frame_data.iloc[0].date_obs
    header["OBSTIMEN"] = frame_data.iloc[-1].date_obs
    header["FWHM0"] = nan_or_none(fwhm0, 3)
    header["FWHMN"] = nan_or_none(fwhmn, 3)
    header["FHHMMED"] = nan_or_none(fwhm_median, 3)

    if telescope != "spec":
        header["COFWHM"] = nan_or_none(cofwhm, 3)
        header["COFWHMST"] = nan_or_none(cofwhmst, 3)

        header["PACOEFFA"] = nan_or_none(pa_coeffs[0])
        header["PACOEFFB"] = nan_or_none(pa_coeffs[1])

    header["PAMIN"] = nan_or_none(pa_min, 4)
    header["PAMAX"] = nan_or_none(pa_max, 4)
    header["PADRIFT"] = nan_or_none(pa_drift, 4)

    header["ZEROPT"] = nan_or_none(solution.zero_point, 3)

    header["SOLVED"] = solution.solved
    header.insert("FRAME0", ("", "/*** CO-ADDED PARAMETERS ***/"))

    # Warnings
    header["WARNPADR"] = drift_warn
    header["WARNTRAN"] = False
    header["WARNMATC"] = not solution.matched if telescope != "spec" else False
    header.insert("WARNPADR", ("", "/*** WARNINGS ***/"))

    if solution.wcs is not None:
        header.extend(wcs_header)
        header.insert("WCSAXES", ("", "/*** CO-ADDED WCS ***/"))

    return header


def create_global_header(solution: GlobalSolution):
    """Creates the header object for a global frame.

    This function is very similar to `.create_coadd_header` but there are
    enough differences that it's worth keeping them separate and duplicate
    a bit of code.

    """

    assert solution.sources is not None

    telescope = solution.telescope

    frame_data = solution.frame_data()
    guider_data = solution.guider_data()

    guiding = guider_data.loc[guider_data.guide_mode == "guide"]

    frame0 = frame_data.frameno.min()
    framen = frame_data.frameno.max()

    isot = frame_data.iloc[0].date_obs
    sjd = get_sjd("LCO", Time(isot, format="isot").to_datetime())

    # Gets the range of FWHM in the sequence
    fwhm0: float | None = None
    fwhmn: float | None = None
    fwhm_median: float | None = None
    fwhm_medians = guiding.fwhm.dropna()
    if len(fwhm_medians) > 0:
        fwhm0 = fwhm_medians.iloc[0]
        fwhmn = fwhm_medians.iloc[-1]
        fwhm_median = float(numpy.median(fwhm_medians))

    # Determine the PA drift due to k-mirror tracking.
    frame_pa = guiding.loc[:, ["frameno", "pa"]].dropna()
    pa_min: float = numpy.nan
    pa_max: float = numpy.nan
    pa_drift: float = numpy.nan

    pa_coeffs = numpy.array([numpy.nan, numpy.nan])
    drift_warn: bool = True

    if telescope == "spec":
        pa_min = frame_pa.pa.min()
        pa_max = frame_pa.pa.max()
        pa_drift = numpy.abs(pa_min - pa_max)
        drift_warn = False

    elif len(frame_pa) >= 2:
        pa_coeffs = polyfit_with_sigclip(
            frame_pa.frameno.to_numpy(numpy.float32),
            frame_pa.pa.to_numpy(numpy.float32),
            sigma=3,
            deg=1,
        )

        pa_vals = numpy.polyval(pa_coeffs, frame_pa.frameno)
        pa_min = numpy.round(pa_vals.min(), 4)
        pa_max = numpy.round(pa_vals.max(), 4)
        pa_drift = numpy.round(numpy.abs(pa_min - pa_max), 4)

        drift_warn = pa_drift > config["coadds"]["warnings"]["pa_drift"]

    wcs_header = []
    pa_warn: bool = False
    if solution.wcs is not None:
        wcs_header = solution.wcs.to_header()

        pa_field = guider_data.pa_field.iloc[0]
        pa = solution.pa
        pa_warn_threshold = config["coadds"]["warnings"]["pa_error"]
        pa_warn = abs(angle_difference(pa, pa_field)) > pa_warn_threshold

    header = header_from_model("GLOBAL_COADD")

    # Basic info
    header["TELESCOP"] = frame_data.iloc[0].telescope
    header["CAMNAME"] = frame_data.iloc[0].camera
    header["MJD"] = sjd
    header.insert("TELESCOP", ("", "/*** BASIC DATA ***/"))

    # Frame info
    header["FRAME0"] = frame0
    header["FRAMEN"] = framen
    header["NFRAMES"] = framen - frame0 + 1

    header["OBSTIME0"] = frame_data.iloc[0].date_obs
    header["OBSTIMEN"] = frame_data.iloc[-1].date_obs

    header["FWHM0"] = nan_or_none(fwhm0, 3)
    header["FWHMN"] = nan_or_none(fwhmn, 3)
    header["FHHMMED"] = nan_or_none(fwhm_median, 3)

    if telescope != "spec":
        header["PACOEFFA"] = nan_or_none(pa_coeffs[0])
        header["PACOEFFB"] = nan_or_none(pa_coeffs[1])

    header["PAMIN"] = nan_or_none(pa_min, 4)
    header["PAMAX"] = nan_or_none(pa_max, 4)
    header["PADRIFT"] = nan_or_none(pa_drift, 4)

    header["ZEROPT"] = nan_or_none(solution.zero_point, 3)

    header["SOLVED"] = solution.solved
    header.insert("FRAME0", ("", "/*** GLOBAL FRAME PARAMETERS ***/"))

    # Pointing
    if telescope != "spec":
        header["XMFPIX"] = guider_data.x_mf_pixel.iloc[0]
        header["ZMFPIX"] = guider_data.z_mf_pixel.iloc[0]
        header["RAFIELD"] = nan_or_none(guider_data.ra_field.iloc[0], 6)
        header["DECFIELD"] = nan_or_none(guider_data.dec_field.iloc[0], 6)
        header["PAFIELD"] = nan_or_none(guider_data.pa_field.iloc[0], 4)
        header["RAMEAS"] = nan_or_none(guider_data.ra.iloc[0], 6)
        header["DECMEAS"] = nan_or_none(guider_data.dec.iloc[0], 6)
        header["PAMEAS"] = nan_or_none(guider_data.pa.iloc[0], 4)

    # Warnings
    header["WARNPA"] = pa_warn
    header["WARNPADR"] = drift_warn
    header["WARNTRAN"] = False
    header.insert("WARNPA", ("", "/*** WARNINGS ***/"))

    if solution.wcs is not None:
        header.extend(wcs_header)
        header.insert("WCSAXES", ("", "/*** CO-ADDED WCS ***/"))

    return header
