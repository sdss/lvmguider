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
import os.path
import pathlib
from functools import partial

from typing import TYPE_CHECKING, Literal, Sequence

import numpy
import pandas
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.time import Time
from packaging.version import Version

from sdsstools.time import get_sjd

from lvmguider import config, log
from lvmguider.dataclasses import (
    CameraSolution,
    CoAdd_CameraSolution,
    FrameData,
    GlobalSolution,
    GuiderSolution,
)
from lvmguider.extraction import extract_marginal, extract_sources
from lvmguider.tools import (
    angle_difference,
    estimate_zeropoint,
    get_dark_subtracted_data,
    get_frameno,
    get_guider_files_from_spec,
    header_from_model,
    nan_or_none,
    polyfit_with_sigclip,
    sort_files_by_camera,
)
from lvmguider.transformations import (
    ag_to_full_frame,
    get_crota2,
    match_with_gaia,
    solve_camera_with_astrometrynet,
    wcs_from_gaia,
)
from lvmguider.types import ARRAY_2D_F32


if TYPE_CHECKING:
    from lvmguider.astrometrynet import AstrometrySolution


AnyPath = str | os.PathLike


MULTIPROCESS_MODE: Literal["frames"] | Literal["cameras"] = "frames"
MULTIPROCESS_NCORES: dict[str, int] = {"frames": 12, "cameras": 2}


def process_all_spec_frames(path: AnyPath, **kwargs):
    """Processes all the spectrograph frames in a directory.

    Parameters
    ----------
    path
        The path to the directory to process. This is usually a
        ``/data/spectro/<MJD>`` directory.
    kwargs
        Keyword arguments to pass to `.coadd_from_spec_frame`.

    """

    path = pathlib.Path(path)
    spec_files = sorted(path.glob("sdR-*-b1-*.fits.gz"))

    for file in spec_files:
        # Skip cals and frames without any guider information.

        header = fits.getheader(str(file))

        if header["IMAGETYP"] != "object":
            continue

        if header["EXPTIME"] < 120:
            log.warning(f"Spec frame {file.name!s} is too short. Skipping.")
            continue

        coadd_from_spec_frame(file, **kwargs)


def coadd_from_spec_frame(
    file: AnyPath,
    outpath: str | None = None,
    telescopes: list[str] = ["sci", "spec", "skye", "skyw"],
    use_time_range: bool | None = None,
    **kwargs,
):
    """Processes the guider frames associated with an spectrograph file.

    Parameters
    ----------
    file
        The spectro file to process. Guider frames will be selected
        from the header keywords in the file.
    output
        The path where to save the global co-added frame. If `None`,
        uses the default path which includes the frame number of the
        spectro file.
    telescopes
        The list of telescope guider frames to process.
    use_time_range
        If `True`, selects guider frames from their timestamps instead
        of using the headers in the spectro file.
    kwargs
        Keyword arguments to pass to `.create_global_coadd`.

    """

    outpath = outpath or config["coadds"]["paths"]["coadd_spec_path"]

    file = pathlib.Path(file)
    if not file.exists():
        raise FileExistsError(f"File {file!s} not found.")

    specno = int(file.name.split("-")[-1].split(".")[0])
    outpath = outpath.format(specno=specno)

    for telescope in telescopes:
        log.info(
            f"Generating co-added frame for {telescope!r} for "
            f"spectrograph file {file!s}"
        )

        try:
            log.debug("Identifying guider frames.")
            frames = get_guider_files_from_spec(
                file,
                telescope=telescope,
                use_time_range=use_time_range,
            )
        except Exception as err:
            log.error(f"Cannot process {telescope!r} for {file!s}: {err}")
            continue

        if len(frames) < 4:
            log.warning(f"No guider frames found for {telescope!r} in {file!s}")
            continue

        try:
            create_global_coadd(frames, telescope=telescope, outpath=outpath, **kwargs)
        except Exception as err:
            log.critical(
                f"Failed generating co-added frames for {file!s} on "
                f"telescope {telescope!r}: {err}"
            )


def create_global_coadd(
    files: Sequence[AnyPath],
    telescope: str,
    outpath: str | None = "default",
    save_camera_coadded: bool = False,
    **coadd_camera_kwargs,
):
    """Produces a global co-added frame.

    Calls `.coadd_camera` with the frames for each camera and produces
    a single co-added file with the extensions from the camera co-added frames
    and an extension ``COADD`` with the global astrometric solution of the
    full frame.

    Parameters
    ----------
    files
        The list of files to co-add. Must include frames from all the cameras for
        a given guider sequence.
    telescope
        The telescope associated with the images.
    outpath
        The path of the global co-added frame. If a relative path, it is written
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
    coadd_camera_partial = partial(coadd_camera, **coadd_camera_kwargs)

    if MULTIPROCESS_MODE == "cameras":
        with multiprocessing.Pool(MULTIPROCESS_NCORES["cameras"]) as pool:
            coadd_solutions = list(pool.map(coadd_camera_partial, camera_files))
    else:
        coadd_solutions = list(map(coadd_camera_partial, camera_files))

    # Now create a global solution.
    root = pathlib.Path(files[0]).parent
    guider_solutions = get_guider_solutions(root, list(frame_nos), telescope)

    gs = GlobalSolution(
        coadd_solutions=coadd_solutions,
        guider_solutions=guider_solutions,
        telescope=telescope,
    )

    # Fit a new WCS from the individual co-added solutions using the full frame.
    sources = gs.sources
    if telescope != "spec":
        if len(sources.ra.dropna()) > 5:
            gs.wcs = wcs_from_gaia(sources, xy_cols=["x_ff", "y_ff"])
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
    create_framedata_partial = partial(
        create_framedata,
        db_connection_params=db_connection_params,
    )

    if MULTIPROCESS_MODE == "frames":
        with multiprocessing.Pool(MULTIPROCESS_NCORES["frames"]) as pool:
            _frames = list(pool.map(create_framedata_partial, paths))
    else:
        _frames = list(map(create_framedata_partial, paths))

    frames = [frame for frame in _frames if frame is not None]

    data_stack: list[ARRAY_2D_F32] = []
    for fd in frames:
        if fd.data is not None and fd.stacked is True:
            data_stack.append(fd.data)
        fd.data = None

    coadd_image: ARRAY_2D_F32 | None = None
    coadd_solution: CoAdd_CameraSolution

    if len(data_stack) == 0 or telescope == "spec":
        if telescope == "spec":
            log.debug(f"Not stacking data for telecope {telescope!r}.")
        else:
            log.error(f"No data to stack for {telescope!r}.")

        coadd_solution = CoAdd_CameraSolution(
            frameno=-1,
            camera=camname,
            telescope=telescope,
        )

    else:
        log.debug(f"Creating median-combined co-added frame for {camname!r}.")

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
            coadd_image = numpy.ma.median(stack_masked, axis=0).data

        else:
            coadd_image = numpy.median(numpy.array(data_stack), axis=0)

        del data_stack
        assert coadd_image is not None

        coadd_solution = process_camera_coadd(
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


def create_framedata(path: pathlib.Path, db_connection_params: dict = {}):
    """Collects information from a guider frame into a `.FrameData` object."""

    # Copy the original path.
    orig_path: pathlib.Path = path
    reprocessed: bool = False

    if not path.exists():
        log.error(f"Cannot find frame {path!s}")
        return None

    try:
        guiderv = fits.getval(str(path), "GUIDERV", "PROC")
    except Exception:
        guiderv = "0.0.0"

    if Version(guiderv) < Version("0.3.0a0"):
        path = reprocess_agcam(
            path,
            db_connection_params=db_connection_params,
        )
        reprocessed = True

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

        # We use the original path here always because reprocessed
        # files do not include image data.
        data, dark_sub = get_dark_subtracted_data(orig_path)
        if dark_sub is False:
            log.debug(f"{log_h} missing dark frame. Fitting and removing background.")

        wcs_mode = proc_header["WCSMODE"]
        stacked = (wcs_mode == "gaia") or reprocessed

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
        reprocessed=reprocessed,
    )


def process_camera_coadd(
    data: ARRAY_2D_F32,
    framedata: list[FrameData],
    db_connection_params={},
):
    """Processes co-added data, extracts sources, and determines the WCS."""

    frame: FrameData | None = None
    for fd in framedata[::-1]:  # In reverse order. Last file is likely to be guiding.
        if fd.solution.solved and (fd.solution.wcs_mode == "gaia" or fd.reprocessed):
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

    # Add global frame pixels.
    xy = coadd_sources.loc[:, ["x", "y"]].to_numpy()
    ff_locs, _ = ag_to_full_frame(f"{frame.telescope}-{frame.camera[0]}", xy)
    coadd_sources.loc[:, ["x_ff", "y_ff"]] = ff_locs

    # Match with Gaia sources.
    coadd_sources, n_matches = match_with_gaia(
        frame.solution.wcs,
        coadd_sources,
        max_separation=3,
        concat=True,
        db_connection_params=db_connection_params,
    )

    wcs = None
    if n_matches < 5:
        log.warning("Insufficient number of Gaia matches. Cannot generate WCS.")
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

        log_h = f"{telescope}-{frameno}:"

        path = root / filename
        if not path.exists():
            try:
                path = reprocess_legacy_guider_frame(root, frameno, telescope)
            except Exception:
                log.error(f"{log_h} failed retrieving guider solution.")
                continue

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

    if len(frame_data) == 0 or len(stacked) == 0:
        raise ValueError("No stacked data found.")

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

        if pa_field is not None:
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
        header["XFFPIX"] = guider_data.x_ff_pixel.iloc[0]
        header["ZFFPIX"] = guider_data.z_ff_pixel.iloc[0]
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


def reprocess_agcam(file: AnyPath, overwrite: bool = False, db_connection_params={}):
    """Reprocesses an old file and converts it to the ``>=0.4.0`` format.

    .. warning::
        This function reprocesses a file on a best-effort basis. The astrometric
        solution is calculated using astrometry.net, which may be different
        from the original solution.

    Paremeters
    ----------
    file
        The file to reprocess.
    overwrite
        Exits early if the reprocessed file already exists.
    db_connection_params
        Database connection details to query Gaia.

    Returns
    -------
    processed
        The path to the processed file. The processed file is written relative
        to the parent of the input file as ``reprocessed/<file>``. Note that
        the reprocessed file does not include the raw original data to
        save space since that would not have changed. Reprocessed files are
        assigned ``GUIDERV=0.99.0``.

    """

    file = pathlib.Path(file)
    reproc_path = file.parent / "reprocessed" / file.name
    sources_path = reproc_path.with_suffix(".parquet")

    if reproc_path.exists() and not overwrite:
        log.debug(f"Found reprocessed file {reproc_path!s}")
        return reproc_path

    log.warning(f"File {file!s} is too old and will be reprocessed.")

    hdul_orig = fits.open(file)

    if "RAW" in hdul_orig:
        raw = hdul_orig["RAW"].header
    else:
        raw = hdul_orig[0].header

    proc = hdul_orig["PROC"].header if "PROC" in hdul_orig else {}

    sources = extract_sources(file)

    solution: AstrometrySolution = solve_camera_with_astrometrynet(
        sources,
        ra=raw["RA"],
        dec=raw["DEC"],
    )
    wcs = solution.wcs

    # Now match with Gaia.
    if solution.wcs is not None:
        sources, _ = match_with_gaia(
            solution.wcs,
            sources,
            concat=True,
            db_connection_params=db_connection_params,
        )

    darkfile = proc.get("DARKFILE", None)
    if darkfile:
        darkfile = os.path.basename(darkfile)

    new_proc = header_from_model("PROC")
    new_proc["TELESCOP"] = raw["TELESCOP"]
    new_proc["CAMNAME"] = raw["CAMNAME"]
    new_proc["DARKFILE"] = darkfile
    new_proc["DIRNAME"] = str(file.parent)
    new_proc["GUIDERV"] = "0.99.0"
    new_proc["SOURCESF"] = sources_path.name
    new_proc["PA"] = nan_or_none(get_crota2(wcs), 4) if wcs else None
    new_proc["ZEROPT"] = nan_or_none(sources.zp.dropna().median())
    new_proc["WCSMODE"] = "none" if wcs is None else "astrometrynet"
    new_proc["ORIGFILE"] = (file.name, "Original file name")
    new_proc["REPROC"] = (True, "Has this file been reprocessed?")

    if wcs is not None:
        new_proc.extend(wcs.to_header())

    reproc_path.parent.mkdir(parents=True, exist_ok=True)

    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(header=raw, name="RAW"),
            fits.ImageHDU(header=new_proc, name="PROC"),
        ]
    )
    hdul.writeto(str(reproc_path), overwrite=True)
    hdul.close()

    sources.to_parquet(sources_path)

    return reproc_path


def reprocess_legacy_guider_frame(
    root: pathlib.Path,
    frameno: int,
    telescope: str,
    overwrite: bool = False,
):
    """Reprocesses a legacy ``proc-`` file.

    .. warning::
        This function reprocesses a file on a best-effort basis. Usually
        the individual frames used to generate the solution have also been
        reprocessed and the results may differ to those actually observed
        on sky.

    Paremeters
    ----------
    root
        The root path (usually the ``agcam`` directory for the given MJD).
    frameno
        The frame number.
    telescope
        The telescope that took the images.
    overwrite
        If `False` and a reprocessed ``lvm.guider`` file has been generated,
        uses that one.

    Returns
    -------
    processed
        The path to the processed file. The processed file is written relative
        to the parent of the input file as
        ``reprocessed/lvm,{telescope}.guider_{frameno:08d}.fits``. Reprocessed
        files are assigned ``GUIDERV=0.99.0``. The sources ``.parquet`` table
        file is also generated.


    """

    log_h = f"{telescope}-{frameno}:"

    proc_file = root / f"proc-lvm.{telescope}.agcam_{frameno:08d}.fits"
    guider_file = root / "reprocessed" / f"lvm.{telescope}.guider_{frameno:08d}.fits"
    guider_sources_file = guider_file.with_suffix(".parquet")

    if not proc_file.exists():
        raise FileNotFoundError(f"Cannot find file {proc_file!s}")

    if overwrite is False and guider_file.exists():
        log.debug(f"{log_h} found reprocessed lvm.guider file {guider_file!s}.")
        return guider_file

    log.warning(f"{log_h} using legacy proc- file {proc_file!s}.")

    proc = dict(fits.getheader(proc_file, "ASTROMETRY"))

    gdata_header = header_from_model("GUIDERDATA_HEADER")
    for key in gdata_header:
        if key in proc:
            gdata_header[key] = proc[key]

    gdata_header["GUIDERV"] = "0.99.0"
    gdata_header["DIRNAME"] = str(guider_file.parent)
    gdata_header["SOURCESF"] = guider_sources_file.name

    gdata_header["XFFPIX"] = config["xz_full_frame"][0]
    gdata_header["ZFFPIX"] = config["xz_full_frame"][1]

    gdata_header["GUIDMODE"] = "acquisition" if proc["ACQUISIT"] else "guide"

    # Get the sources. Try first the location of the new-style sources file.
    # Otherwise check if there's a reprocessed file.
    sources: list[pandas.DataFrame] = []
    for key in ["FILE0", "FILE1"]:
        filex = proc.get(key, None)
        if filex is None:
            continue

        filex = pathlib.Path(filex).name

        found: bool = False
        if (root / filex).with_suffix(".parquet").exists():
            sources_file = (root / filex).with_suffix(".parquet")
            sources.append(pandas.read_parquet(sources_file))
            found = True

        elif (root / "reprocessed" / filex).with_suffix(".parquet").exists():
            sources_file = (root / "reprocessed" / filex).with_suffix(".parquet")
            sources.append(pandas.read_parquet(sources_file))
            found = True

        if found:
            if "east" in filex:
                gdata_header["FILEEAST"] = filex
            else:
                gdata_header["FILEWEST"] = filex

    if len(sources) == 0:
        raise RuntimeError(f"No sources found for guider frame {proc_file!s}")

    sources_concat = pandas.concat(sources)
    if len(sources_concat.ra.dropna()) > 5:
        wcs = wcs_from_gaia(sources_concat, xy_cols=["x_ff", "y_ff"])
    else:
        log.warning(f"Insufficient Gaia matches for {proc_file!s}. Cannot fit WCS.")
        wcs = None

    if wcs is not None:
        gdata_header.extend(wcs.to_header())

        gdata_header["PAMEAS"] = nan_or_none(get_crota2(wcs), 4)
        gdata_header["ZEROPT"] = nan_or_none(sources_concat.zp.dropna().median(), 3)
        gdata_header["SOLVED"] = True

    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(header=gdata_header, name="GUIDERDATA"),
        ]
    )
    hdul.writeto(str(guider_file), overwrite=True)

    sources_concat.to_parquet(guider_sources_file)

    return guider_file
