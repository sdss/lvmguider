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
import re
import time
from functools import partial
from threading import Lock

from typing import TYPE_CHECKING, Literal, Sequence

import numpy
import pandas
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.time import Time
from packaging.version import Version
from watchdog.events import (
    FileCreatedEvent,
    FileMovedEvent,
    PatternMatchingEventHandler,
)
from watchdog.observers.polling import PollingObserver

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
from lvmguider.plotting import plot_qa
from lvmguider.tools import (
    angle_difference,
    dataframe_to_database,
    estimate_zeropoint,
    get_dark_subtracted_data,
    get_frameno,
    get_guider_files_from_spec,
    get_raw_extension,
    get_spec_frameno,
    header_from_model,
    isot_to_sjd,
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

MULTIPROCESS_MODE_TYPE = Literal["frames"] | Literal["cameras"] | Literal["telescopes"]
MULTIPROCESS_NCORES: dict[str, int] = {"frames": 4, "cameras": 2, "telescopes": 4}

TELESCOPES: list[str] = ["sci", "spec", "skye", "skyw"]


def process_all_spec_frames(
    path: AnyPath,
    create_summary: bool = True,
    write_to_database: bool = True,
    multiprocess_mode: MULTIPROCESS_MODE_TYPE | None = "telescopes",
    **kwargs,
):
    """Processes all the spectrograph frames in a directory.

    Parameters
    ----------
    path
        The path to the directory to process. This is usually a
        ``/data/spectro/<MJD>`` directory.
    create_summary
        Generates a single table file with all the guider data for
        the processed files.
    write_to_database
        Whether to load the co-added data to the database.
    multiprocess_mode
        Defines how to use parallel processing when processing multiple
        images. Options are ``telescopes`` which will process each telescope
        for a spectrograph image in parallel, ``cameras``, which will process
        each camera in parallel, and ``frames`` which will process frames within
        one camera in parallel. The number of cores used by each one of
        these modes can be adjusted by modifying the `.MULTIPROCESS_NCORES`
        dictionary.
    kwargs
        Keyword arguments to pass to `.coadd_from_spec_frame`.

    """

    path = pathlib.Path(path)
    spec_files = sorted(path.glob("sdR-*-b1-*.fits.gz"))

    if len(spec_files) == 0:
        return

    date_obs = fits.getval(spec_files[0], "OBSTIME")
    sjd = get_sjd("LCO", Time(date_obs, format="isot").to_datetime())

    all_paths: dict[int, list[pathlib.Path]] = {}

    for file in spec_files:
        header = fits.getheader(str(file))
        specno = get_spec_frameno(file)

        # Skip cals and frames without any guider information.
        if header["IMAGETYP"] != "object":
            continue

        paths = coadd_from_spec_frame(
            file,
            multiprocess_mode=multiprocess_mode,
            write_to_database=write_to_database,
            **kwargs,
        )
        all_paths[specno] = paths

    if create_summary:
        gs_paths: list[pathlib.Path] = []
        spec_nos: list[int] = []

        for spec_no in all_paths:
            gs_paths += all_paths[spec_no]
            spec_nos += [spec_no] * len(all_paths[spec_no])

        g_fs = [pp.with_name(pp.stem + "_guiderdata.parquet") for pp in gs_paths]
        g_fs = [path for path in g_fs if path.exists()]
        if len(g_fs) > 0:
            base_path = g_fs[0].parent
            outpath = base_path / f"lvm.guider.mjd_{sjd}_guiderdata.parquet"
            log.info(f"Saving MJD summary file to {outpath!s}")
            create_summary_file(g_fs, outpath, list(spec_nos))

        f_fs = [pp.with_name(pp.stem + "_frames.parquet") for pp in gs_paths]
        f_fs = [path for path in f_fs if path.exists()]
        if len(f_fs) > 0:
            base_path = f_fs[0].parent
            outpath = base_path / f"lvm.guider.mjd_{sjd}_frames.parquet"
            log.info(f"Saving MJD summary file to {outpath!s}")
            create_summary_file(f_fs, outpath, list(spec_nos))


def coadd_telescope(
    file: pathlib.Path,
    outpath: str | None,
    telescope: str,
    use_time_range: bool = False,
    write_to_database: bool = True,
    **kwargs,
):
    """A helper function that processes one telescopes for a spectrograph image.

    This function is called by `.coadd_from_spec_frame` for a given ``file``
    and ``telescope``. It's split from the code from that function to allow
    parallelisation.

    """

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
        return

    if len(frames) < 4:
        log.warning(f"No guider frames found for {telescope!r} in {file!s}")
        return

    exposure_match = re.search("[0-9]{8}", file.name)
    exposure_no = int(exposure_match.group(0)) if exposure_match else None

    try:
        gs = create_global_coadd(
            frames,
            telescope=telescope,
            outpath=outpath,
            **kwargs,
        )

        if gs.path is not None:
            if write_to_database and exposure_no is not None:
                coadd_to_database(gs.path, exposure_no=exposure_no)
            return gs.path

    except Exception as err:
        log.critical(
            f"Failed generating co-added frames for {file!s} on "
            f"telescope {telescope!r}: {err}"
        )
        return None


def coadd_from_spec_frame(
    file: AnyPath,
    outpath: str | None = None,
    telescopes: list[str] = TELESCOPES,
    use_time_range: bool | None = None,
    write_to_database: bool = True,
    multiprocess_mode: MULTIPROCESS_MODE_TYPE | None = "telescopes",
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
    multiprocess_mode
        See `.process_all_spec_frames`.
    kwargs
        Keyword arguments to pass to `.create_global_coadd`.

    """

    outpath = outpath or config["coadds"]["paths"]["coadd_spec_path"]

    file = pathlib.Path(file)
    if not file.exists():
        raise FileExistsError(f"File {file!s} not found.")

    specno = get_spec_frameno(file)
    outpath = outpath.format(specno=specno)

    paths: list[pathlib.Path | None] = []

    coadd_telescope_p = partial(
        coadd_telescope,
        file,
        outpath,
        use_time_range=use_time_range,
        multiprocess_mode=multiprocess_mode,
        write_to_database=write_to_database,
        **kwargs,
    )
    if multiprocess_mode == "telescopes":
        with multiprocessing.Pool(MULTIPROCESS_NCORES["telescopes"]) as pool:
            paths += pool.map(coadd_telescope_p, telescopes)
    else:
        for telescope in telescopes:
            paths.append(coadd_telescope_p(telescope))

    return [path for path in paths if path is not None]


def create_global_coadd(
    files: Sequence[AnyPath],
    telescope: str,
    outpath: str | None = "default",
    save_camera_coadded: bool = False,
    generate_qa: bool = True,
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
        It includes the ``multiprocess_mode`` parameter which if set to ``cameras``
        will process each camera in parallel.

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

    multiprocess_mode = coadd_camera_kwargs.get("multiprocess_mode", None)
    if multiprocess_mode == "cameras":
        with multiprocessing.Pool(MULTIPROCESS_NCORES["cameras"]) as pool:
            coadd_solutions = list(pool.map(coadd_camera_partial, camera_files))
    else:
        coadd_solutions = list(map(coadd_camera_partial, camera_files))

    # Now create a global solution.
    root = pathlib.Path(files[0]).parent

    get_guider_solutions_p = partial(get_guider_solutions, root, telescope)
    if multiprocess_mode == "cameras":
        with multiprocessing.Pool(MULTIPROCESS_NCORES["cameras"]) as pool:
            guider_solutions = list(pool.map(get_guider_solutions_p, list(frame_nos)))
    else:
        guider_solutions = map(get_guider_solutions_p, list(frame_nos))

    gs = GlobalSolution(
        coadd_solutions=coadd_solutions,
        guider_solutions=[gs for gs in guider_solutions if gs is not None],
        telescope=telescope,
    )

    # Fit a new WCS from the individual co-added solutions using the full frame.
    sources = gs.sources
    if telescope != "spec":
        if len(sources.ra.dropna()) > 5:
            gs.wcs = wcs_from_gaia(sources, xy_cols=["x_ff", "z_ff"])
        else:
            log.warning("Unable to fit global WCS. Not enough matched sources.")

    if outpath is not None:
        # Create the path for the output file.
        sample_raw_hdu = get_raw_extension(files[0])
        date_obs = sample_raw_hdu.header["DATE-OBS"]
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
                    name=f"COADD_{coadd.camera.upper()}",
                )
            )

        hdul.writeto(str(path), overwrite=True)

        # Write sources to file.
        if gs.sources is not None:
            sources_path = path.with_name(path.stem + "_sources.parquet")
            log.debug(f"Writing co-added sources to {sources_path!s}")
            gs.sources.reset_index(drop=True, inplace=True)
            gs.sources.to_parquet(sources_path)

        # Write guider data.
        guider_data = gs.guider_data()
        guider_path = path.with_name(path.stem + "_guiderdata.parquet")
        log.debug(f"Writing guide data to {guider_path!s}")
        guider_data.reset_index(drop=True, inplace=True)
        guider_data.to_parquet(guider_path)

        # Write frame data.
        frame_data = gs.frame_data()
        frames_path = path.with_name(path.stem + "_frames.parquet")
        log.debug(f"Writing frame data to {frames_path!s}")
        frame_data.reset_index(drop=True, inplace=True)
        frame_data.to_parquet(frames_path)

        if generate_qa:
            log.info(f"Generating QA plots for {path.absolute()!s}")
            plot_qa(gs)

    return gs


def create_summary_file(
    files: Sequence[AnyPath],
    outpath: AnyPath,
    spec_framenos: list[int] | None = None,
):
    """Concatenates the guider data from a list of files.

    Parameters
    ----------
    files
        The list of files to concatenate. These must be the
        ``lvm.{telescope}.guider_{seqno}.parquet`` with one row corresponding
        to each guide step in the sequence.
    outpath
        The ``parquet`` table file where to save the data.
    spec_framenos
        A list of spectrograph frame numbers. The length must match
        the number of ``files``. If provided, a column ``spec_frameno`` will
        be added to each file table.

    Returns
    -------
    data
        A data frame with the concatenated data.

    """

    dfs: list[pandas.DataFrame] = []
    for ii, file in enumerate(files):
        df = pandas.read_parquet(file)

        if spec_framenos is not None and len(spec_framenos) > 0:
            df["spec_frameno"] = spec_framenos[ii]
            df["spec_frameno"] = df["spec_frameno"].astype("Int16")

        dfs.append(df)

    df_concat = pandas.concat(dfs)
    df_concat.sort_values(["telescope", "frameno"])

    outpath = pathlib.Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df_concat.reset_index(drop=True, inplace=True)
    df_concat.to_parquet(str(outpath))

    return df_concat


def coadd_camera(
    files: Sequence[AnyPath],
    outpath: str | None = "default",
    use_sigmaclip: bool = False,
    sigma: float = 3.0,
    database_profile: str = "default",
    database_params: dict = {},
    overwrite_reprocess: bool = False,
    multiprocess_mode: MULTIPROCESS_MODE_TYPE | None = None,
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
    overwrite_reprocess
        Whether to overwrite reprocessed files or use already existing ones.
    multiprocess_mode
        See `.process_all_spec_frames`.

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
    sample_raw_header = get_raw_extension(paths[0]).header

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
        overwrite_reprocess=overwrite_reprocess,
    )

    if multiprocess_mode == "frames":
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

    if len(data_stack) == 0:
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
            coadd_solution.sources.reset_index(drop=True, inplace=True)
            coadd_solution.sources.to_parquet(sources_path)

        # Write frame data.
        frame_data = coadd_solution.frame_data()
        frames_path = path.with_name(path.stem + "_frames.parquet")
        log.debug(f"Writing frame data to {frames_path!s}")
        frame_data.reset_index(drop=True, inplace=True)
        frame_data.to_parquet(frames_path)

    return coadd_solution


def create_framedata(
    path: pathlib.Path,
    db_connection_params: dict = {},
    overwrite_reprocess: bool = False,
):
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
        guiderv = "0.0.1"

    # Guider version 0.0.1 means that there is not PROC extension.
    # Guider version 0.0.0 means that the image was never processed by
    #    the guider but otherwise has the right format so it does not need to
    #    be reprocessed. Images with 0.0.0 have WCSMODE=none.
    # Guider version 0.99.0 means it has been reprocessed.

    if guiderv != "0.0.0" and Version(guiderv) < Version("0.4.0a0"):
        new_path = reprocess_agcam(
            path,
            db_connection_params=db_connection_params,
            overwrite=overwrite_reprocess,
        )
        if new_path is not None:
            reprocessed = True
            path = new_path

    with fits.open(str(path)) as hdul:
        # RAW header
        if "RAW" in hdul:
            raw_header = hdul["RAW"].header
        else:
            raw_header = hdul[0].header

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

        wcs_mode = proc_header["WCSMODE"].lower()
        stacked = (wcs_mode == "gaia") or reprocessed

        try:
            solution = CameraSolution.open(path)
        except Exception as err:
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
        if fd.solution.solved:
            frame = fd
            break

    if frame is None or frame.solution.wcs is None:
        raise RuntimeError("Cannot find WCS reference solution for coadded frame.")

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
    coadd_sources.loc[:, ["x_ff", "z_ff"]] = ff_locs

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


def get_guider_solutions(root: pathlib.Path, telescope: str, frameno: int):
    """Collects guider solutions for a frame."""

    filename = f"lvm.{telescope}.guider_{frameno:08d}.fits"

    log_h = f"Frame {telescope}-{frameno}:"

    path = root / filename
    if not path.exists():
        try:
            path = reprocess_legacy_guider_frame(root, frameno, telescope)
        except Exception as err:
            log.warning(f"{log_h} failed retrieving guider solution: {err}")
            return None

    guider_data = fits.getheader(path, "GUIDERDATA")

    solution = GuiderSolution.open(path)
    solution.guide_mode = guider_data["GUIDMODE"]
    solution.ra_field = guider_data["RAFIELD"]
    solution.dec_field = guider_data["DECFIELD"]
    solution.pa_field = guider_data.get("PAFIELD", numpy.nan)

    return solution


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
    cofwhmst = solution.sources.loc[solution.sources.valid == 1].fwhm.dropna().std()

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
    header["FWHMMED"] = nan_or_none(fwhm_median, 3)

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
    header["WARNTRAN"] = solution.transp_warning()
    header["WARNMATC"] = not solution.matched if telescope != "spec" else False
    header["WARNFWHM"] = solution.fwhm_warning()
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

    if len(frame_data) == 0 or len(guider_data) == 0:
        raise RuntimeError("No frame or guider data to combine.")

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
    header["FWHMMED"] = nan_or_none(fwhm_median, 3)

    if telescope != "spec":
        header["PACOEFFA"] = nan_or_none(pa_coeffs[0])
        header["PACOEFFB"] = nan_or_none(pa_coeffs[1])

    header["PAMIN"] = nan_or_none(pa_min, 4)
    header["PAMAX"] = nan_or_none(pa_max, 4)
    header["PADRIFT"] = nan_or_none(pa_drift, 4)

    header["ZEROPT"] = nan_or_none(solution.zero_point, 3)

    header["SOLVED"] = solution.solved
    header["NCAMSOL"] = solution.n_cameras_solved
    header.insert("FRAME0", ("", "/*** GLOBAL FRAME PARAMETERS ***/"))

    # Pointing
    if telescope != "spec":
        header["XFFPIX"] = guider_data.x_ff_pixel.iloc[0]
        header["ZFFPIX"] = guider_data.z_ff_pixel.iloc[0]
        header["RAFIELD"] = nan_or_none(guider_data.ra_field.iloc[0], 6)
        header["DECFIELD"] = nan_or_none(guider_data.dec_field.iloc[0], 6)
        header["PAFIELD"] = nan_or_none(guider_data.pa_field.iloc[0], 4)
        header["RAMEAS"] = nan_or_none(solution.pointing[0], 6)
        header["DECMEAS"] = nan_or_none(solution.pointing[1], 6)
        header["PAMEAS"] = nan_or_none(solution.pa, 4)

    # Warnings
    header["WARNPA"] = pa_warn
    header["WARNPADR"] = drift_warn
    header["WARNTRAN"] = solution.transp_warning()
    header["WARNMATC"] = solution.wcs is None if telescope != "spec" else False
    header["WARNFWHM"] = solution.fwhm_warning()
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

    hdul_orig = fits.open(file)

    if "PROC" not in hdul_orig:
        # Just return, there's an informative error later.
        return

    log.warning(f"File {file!s} is too old and will be reprocessed.")

    if "RAW" in hdul_orig:
        raw = hdul_orig["RAW"].header
    else:
        raw = hdul_orig[0].header

    proc = hdul_orig["PROC"].header if "PROC" in hdul_orig else {}

    data_sub, _ = get_dark_subtracted_data(file)
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

        zp = estimate_zeropoint(data_sub, sources)
        sources.update(zp)

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
        new_proc["SOLVED"] = True
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

    sources.reset_index(drop=True, inplace=True)
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
    n_solved: int = 0
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

            hdul = fits.open(root / filex)
            if "PROC" in hdul and hdul["PROC"].header["WCSMODE"].lower() != "none":
                n_solved += 1

            hdul.close()

    if len(sources) == 0:
        raise RuntimeError(f"No sources found for guider frame {proc_file!s}")

    gdata_header["NCAMSOL"] = n_solved

    sources_concat = pandas.concat(sources)
    if len(sources_concat.ra.dropna()) > 5:
        wcs = wcs_from_gaia(sources_concat, xy_cols=["x_ff", "z_ff"])
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

    sources_concat.reset_index(drop=True, inplace=True)
    sources_concat.to_parquet(guider_sources_file)

    return guider_file


def watch_for_files(base_path: str = "/data/spectro"):
    """Watches and processes guider frames for new spectrograph images."""

    AG_PATH = pathlib.Path("/data/agcam")

    sjd: int = 0

    observer = PollingObserver(timeout=5)
    handler = SpecPatternEventHandler()

    observer.schedule(handler, "/data/agcam")
    observer.start()

    try:
        while True:
            time.sleep(1)

            # Check if the SJD has changed.
            new_sjd = get_sjd("LCO")

            if new_sjd != sjd:
                observer.unschedule_all()

                if not os.path.exists(f"{base_path}/{new_sjd}"):
                    continue

                log.info(f"Switching to SJD {new_sjd}")
                sjd = new_sjd

                (AG_PATH / str(sjd) / "coadds").mkdir(parents=True, exist_ok=True)
                log.start_file_logger(
                    str(AG_PATH / f"{sjd}/coadds/coadds_{sjd}.log"),
                    rotating=False,
                )

                observer.schedule(handler, f"{base_path}/{sjd}")

                log.info(f"Watching directory {base_path}/{sjd}")

    except KeyboardInterrupt:
        observer.stop()
    except Exception as err:
        print(err)

    observer.join()


class SpecPatternEventHandler(PatternMatchingEventHandler):
    """Handles newly created spectrograph files."""

    def __init__(self):
        self.lock = Lock()

        super().__init__(
            patterns=["*-b1-*.fits.gz"],
            ignore_directories=True,
            case_sensitive=True,
        )

    def on_any_event(self, event: FileCreatedEvent | FileMovedEvent):
        """Runs the co-add code when a new file is created/moved."""

        # Do not process more than one file at the same time.
        while self.lock.locked():
            time.sleep(1)

        self.lock.acquire()

        try:
            if event.event_type == "moved":
                new_file = event.dest_path  # type: ignore
            elif event.event_type == "created":
                new_file = event.src_path
            else:
                log.debug(f"Not handling event {event!r}")
                return

            if new_file is None or new_file == "":
                return

            path = pathlib.Path(new_file).absolute()

            if not path.exists():
                log.warning(f"Detected file {path!s} does not exist!")
                return

            image_type = fits.getval(str(path), "IMAGETYP")
            if image_type != "object":
                log.info(f"Detected file {path}. Not an object image, skipping.")
                return

            log.info(f"Processing spectrograph frame {get_spec_frameno(path)}")
            outpaths = coadd_from_spec_frame(path, multiprocess_mode="telescopes")

            if len(outpaths) == 0:
                log.info(f"All done for {path}")
                return

            # Recreate the summary files.
            log.info("Updating summary files.")

            sjd = get_sjd("LCO")
            parent_coadds = outpaths[0].parent

            for table in ["frames", "guiderdata"]:
                cfs = parent_coadds.glob("lvm.*.coadd*.fits")
                tfs = [pp.with_name(pp.stem + f"_{table}.parquet") for pp in cfs]
                spec_nos = [int(ff.name.split("_")[-2][1:]) for ff in tfs]

                if len(spec_nos) > 0:
                    create_summary_file(
                        tfs,
                        parent_coadds / f"lvm.guider.mjd_{sjd}_{table}.parquet",
                        spec_framenos=spec_nos,
                    )

            log.info(f"All done for {path}")

        except Exception as err:
            log.error(f"Error found while processing file: {err}")

        finally:
            # Release the lock so other files can be processed.
            self.lock.release()


def coadd_to_database(
    coadd_file: AnyPath,
    exposure_no: int | None = None,
    **db_connection_params,
):
    """Loads the co-add data into the databse."""

    coadd_file = pathlib.Path(coadd_file)

    log.info("Ingesting co-added files into the database.")

    if not coadd_file.exists():
        raise RuntimeError(f"Cannot find co-add file {coadd_file}")

    global_h = fits.getheader(coadd_file, "GLOBAL")
    sjd = global_h["MJD"]

    delete_columns: list[str] = ["telescope"]
    if exposure_no is not None:
        delete_columns.append("exposure_no")

    # Load global coadd data.
    log.debug("Loading coadd data to database.")
    cards = [
        "mjd",
        "frame0",
        "framen",
        "nframes",
        "obstime0",
        "obstimen",
        "fwhm0",
        "fwhmn",
        "fwhmmed",
        "pacoeffa",
        "pacoeffb",
        "pamin",
        "pamax",
        "padrift",
        "zeropt",
        "solved",
        "ncamsol",
        "xffpix",
        "zffpix",
        "rafield",
        "decfield",
        "pafield",
        "rameas",
        "decmeas",
        "pameas",
        "warnpa",
        "warnpadr",
        "warntran",
        "warnmatc",
        "warnfwhm",
    ]
    try:
        global_df = pandas.DataFrame()
        for card in cards:
            global_df[card] = [global_h[card]]
        global_df["telescope"] = global_h["TELESCOP"]
        if exposure_no:
            global_df["exposure_no"] = exposure_no

        table_name = config["database"]["guider_coadd_table"]
        dataframe_to_database(
            global_df,
            table_name,
            delete_columns=delete_columns,
            **db_connection_params,
        )
    except Exception as err:
        log.error(f"Failed loading frames to database: {str(err).strip()}")

    # Load AG frames.
    ag_frames_file = coadd_file.with_name(coadd_file.stem + "_frames.parquet")
    if ag_frames_file.exists():
        log.debug("Loading AG frames to database.")
        try:
            ag_frames = pandas.read_parquet(ag_frames_file)
            if exposure_no:
                ag_frames["exposure_no"] = exposure_no
            if len(ag_frames) > 0:
                ag_frames_mjd = isot_to_sjd(ag_frames["date_obs"].iloc[0])
                ag_frames["mjd"] = ag_frames_mjd
                ag_frames["stacked"] = ag_frames.stacked.astype(bool)
                ag_frames["solved"] = ag_frames.solved.astype(bool)
                table_name = config["database"]["agcam_frame_table"]
                dataframe_to_database(
                    ag_frames,
                    table_name,
                    delete_columns=delete_columns,
                    **db_connection_params,
                )
        except Exception as err:
            log.error(f"Failed loading frames to database: {str(err).strip()}")
    else:
        log.error(f"Cannot find AG frames file {ag_frames_file}")

    # Load guider data.
    guider_frames_file = coadd_file.with_name(coadd_file.stem + "_guiderdata.parquet")
    if guider_frames_file.exists():
        log.debug("Loading guider frames to database.")
        try:
            guider_frames = pandas.read_parquet(guider_frames_file)
            if exposure_no:
                guider_frames["exposure_no"] = exposure_no
            if len(guider_frames) > 0:
                guider_frames["mjd"] = sjd
                guider_frames["solved"] = guider_frames.solved.astype(bool)
                guider_frames["applied"] = guider_frames.applied.astype(bool)
                table_name = config["database"]["guider_frame_table"]
                dataframe_to_database(
                    guider_frames,
                    table_name,
                    delete_columns=delete_columns,
                    **db_connection_params,
                )
        except Exception as err:
            log.error(f"Failed loading frames to database: {str(err).strip()}")
    else:
        log.error(f"Cannot find guider frames file {guider_frames_file}")
