#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-03-13
# @Filename: tools.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import concurrent.futures
import os.path
import pathlib
import re
import warnings
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from time import time

from typing import TYPE_CHECKING, Any, Sequence

import nptyping
import numpy
import pandas
import peewee
import pgpasslib
import sep
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.time import Time, TimezoneInfo
from psycopg2 import OperationalError

from sdsstools import read_yaml_file
from sdsstools.time import get_sjd

from lvmguider import config, log
from lvmguider.types import ARRAY_1D_F32, ARRAY_2D_F32


if TYPE_CHECKING:
    from astropy.wcs import WCS

    from lvmguider.actor import GuiderCommand


async def run_in_executor(fn, *args, catch_warnings=False, executor="thread", **kwargs):
    """Runs a function in an executor.

    In addition to streamlining the use of the executor, this function
    catches any warning issued during the execution and reissues them
    after the executor is done. This is important when using the
    actor log handler since inside the executor there is no loop that
    CLU can use to output the warnings.

    In general, note that the function must not try to do anything with
    the actor since they run on different loops.

    """

    fn = partial(fn, *args, **kwargs)

    if executor == "thread":
        executor = concurrent.futures.ThreadPoolExecutor
    elif executor == "process":
        executor = concurrent.futures.ProcessPoolExecutor
    else:
        raise ValueError("Invalid executor name.")

    if catch_warnings:
        with warnings.catch_warnings(record=True) as records:
            with executor() as pool:
                result = await asyncio.get_event_loop().run_in_executor(pool, fn)

        for ww in records:
            warnings.warn(ww.message, ww.category)

    else:
        with executor() as pool:
            result = await asyncio.get_running_loop().run_in_executor(pool, fn)

    return result


def create_summary_table(
    path: str | pathlib.Path = ".",
    pattern: str = "lvm.*.agcam*.fits*",
) -> Table:
    """Returns a table with a summary of agcam information for the files in a directory.

    Parameters
    ----------
    path
        The path in which to search for file.
    pattern
        A pattern to select files. Defaults to selecting all raw frames.

    Returns
    -------
    table
        A table of frame data.

    """

    path = pathlib.Path(path)
    files = path.glob(pattern)

    columns = [
        "frameno",
        "telescope",
        "camera",
        "filename",
        "date-obs",
        "exptime",
        "imagetype",
        "ra",
        "dec",
        "nsources",
    ]
    data = []
    for file_ in files:
        file_match = re.match(r"lvm\.([a-z]+)\..+?\_([0-9]+)\.fits", file_.name)

        telescope = file_match.group(1) if file_match else ""
        frameno = int(file_match.group(2)) if file_match else -1

        hdus = fits.open(file_)
        if "SOURCES" in hdus:
            nsources = len(hdus["SOURCES"].data)
        else:
            nsources = -1

        header = hdus[0].header

        data.append(
            (
                frameno,
                telescope,
                header["CAMNAME"],
                file_.name,
                header["DATE-OBS"],
                header["EXPTIME"],
                header["IMAGETYP"],
                header["RA"],
                header["DEC"],
                nsources,
            )
        )

    tt = Table(rows=data, names=columns)
    tt.sort(["frameno", "telescope", "camera"])

    return tt


def get_proc_path(filename: str | pathlib.Path):
    """Returns the ``proc-`` path for a given file (DEPRECATED)."""

    path = pathlib.Path(filename).absolute()
    dirname = path.parent
    basename = path.name.replace(".east", "").replace(".west", "")
    proc_path = dirname / ("proc-" + basename)

    return proc_path


def get_guider_path(filename: str | pathlib.Path):
    """Returns the ``lvm.guider`` path for a given ``lvm.agcam`` file."""

    filename = pathlib.Path(filename).absolute()
    frameno = get_frameno(filename)

    match = re.search("(sci|spec|skye|skyw)", str(filename))
    if match is None:
        raise ValueError(f"Invalid file path {filename!s}")

    telescope = match.group()
    basename = f"lvm.{telescope}.guider_{frameno:08d}.fits"

    return filename.parent / basename


@contextmanager
def elapsed_time(command: GuiderCommand, task_name: str = "unnamed"):
    """Context manager to output the elapsed time for a task."""

    t0 = time()

    yield

    command.actor.log.debug(f"Elapsed time for task {task_name!r}: {time()-t0:.3f} s")


def get_frameno(file: os.PathLike | str) -> int:
    """Returns the frame number for a frame."""

    match = re.match(r"^.+?([0-9]+)\.fits$", str(file))

    if match is None:
        raise ValueError("Invalid file format. Cannot determine frame number.")

    return int(match.group(1))


def get_db_connection(
    profile: str = "default",
    dbname: str | None = None,
    host: str | None = None,
    port: int | None = None,
    user: str | None = None,
    password: str | None = None,
    pgpass_path: str | None = None,
):
    """Returns a connection to the LVM database.

    Any parameters not provided will default to the configuration values
    from the selected profile.

    Returns
    -------
    connection
        A ``peewee`` ``PostgresqlDatabase`` object. The connection is returned
        not connected and it's the user's responsibility to connect and check
        for errors.

    """

    PARAMS = ["host", "port", "user", "dbname", "password", "pgpass_path"]

    default_params = config.get("database", {}).get(profile, {}).copy()
    default_params = {kk: vv for kk, vv in default_params.items() if kk in PARAMS}

    call_params = dict(
        dbname=dbname,
        host=host,
        port=port,
        user=user,
        password=password,
        pgpass_path=pgpass_path,
    )
    call_params = {kk: vv for kk, vv in call_params.items() if vv is not None}

    db_params = default_params
    db_params.update(call_params)

    pgpass_path = db_params.pop("pgpass_path", None)
    password = db_params.pop("password", None)
    if password is None and pgpass_path is not None:
        password = pgpasslib.getpass(**db_params)

    log.debug(f"Connecting to database with params: {db_params!r}")

    dbname = db_params.pop("dbname")
    conn = peewee.PostgresqlDatabase(database=dbname, password=password, **db_params)

    return conn


def get_model(key: str):
    """Returns a key from the guider datamodel."""

    datamodel = read_yaml_file(pathlib.Path(__file__).parent / "etc/datamodel.yml")

    return datamodel[key]


def dataframe_from_model(model_name: str):
    """Reads a data model and returns an empty data frame with the right types."""

    model = get_model(model_name)
    return pandas.DataFrame({k: pandas.Series(dtype=v) for k, v in model.items()})


def header_from_model(model_name: str):
    """Reads a data model and returns a ``Header`` object."""

    model = get_model(model_name)
    return fits.Header([(k, *v) for k, v in model.items()])


def update_fits(
    file: str | pathlib.Path,
    ext: int | str,
    data: nptyping.NDArray | None = None,
    header: fits.Header | dict[str, Any | tuple[Any, str]] | None = None,
):
    """Updates or creates an image HDU in an existing FITS.

    Parameters
    ----------
    file
        The FITS file to update.
    ext
        The extension to update or create.
    data
        A Numpy array with new data. If defined and the extension does not
        exist, the new extension will be a compressed image HDU.
    header
        The header data to update the extension.


    """

    file = pathlib.Path(file)
    if not file.exists():
        raise FileNotFoundError(f"File {file!s} was not found.")

    with fits.open(str(file), mode="update") as hdul:
        name = ext if isinstance(ext, str) else None

        try:
            hdu = hdul[ext]
        except (IndexError, KeyError):
            if data is not None:
                hdu = fits.CompImageHDU(data=data, header=header, name=name)
            else:
                hdu = fits.ImageHDU(data=data, header=header, name=name)

            hdul.append(hdu)
            hdul.close()
            return

        hdu.data = data

        if isinstance(header, fits.Header):
            hdu.header.update(header)
        elif header is not None:
            for key, value in header.items():
                hdu.header[key] = value


def get_gaia_sources(
    wcs: WCS,
    db_connection_params: dict[str, Any] = {},
    include_lvm_mags: bool = True,
):
    """Returns a data frame with Gaia source information from a WCS.

    Parameters
    ----------
    wcs
        A WCS associated with the image. Used to determine the centre of the
        image and perform a radial query in the database.
    include_lvm_mags
        If `True`, match to ``lvm_magnitude`` and return LVM AG passband
        magnitudes and fluxes.
    db_connection_params
        A dictionary of DB connection parameters to pass to `.get_db_connection`.

    """

    XZ_AG_FRAME = config["xz_ag_frame"]

    # A bit larger than reality to account for WCS imprecision.
    CAM_FOV = max(XZ_AG_FRAME) / 3600 * 1.2

    conn = get_db_connection(**db_connection_params)

    try:
        conn.connect()
    except OperationalError as err:
        raise RuntimeError(f"Cannot connect to database: {err}")

    # Get RA/Dec of centre of frame.
    skyc = wcs.pixel_to_world(*XZ_AG_FRAME)
    ra: float = skyc.ra.deg
    dec: float = skyc.dec.deg

    # Query lvm_magnitude and gaia_dr3_source in a radial query around RA/Dec.

    gdr3_sch, gdr3_table = config["database"]["gaia_dr3_source_table"].split(".")
    lmag_sch, lmag_table = config["database"]["lvm_magnitude_table"].split(".")
    GDR3 = peewee.Table(gdr3_table, schema=gdr3_sch).bind(conn)
    LMAG = peewee.Table(lmag_table, schema=lmag_sch).bind(conn)

    if include_lvm_mags:
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

    else:
        query = (
            GDR3.select(
                GDR3.c.ra,
                GDR3.c.dec,
                GDR3.c.pmra,
                GDR3.c.pmdec,
                GDR3.c.phot_g_mean_mag,
            )
            .where(peewee.fn.q3c_radial_query(GDR3.c.ra, GDR3.c.dec, ra, dec, CAM_FOV))
            .dicts()
        )

    with conn.atomic():
        conn.execute_sql("SET LOCAL work_mem='2GB'")
        conn.execute_sql("SET LOCAL enable_seqscan=false")
        data = query.execute(conn)

    df = pandas.DataFrame.from_records(data)

    # Cast to correct types
    for column in ["pmra", "pmdec", "phot_g_mean_mag", "lmag_ab", "lflux"]:
        df[column] = df[column].astype(numpy.float32)

    return df


def estimate_zeropoint(
    image: ARRAY_2D_F32,
    sources: pandas.DataFrame,
    gain: float = 5,
    ap_radius: float = 6,
    ap_bkgann: tuple[float, float] = (6, 10),
):
    """Determines the ``lmag`` zeropoint for each source.

    Parameters
    ----------
    image
        The image on which to perform aperture photometry.
    sources
        A data frame with extracted sources. It must have been matched with
        Gaia and include an ``lmag_ab`` column with the AB magnitude of the
        sources in the LVM AG bandpass.
    gain
        The gain of the detector in e-/ADU.
    ap_radius
        The radius to use for aperture photometry, in pixels.
    ap_bkgann
        The inner and outer radii of the annulus used to determine the background
        around each source.

    Returns
    -------
    zp_data
        A Pandas data frame with the same length as the input ``sources`` with
        columns from the aperture photometry results and the LVM zero point.

    """

    # Do aperture photometry around the detections.
    flux_adu, fluxerr_adu, _ = sep.sum_circle(
        image,
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
    valid = flux_adu > 0
    zp = -2.5 * numpy.log10(flux_adu[valid] * gain) - sources.lmag_ab[valid]

    df = pandas.DataFrame(columns=["ap_flux", "ap_fluxerr", "zp"], dtype=numpy.float32)

    # There is a weirdness in Pandas that if flux_adu is a single value then the
    # .loc[:, "ap_flux"] would not set any values. So we replace the entire column
    # but endure the dtypes.
    df["ap_flux"] = flux_adu.astype(numpy.float32)
    df["ap_fluxerr"] = fluxerr_adu.astype(numpy.float32)
    df.loc[valid, "zp"] = zp.astype(numpy.float32)

    df.index = sources.index

    return df


def get_dark_subtracted_data(file: pathlib.Path | str) -> tuple[ARRAY_2D_F32, bool]:
    """Returns a background or dark subtracted image."""

    path = pathlib.Path(file)

    hdul = fits.open(str(path))

    exptime = hdul["RAW"].header["EXPTIME"]

    # Data in counts per second.
    data: ARRAY_2D_F32 = hdul["RAW"].data.copy().astype("f4") / exptime

    if "PROC" in hdul:
        # Get data and subtract dark or fit background.
        dirname = hdul["PROC"].header.get("DIRNAME", path.parent)
        dark_file = hdul["PROC"].header.get("DARKFILE", "")

        dark_path = pathlib.Path(dirname) / dark_file

    else:
        dark_file = ""
        dark_path = None

    if dark_file != "" and dark_path is not None and dark_path.exists():
        dark: ARRAY_2D_F32 = fits.getdata(str(dark_path), "RAW").astype("f4")
        dark_exptime: float = fits.getval(str(dark_path), "EXPTIME", "RAW")
        data = data - dark / dark_exptime

        dark_sub = True

    else:
        data = data - sep.Background(data).back()
        dark_sub = False

    return data, dark_sub


def get_files_in_time_range(
    path: pathlib.Path,
    time0: Time,
    time1: Time,
    pattern: str = "*",
):
    """Returns all files in a directory with modification time in the time range.

    Parameters
    ----------
    path
        The path on which to search for files.
    tile0,time1
        The range of times, as astropy ``Time`` objects.
    pattern
        A pattern to use to subset the files in the directory. Defaults to
        search all files.

    Returns
    -------
    files
        A list of paths to files in the directory that where last modified in the
        selected time range.

    """

    files = path.glob(pattern)

    tz = TimezoneInfo()
    dt0 = time0.to_datetime(tz).timestamp()
    dt1 = time1.to_datetime(tz).timestamp()

    return [
        file
        for file in files
        if os.path.getmtime(file) > dt0 and os.path.getmtime(file) < dt1
    ]


def get_agcam_in_time_range(
    path: pathlib.Path,
    time0: Time,
    time1: Time,
    pattern: str = "*.fits",
):
    """Similar to `.get_files_in_time_range` but uses the keyword ``DATE-OBS``."""

    files = path.glob(pattern)

    matched: list[pathlib.Path] = []
    for file in files:
        try:
            hdul = fits.open(file)
        except Exception:
            continue

        if "RAW" in hdul:
            header = hdul["RAW"].header
        else:
            header = hdul[0].header

        if "DATE-OBS" not in header:
            continue

        time = Time(header["DATE-OBS"], format="isot")
        # print(time, time0, time1, time < time1, time > time0)
        if time > time0 and time < time1:
            matched.append(file)

    return matched


def get_guider_files_from_spec(
    spec_file: str | pathlib.Path,
    telescope: str | None = None,
    agcam_path: str | pathlib.Path = "/data/agcam",
    camera: str | None = None,
    use_time_range: bool | None = None,
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
    use_time_range
        By default (``time_time_range=None``) the function will check the
        ``G{telescope}FR0`` and ``G{telescope}FRN`` keywords to determine the
        range of guider frames. If those keywords are not present, the AG
        files that were last modified in the range of the integration will
        be found. With `True`, the last modification range method will always
        be used.

    Returns
    -------
    frames
        A list of AG file names that match the range of guide frames
        taken during the spectrograph exposure. Missing frames in the
        range are not included.

    """

    spec_file = pathlib.Path(spec_file)
    if not spec_file.exists():
        raise FileExistsError(f"Cannot find file {spec_file!s}")

    header = fits.getheader(str(spec_file))

    if telescope is None:
        telescope = config["telescope"]

    assert isinstance(telescope, str)

    sjd = get_sjd("LCO", date=Time(header["OBSTIME"], format="isot").to_datetime())

    agcam_path = pathlib.Path(agcam_path) / str(sjd)
    if not agcam_path.exists():
        raise FileExistsError(f"Cannot find agcam path {agcam_path!s}")

    frame0 = header.get(f"G{telescope.upper()}FR0", None)
    frame1 = header.get(f"G{telescope.upper()}FRN", None)

    if use_time_range is True or (frame0 is None or frame1 is None):
        log.warning(f"Matching guider frames by date for file {spec_file.name}")

        time0 = Time(header["INTSTART"])
        time1 = Time(header["INTEND"])
        files = get_agcam_in_time_range(
            agcam_path,
            time0,
            time1,
            pattern=f"lvm.{telescope}.*.fits",
        )

        if len(files) == 0:
            raise ValueError(f"Cannot find guider frames for {spec_file!s}")

        return sorted(files)

    return get_frame_range(agcam_path, telescope, frame0, frame1, camera=camera)


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
        selects all cameras.

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


def polyfit_with_sigclip(
    x: ARRAY_1D_F32,
    y: ARRAY_1D_F32,
    sigma: int = 3,
    deg: int = 1,
):
    """Fits a polynomial to data after sigma-clipping the dependent variable."""

    valid = ~sigma_clip(y, sigma=sigma, masked=True).mask  # type:ignore

    return numpy.polyfit(x[valid], y[valid], deg)


def nan_or_none(value: float | None, precision: int | None = None):
    """Replaces ``NaN`` with `None`. If the value is not ``NaN``, rounds up."""

    if value is None or numpy.isnan(value):
        return None

    if precision is not None:
        return numpy.round(value, precision)

    return value


def sort_files_by_camera(files: Sequence[str | os.PathLike]):
    """Returns a dictionary of camera name to list of files."""

    camera_to_files: defaultdict[str, list[pathlib.Path]] = defaultdict(list)
    for file in files:
        for camera in ["east", "west"]:
            if camera in str(file):
                camera_to_files[camera].append(pathlib.Path(file))
                break

    return camera_to_files


def angle_difference(angle1: float, angle2: float):
    """Calculates the shorted difference between two angles."""

    diff = (angle2 - angle1 + 180) % 360 - 180
    return diff + 360 if diff < -180 else diff
