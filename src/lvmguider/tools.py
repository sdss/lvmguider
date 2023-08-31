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
from contextlib import contextmanager
from functools import partial
from time import time

from typing import TYPE_CHECKING, Any

import numpy
import pandas
import peewee
import pgpasslib
import sep
from astropy.io import fits
from astropy.table import Table
from psycopg2 import OperationalError

from sdsstools import read_yaml_file

from lvmguider import config, log
from lvmguider.types import ARRAY_2D_F32


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


def get_frameno(file: pathlib.Path | str) -> int:
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


def update_fits(
    file: str | pathlib.Path,
    extension_data: dict[str | int, dict[str, Any] | pandas.DataFrame | None],
):
    """Updates an existing FITS file by extending a header or table or adding a new HDU.

    Parameters
    ----------
    file
        The FITS file to update.
    extension_data
        A dictionary of extension name to data to update. If the extension does not
        exists a new one will be created. The data can be a Pandas data frame,
        which will result in a new binary table, or a dictionary to update
        an image extension. The dictionary may include a ``__data__`` value
        with the image data. Image extensions are created as ``RICE_1``-compressed
        extensions.

    """

    file = pathlib.Path(file)
    if not file.exists():
        raise FileNotFoundError(f"File {file!s} was not found.")

    with fits.open(str(file), mode="update") as hdul:
        for key, data in extension_data.copy().items():
            if data is None:
                continue

            try:
                hdu = hdul[key]
            except (IndexError, KeyError):
                hdu = None

            extension_name = name = key if isinstance(key, str) else None

            if isinstance(data, pandas.DataFrame):
                bintable = fits.BinaryTableHDU(data=Table.from_pandas(data), name=name)
                if hdu is not None:
                    hdul[key] = bintable
                else:
                    hdul.append(bintable)

            else:
                image_data = extension_data.pop("__data__", None)
                if hdu:
                    if image_data is not None:
                        hdu.data = image_data
                    hdu.header.update(image_data)
                else:
                    hdul.append(
                        fits.CompImageHDU(
                            data=image_data,
                            header=data,
                            name=extension_name,
                        )
                    )


def get_gaia_sources(
    wcs: WCS,
    db_connection_params: dict[str, Any] = {},
    include_lvm_mags: bool = True,
    use_cache: bool = True,
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
    use_cache
        If `True` and the query centre is within 5 arcsec of the previous query,
        returns the same results without querying the database.

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
    zp = -2.5 * numpy.log10(flux_adu * gain) - sources.lmag_ab

    df = pandas.DataFrame()
    df["ap_flux"] = flux_adu
    df["ap_fluxerr"] = fluxerr_adu
    df["zp"] = zp

    df.index = sources.index

    return df


def get_dark_subtrcted_data(file: pathlib.Path | str) -> tuple[ARRAY_2D_F32, bool]:
    """Returns a background or dark subtracted image."""

    hdul = fits.open(str(file))

    exptime = hdul["RAW"].header["EXPTIME"]

    # Data in counts per second.
    data: ARRAY_2D_F32 = hdul["RAW"].data.copy().astype("f4") / exptime

    # Get data and subtract dark or fit background.
    dark_file = hdul["PROC"].header["DARKFILE"]
    if dark_file != "" and os.path.exists(dark_file):
        dark: ARRAY_2D_F32 = fits.getdata(dark_file, "RAW").astype("f4")
        dark_exptime: float = fits.getval(dark_file, "EXPTIME", "RAW")
        data = data - dark / dark_exptime

        dark_sub = True

    else:
        data = data - sep.Background(data).back()
        dark_sub = False

    return data, dark_sub
