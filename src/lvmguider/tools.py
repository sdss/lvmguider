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
import pathlib
import re
import warnings
from contextlib import contextmanager
from functools import partial
from time import time

from typing import TYPE_CHECKING

import numpy
import pandas
import peewee
import pgpasslib
from astropy.io import fits
from astropy.table import Table

from lvmguider import config, log


if TYPE_CHECKING:
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
    """Returns the proc- path for a given file."""

    path = pathlib.Path(filename).absolute()
    dirname = path.parent
    basename = path.name.replace(".east", "").replace(".west", "")
    proc_path = dirname / ("proc-" + basename)

    return proc_path


async def append_extension(
    filename: str | pathlib.Path,
    data: numpy.ndarray | None = None,
    table: pandas.DataFrame | numpy.recarray | Table | None = None,
    header: tuple | list = (),
    compress: str | None = None,
    name: str | None = None,
):
    """Updates a FITS file with a new extension.

    Parameters
    ----------
    filename
        The path to the file to update.
    data
        A numpy array with the image data to add.
    table
        A table to add as a binary table extension
    header
        The header to add to the extension, as a list of tuples,
        each one of them including the keyword name, value, and
        optionally a comment.
    compress
        If ``data`` is passed, whether to use a compressed image
        extension. Must be the compression algorithm to use
        (e.g., ``RICE_1``).
    name
        The name to give to the extension.

    """

    if data is not None and table is not None:
        raise ValueError("data and table are mutually exclusive.")

    with fits.open(filename, mode="update") as hdul:
        if data is not None:
            if compress is None:
                ext = fits.ImageHDU(data=data, header=header, name=name)
            else:
                ext = fits.CompImageHDU(
                    data=data,
                    header=header,
                    compression_type=compress,
                    name=name,
                )
        else:
            if isinstance(table, pandas.DataFrame):
                table_data = Table.from_pandas(table)
            else:
                table_data = table

            ext = fits.BinTableHDU(data=table_data, name=name)

        hdul.append(ext)


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
