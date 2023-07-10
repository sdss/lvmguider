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
import shutil
import warnings
from functools import partial

import pandas
from astropy.io import fits
from astropy.table import Table


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


def reprocess_proc_image(
    filename: str | pathlib.Path,
    telescope: str,
    output_path: str | pathlib.Path,
    keep_previous_wcs: bool = True,
    solve_individual_cameras: bool = True,
    generate_astrometrynet_outputs: bool = False,
    index_paths: dict[int, str] | None = None,
    scales: dict[int, list[int]] | None = None,
):
    """Reprocesses a proc- file."""

    from lvmguider.astrometrynet import astrometrynet_quick
    from lvmguider.transformations import ag_to_master_frame, solve_locs

    proc_orig = pathlib.Path(filename).absolute()
    output_path = pathlib.Path(output_path).absolute()

    hdus = fits.open(str(proc_orig))
    header = hdus[1].header

    ra = header["RAFIELD"]
    dec = header["DECFIELD"]

    if "SOURCES" not in hdus:
        warnings.warn("No SOURCES found.", UserWarning)
        return None

    sources = pandas.DataFrame(hdus["SOURCES"].data)
    sources.loc[:, "x_master"] = 0.0
    sources.loc[:, "y_master"] = 0.0

    output_path.mkdir(parents=True, exist_ok=True)
    proc_new = output_path / proc_orig.name

    shutil.copyfile(proc_orig, proc_new)

    proc_new = output_path / proc_orig.name
    new_sources_cam = []
    for camname in ["east", "west"]:
        sources_cam = sources.loc[sources.camera == camname]
        xy = sources_cam[["x", "y"]].values

        camera = f"{telescope}-{camname[0]}"
        file_locs, _ = ag_to_master_frame(camera, xy)
        sources_cam.loc[:, ["x_master", "y_master"]] = file_locs
        new_sources_cam.append(sources_cam)

    new_sources = pandas.concat(new_sources_cam)
    output_root = str(proc_new).replace(".fits.gz", "").replace(".fits", "")

    xyls = new_sources.loc[:, ["x_master", "y_master", "flux"]]
    xyls.rename(columns={"x_master": "x", "y_master": "y"}, inplace=True)

    try:
        solution = solve_locs(
            xyls,
            ra=ra,
            dec=dec,
            full_frame=True,
            index_paths=index_paths,
            scales=scales,
            output_root=output_root if generate_astrometrynet_outputs else None,
        )
        new_wcs = solution.wcs
    except RuntimeError:
        print(f"[WARNING]: Failed solving {filename}.")
        solution = None
        new_wcs = None

    with fits.open(str(proc_new), "update") as hdul:
        del hdul["SOURCES"]
        new_sources_t = Table.from_pandas(new_sources)
        hdul.append(fits.BinTableHDU(data=new_sources_t, name="SOURCES"))

        if new_wcs is None and keep_previous_wcs:
            hdul.append(fits.ImageHDU(name="REPROC"))
        elif new_wcs:
            if keep_previous_wcs:
                new_hdu = fits.ImageHDU(name="REPROC")
                new_hdu.header += new_wcs.to_header(relax=True)
                hdul.append(new_hdu)
            else:
                orig_hdu = hdul[1]
                orig_hdu.header += new_wcs.to_header(relax=True)

    if solve_individual_cameras:
        for camname in ["east", "west"]:
            sources_cam = new_sources.loc[new_sources.camera == camname]

            solution = astrometrynet_quick(
                {5200: "/data/astrometrynet/5200"},
                sources_cam,
                ra=ra,
                dec=dec,
                radius=5.0,
                pixel_scale=1.0,
                pixel_scale_factor_hi=1.2,
                pixel_scale_factor_lo=0.8,
                scales={5200: [5, 6]},
                plot=False,
                output_root=(output_root + f"_{camname}") if output_root else None,
            )

            with fits.open(str(proc_new), "update") as hdul:
                new_hdu = fits.ImageHDU(name=camname.upper())
                if solution.wcs:
                    new_hdu.header += solution.wcs.to_header(relax=True)
                    new_hdu.header["SOLVED"] = True
                else:
                    print(f"[WARNING]: Failed solving {filename} ({camname}).")
                    new_hdu.header["SOLVED"] = False
                hdul.append(new_hdu)

    return solution


def reprocess_files(
    filenames: list[str],
    telescope: str,
    proc_output_path: str | pathlib.Path,
    solve_individual_cameras: bool = True,
    generate_astrometrynet_outputs: bool = False,
    index_paths: dict[int, str] | None = None,
    scales: dict[int, list[int]] | None = None,
    **kwargs,
):
    """Reprocesses raw files.

    This basically calls `.solve_from_files` but writes the proc- file
    to a new location and allows to specify indexes and scales.

    """

    from lvmguider.astrometrynet import astrometrynet_quick
    from lvmguider.transformations import solve_from_files

    proc_output_path = pathlib.Path(proc_output_path)
    parent = proc_output_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    output_root = (
        str(proc_output_path).replace(".fits.gz", "").replace(".fits", "")
        if generate_astrometrynet_outputs
        else None
    )

    solution, new_sources = solve_from_files(
        filenames,
        telescope,
        reextract_sources=True,
        solve_locs_kwargs={
            "index_paths": index_paths,
            "output_root": output_root,
            "scales": scales,
            **kwargs,
        },
    )
    assert solution.wcs

    new_sources_t = Table.from_pandas(new_sources)
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(header=solution.wcs.to_header(relax=True)),
            fits.BinTableHDU(data=new_sources_t, name="SOURCES"),
        ]
    )

    if solve_individual_cameras:
        header = fits.getheader(filenames[0])
        ra = header["RA"]
        dec = header["DEC"]

        for camname in ["east", "west"]:
            sources_cam = new_sources.loc[new_sources.camera == camname]

            cam_solution = astrometrynet_quick(
                {5200: "/data/astrometrynet/5200"},
                sources_cam,
                ra=ra,
                dec=dec,
                radius=5.0,
                pixel_scale=1.0,
                pixel_scale_factor_hi=1.2,
                pixel_scale_factor_lo=0.8,
                scales={5200: [5, 6]},
                plot=False,
                raise_on_unsolved=True,
                output_root=(output_root + f"_{camname}") if output_root else None,
                **kwargs,
            )

            new_hdu = fits.ImageHDU(name=camname.upper())
            if cam_solution.wcs:
                new_hdu.header += cam_solution.wcs.to_header(relax=True)
                new_hdu.header["SOLVED"] = True
            else:
                print(f"[WARNING]: Failed solving {camname}.")
                new_hdu.header["SOLVED"] = False
            hdul.append(new_hdu)

    hdul.writeto(str(proc_output_path), overwrite=True)

    return solution
