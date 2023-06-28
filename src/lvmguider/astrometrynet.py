#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-18
# @Filename: astrometrynet.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import os
import pathlib
import re
import subprocess
import tempfile
import time
import warnings
from glob import glob

from typing import NamedTuple, Optional, TypeVar

import numpy
import pandas
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS


PathLike = TypeVar("PathLike", pathlib.Path, str)


class TimedProcess(NamedTuple):
    """A completed process which includes its elapsed time."""

    process: asyncio.subprocess.Process | subprocess.CompletedProcess
    elapsed: float


class AstrometryNet:
    """A wrapper for the astrometry.net ``solve-field`` command.

    Parameters
    ----------
    configure_params
        Parameters to be passed to `.configure`.
    """

    def __init__(self, **configure_params):
        solve_field_cmd = subprocess.run(
            "which solve-field",
            shell=True,
            capture_output=True,
        )
        solve_field_cmd.check_returncode()

        self.solve_field_cmd = solve_field_cmd.stdout.decode().strip()

        self._options = {}
        self.configure(**configure_params)

    def configure(
        self,
        backend_config: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        sort_column: Optional[str] = None,
        sort_ascending: Optional[bool] = None,
        no_plots: Optional[bool] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius: Optional[float] = None,
        scale_low: Optional[float] = None,
        scale_high: Optional[float] = None,
        scale_units: Optional[str] = None,
        dir: Optional[str] = None,
        **kwargs,
    ):
        """Configures how to run of ``solve-field```.

        The parameters this method accepts are identical to those of
        ``solve-field`` and are passed unchanged.

        Parameters
        ----------
        backend_config
            Use this config file for the ``astrometry-engine`` program.
        width
            Specify the field width, in pixels.
        height
            Specify the field height, in pixels.
        sort_column
            The FITS column that should be used to sort the sources.
        sort_ascending
            Sort in ascending order (smallest first);
            default is descending order.
        no_plot
            Do not produce plots.
        ra
            RA of field center for search, in degrees.
        dec
            Dec of field center for search, in degrees.
        radius
            Only search in indexes within ``radius`` degrees of the field
            center given by ``ra`` and ``dec``.
        scale_low
            Lower bound of image scale estimate.
        scale_high
            Upper bound of image scale estimate.
        scale_units
            In what units are the lower and upper bounds? Choices:
            ``'degwidth'``, ``'arcminwidth'``, ``'arcsecperpix'``,
            ``'focalmm'``.
        dir
            Path to the directory where all output files will be saved.
        """

        self._options = {
            "backend-config": backend_config,
            "width": width,
            "height": height,
            "sort-column": sort_column,
            "sort-ascending": sort_ascending,
            "no-plots": no_plots,
            "ra": ra,
            "dec": dec,
            "radius": radius,
            "scale-low": scale_low,
            "scale-high": scale_high,
            "scale-units": scale_units,
            "dir": dir,
            "overwrite": True,
        }
        self._options.update(self._format_options(kwargs))

        return

    def _format_options(self, options: dict):
        """Formats the option names."""

        formatted_options = {}

        for key, value in options.items():
            formatted_key = key.replace("_", "-")
            formatted_options[formatted_key] = value

        return formatted_options

    def _build_command(self, files, options=None):
        """Builds the ``solve-field`` command to run."""

        if options is None:
            options = self._options

        flags = ["no-plots", "sort-ascending", "overwrite"]

        cmd = [self.solve_field_cmd]

        for option in options:
            if options[option] is None:
                continue
            if option in flags:
                if options[option] is True:
                    cmd.append("--" + option)
            else:
                cmd.append("--" + option)
                cmd.append(str(options[option]))

        cmd += list(files)

        return cmd

    def run_sync(
        self,
        files: PathLike | list[PathLike],
        shell: bool = True,
        stdout: Optional[PathLike] = None,
        stderr: Optional[PathLike] = None,
        **kwargs,
    ):
        """Runs astrometry.net synchronously.

        Parameters
        ----------
        files
            List of files to be processed.
        shell
            Whether to call `subprocess.run` with ``shell=True``.
        stdout
            Path where to save the stdout output.
        stderr
            Path where to save the stderr output.
        kwargs
            Configuration parameters (see `.configure`) to override. The
            configuration applies only to this run of ``solve-field`` and it
            is not saved.

        Returns
        -------
        `subprocess.CompletedProcess`
            The completed process.

        """

        options = self._options.copy()
        options.update(self._format_options(kwargs))

        if not isinstance(files, (tuple, list)):
            files = [files]

        t0 = time.time()

        args = self._build_command(list(map(str, files)), options=options)
        if shell:
            args = " ".join(args)

        cmd = subprocess.run(
            args,
            shell=shell,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        elapsed = time.time() - t0

        if stdout:
            with open(stdout, "wb") as out:
                if isinstance(args, str):
                    out.write(args.encode() + b"\n")
                else:
                    out.write(" ".join(args).encode() + b"\n")
                out.write(cmd.stdout)

        if stderr:
            with open(stderr, "wb") as err:
                err.write(cmd.stderr)

        return TimedProcess(cmd, elapsed)

    async def run_async(
        self,
        files: PathLike | list[PathLike],
        shell: bool = True,
        stdout: Optional[PathLike] = None,
        stderr: Optional[PathLike] = None,
        **kwargs,
    ) -> TimedProcess:
        """Runs astrometry.net asynchronously.

        Parameters
        ----------
        files
            List of files to be processed.
        shell
            Whether to call `subprocess.run` with ``shell=True``.
        stdout
            Path where to save the stdout output.
        stderr
            Path where to save the stderr output.
        kwargs
            Configuration parameters (see `.configure`) to override. The
            configuration applies only to this run of ``solve-field`` and it
            is not saved.

        Returns
        -------
        `subprocess.CompletedProcess`
            The completed process.

        """

        options = self._options.copy()
        options.update(self._format_options(kwargs))

        if not isinstance(files, (tuple, list)):
            files = [files]

        t0 = time.time()

        args = self._build_command(list(map(str, files)), options=options)

        if shell:
            cmd = await asyncio.create_subprocess_shell(
                " ".join(args),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        else:
            cmd = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        stdout_bytes, stderr_bytes = await cmd.communicate()

        elapsed = time.time() - t0

        if stdout:
            with open(stdout, "wb") as out:
                if isinstance(args, str):
                    out.write(args.encode() + b"\n")
                else:
                    out.write(" ".join(args).encode() + b"\n")
                out.write(stdout_bytes)

        if stderr:
            with open(stderr, "wb") as err:
                err.write(stderr_bytes)

        return TimedProcess(cmd, elapsed)


def astrometrynet_quick(
    index_path: str,
    regions: pandas.DataFrame | Table | numpy.ndarray,
    ra: float,
    dec: float,
    pixel_scale: float,
    pixel_scale_factor_hi: float = 1.1,
    pixel_scale_factor_lo: float = 0.9,
    output_root: str | None = None,
    scales: int | list[int] | None = None,
    radius: float = 0.5,
    width: float | None = None,
    height: float | None = None,
    series: int | None = None,
    plot: bool = False,
    verbose: bool = False,
    **kwargs,
):
    """Quickly process a set of detections using astrometry.net.

    Parameters
    ----------
    index_path
        The path to the index files to use.
    regions
        A pandas data frame of source detections with at least three columns:
        ``x``, ``y``, and ``flux``. ``flux`` can be set to all ones if the
        fluxes have not been measured.
    ra
        The estimated RA of the field, in degrees.
    dec
        The estimated declination of the field, in degrees.
    pixel_scale
        The estimated pixel scale, in arcsec.
    pixel_scale_factor_lo
        A multiplicative factor for the pixel scale to calculate the
        lowest possible pixel scale to attempt.
    pixel_scale_factor_hi
        A multiplicative factor for the pixel scale to calculate the
        highest possible pixel scale to attempt.
    output_root
        The path where to save astrometry.net file. Must include the directory
        and a root after which input and output files will be named. If `None`,
        the xyls file is saved to a temporary file.
    scales
        Index files scales to use. Otherwise uses all index files.
    radius
        The radius, in degrees, around ``ra`` and ``dec`` to search.
    width
        The width of the image in pixels.
    height
        The height of the image in pixels.
    series
        The index series. If not provided and needed, will try to determine from
        ``index_path``.
    plot
        Whether to have astrometry.net generate plots.
    verbose
        Raise an error if the field is not solved.
    kwargs
        Other options to pass to `.AstrometryNet`.

    Returns
    -------
    wcs
        An astropy WCS with the solved field, or `None` if the field could
        not be solved.

    """

    if not isinstance(regions, Table):
        regions = pandas.DataFrame(regions)
        xyls = Table.from_pandas(regions)
    else:
        xyls = regions

    if output_root:
        output_root = os.path.abspath(output_root)
    else:
        output_root = tempfile.NamedTemporaryFile().name

    dirname = os.path.dirname(output_root)
    outfile = os.path.basename(output_root)

    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    index_path = os.path.abspath(index_path)
    index_files = glob("index-*.fit*", root_dir=index_path)

    if len(index_files) == 0:
        raise ValueError(f"No index files found in {index_path}.")

    backend_config = os.path.join(dirname, f"{outfile}.cfg")
    with open(backend_config, "w") as ff:
        ff.write(
            f"""inparallel
cpulimit 30
add_path {index_path}
"""
        )

        if scales is None:
            ff.write("autoindex\n")
        else:
            if series is None:
                match = re.match(r"index\-([0-9]+)\-", str(index_files[0]))
                if match:
                    series = int(int(match.group(1)) / 100) * 100
                else:
                    raise ValueError("Index series cannot be determined.")

            if isinstance(scales, int):
                scales = [scales]
            for scale in scales:
                series_scale = series + scale
                scale_files = glob(f"index-{series_scale}-*.fits", root_dir=index_path)
                for scale_file in scale_files:
                    ff.write(f"index {scale_file}\n")

    opts = dict(
        backend_config=str(backend_config),
        width=width or max(regions.loc[:, "x"]),
        height=height or max(regions.loc[:, "y"]),
        no_plots=not plot,
        scale_low=pixel_scale * pixel_scale_factor_lo,
        scale_high=pixel_scale * pixel_scale_factor_hi,
        scale_units="arcsecperpix",
        sort_column="flux",
        radius=radius,
    )
    opts.update(kwargs)

    astrometry = AstrometryNet(**opts)

    xyls_path = os.path.join(dirname, outfile + ".xyls")
    hdus = fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU(data=xyls.as_array())]  # type:ignore
    )
    hdus.writeto(xyls_path, overwrite=True)

    wcs_output = pathlib.Path(dirname) / (outfile + ".wcs")
    wcs_output.unlink(missing_ok=True)

    stdout = os.path.join(dirname, outfile + ".stdout")
    stderr = os.path.join(dirname, outfile + ".stderr")
    astrometry.run_sync(
        [xyls_path],
        stdout=stdout,
        stderr=stderr,
        ra=ra,
        dec=dec,
    )

    if wcs_output.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wcs = WCS(open(wcs_output).read())
        wcs_output.unlink()
        return wcs
    else:
        if verbose:
            if os.path.exists(stdout):
                stdout = open(stderr).read()
                stderr = open(stderr).read()
                raise RuntimeError(stdout + "\n" + stderr)
            else:
                raise RuntimeError("Unexpected error. No stderr was generated.")

    return None
