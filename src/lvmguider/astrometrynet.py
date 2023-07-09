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
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass
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
            "no-tweak": True,
            "tweak-order": 0,
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

        flags = ["no-plots", "sort-ascending", "overwrite", "no-tweak"]

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
    index_paths: dict[int, str],
    regions: pandas.DataFrame,
    ra: float,
    dec: float,
    pixel_scale: float,
    pixel_scale_factor_hi: float = 1.1,
    pixel_scale_factor_lo: float = 0.9,
    output_root: str | None = None,
    scales: dict[int, list[int]] | None = None,
    radius: float = 0.5,
    width: float | None = None,
    height: float | None = None,
    plot: bool = False,
    cpulimit: int = 30,
    raise_on_unsolved: bool = False,
    **kwargs,
) -> AstrometrySolution:
    """Quickly process a set of detections using astrometry.net.

    Parameters
    ----------
    index_paths
        The paths to the index files to use. A dictionary of series number
        to path on disk, e.g.,
        ``{5200: "/data/astrometrynet/5200", 4100: "/data/astrometrynet/5100"}``.
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
        Index scales to use. Otherwise uses all index files. The format
        is a dictionary of series number to a list of scales, e.g.,
        ``{5200: [4, 5], 4100: [10, 11]}``.
    radius
        The radius, in degrees, around ``ra`` and ``dec`` to search.
    width
        The width of the image in pixels.
    height
        The height of the image in pixels.
    plot
        Whether to have astrometry.net generate plots.
    cpulimit
        Maximum time to wait for a solution, in seconds.
    raise_on_unsolved
        Raise an error if the field is not solved.
    kwargs
        Other options to pass to `.AstrometryNet`.

    Returns
    -------
    solution
        An `.AstrometrySolution` object with the astrometric solution.

    """

    if "flux" not in regions:
        regions.loc[:, "flux"] = 1

    xyls = Table.from_pandas(regions)

    if output_root:
        output_root = os.path.abspath(output_root)
    else:
        output_root = tempfile.NamedTemporaryFile().name
        plot = False

    dirname = os.path.dirname(output_root)
    outfile = os.path.basename(output_root)

    corr_file = os.path.join(dirname, outfile + ".corr")

    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    backend_config = os.path.join(dirname, f"{outfile}.cfg")
    with open(backend_config, "w") as ff:
        ff.write("inparallel\n")
        ff.write(f"cpulimit {cpulimit}\n")

        for index_path in index_paths.values():
            ff.write(f"add_path {os.path.abspath(index_path)}\n")

        for series, index_path in index_paths.items():
            index_path = os.path.abspath(index_path)
            this_scales = scales.get(series, None) if scales is not None else None
            if this_scales is None:
                scale_files = glob("index-*", root_dir=index_path)
                for scale_file in scale_files:
                    ff.write(f"index {scale_file}\n")
            else:
                for this_scale in this_scales:
                    series_scale = series + this_scale
                    scale_files = glob(f"index-{series_scale}*", root_dir=index_path)
                    if len(scale_files) == 0:
                        warnings.warn(f"No indices found for scale {series_scale}.")
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
        corr=corr_file,
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

    stdout_s = open(stdout).read() if os.path.exists(stdout) else ""
    stderr_s = open(stderr).read() if os.path.exists(stderr) else ""

    solution = AstrometrySolution(wcs=None, xyls=xyls.to_pandas(), stdout="", stderr="")

    if not wcs_output.exists():
        if raise_on_unsolved:
            if stdout_s != "":
                raise RuntimeError(stdout_s + "\n" + stderr_s)
            else:
                raise RuntimeError("Unexpected error. No stderr was generated.")
        else:
            return solution

    warnings.simplefilter("ignore")
    wcs = WCS(open(wcs_output).read(), relax=True)

    solution.wcs = wcs
    solution.stdout = open(stdout).read()
    solution.stderr = open(stderr).read()

    if os.path.exists(corr_file):
        stars = Table.read(corr_file).to_pandas()
        solution.stars = stars

    return solution


@dataclass
class AstrometrySolution:
    """An astrometric solution."""

    wcs: WCS | None
    xyls: pandas.DataFrame
    stdout: str
    stderr: str
    stars: pandas.DataFrame | None = None

    def __repr__(self) -> str:
        return f"<AstrometrySolution (solved={self.solved})>"

    @property
    def solved(self):
        """Whether a solution was found."""

        return self.wcs is not None

    @property
    def transformation_matrix(self):
        """Returns the transformation matrix."""

        if not self.wcs:
            raise RuntimeError("No solution found.")

        return self.wcs.wcs.cd

    @property
    def rotation(self):
        """Returns the rotation angle."""

        if not self.wcs:
            raise RuntimeError("No solution found.")

        tm = self.transformation_matrix
        return numpy.degrees(numpy.arctan2(tm[0, 0], tm[1, 0]))

    @property
    def pixel_scale(self):
        """Returns the pixel scale."""

        if not self.wcs:
            raise RuntimeError("No solution found.")

        tm = self.transformation_matrix

        return numpy.abs(tm[0, 0] * 3600)

    def pixel_to_world(self, x: float, y: float):
        """Returns the sky coordinate for a pixel position."""

        if not self.wcs:
            raise RuntimeError("No solution found.")

        return self.wcs.pixel_to_world(x, y)
