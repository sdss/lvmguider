#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-03-13
# @Filename: focus.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import pathlib

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn
from astropy.io import fits
from scipy.interpolate import UnivariateSpline

from sdsstools.logger import get_logger

from lvmguider.tools import run_in_executor


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand


def fit_parabola(x_arr: list[float], y_arr: list[float], y_err: list[float]):
    """Fits a parabola to the data."""

    w = 1 / numpy.array(y_err)
    a, b, c = numpy.polyfit(x_arr, y_arr, 2, w=w, full=False)

    corr_matrix = numpy.corrcoef(y_arr, numpy.polyval([a, b, c], x_arr))
    corr = corr_matrix[0, 1]
    R_sq = corr**2

    return (a, b, c, R_sq)


def fit_spline(x: numpy.ndarray, y: numpy.ndarray, w: numpy.ndarray | None = None):
    """Fits a spline to data."""

    spl = UnivariateSpline(x, y)

    corr_matrix = numpy.corrcoef(y, spl(x))
    R2 = corr_matrix[0, 1] ** 2

    return spl, R2


class Focuser:
    """Focus a telescope.

    Parameters
    ----------
    telescope
        The telescope being focused: ``'sci'``, ``'spec'``, ``'skye'``,
        or ``'skyw'``.

    """

    def __init__(self, telescope: str):
        self.telescope = telescope
        self.foc_actor = f"lvm.{self.telescope}.foc"  # Focuser

    async def focus(
        self,
        command: GuiderCommand,
        initial_guess: float | None = None,
        step_size: float = 0.5,
        steps: int = 7,
        exposure_time: float = 5.0,
        fit_method="parabola",
        plot: bool = True,
        plot_dir: str = "qa/focus",
    ) -> tuple[pandas.DataFrame, dict]:
        """Performs the focus routine.

        Parameters
        ----------
        command
            The actor command to use to talk to other actors.
        initial_guess
            An initial guess of the focus position, in DT.
        step_size
            The resolution of the focus sweep in DT steps.
        steps
            Number of steps in the focus sweep. Must be an odd number.
        exposure_time
            The exposure time for each AG frame.
        fit_method
            The method used to fit the data. Either ``'parabola'`` for a second
            order polynomial fit, or ``'spline'`` to fit a univariate spline
            to the data.
        plot
            Whether to produce focus plots. The plots are saved asynchronously
            and this method may return before the plots are available.
        plot_dir
            The path where to save the focus plot. Always relative to the
            path of the AG frames. The plot will be saved as
            ``{plot_dir}/focus_{telescope}_{frame0}_{frame1}.pdf`` where
            ``{frame0}`` and ``{frame1}`` are the first and last frame number
            of the sweep sequence.

        """

        cameras = command.actor.cameras

        steps = int(steps)
        if steps % 2 == 0:
            steps += 1

        if initial_guess is None:
            cmd = await command.send_command(self.foc_actor, "getPosition")
            if cmd.status.did_fail:
                raise RuntimeError("Failed retrieving position from focuser.")
            initial_guess = cmd.replies.get("Position")
            command.debug(f"Using focuser position: {initial_guess} DT")

        assert initial_guess is not None

        focus_grid = numpy.arange(
            initial_guess - (steps // 2) * step_size,
            initial_guess + (steps // 2 + 1) * step_size,
            step_size,
        )

        if numpy.any(focus_grid <= 0):
            raise ValueError("Focus values out of range.")

        source_list: list[pandas.DataFrame] = []
        files: list[pathlib.Path] = []
        framenos: list[int] = []

        for focus_position in focus_grid:
            await self.goto_focus_position(command, focus_position)

            step_files, frameno, sources = await cameras.expose(
                command,
                exposure_time=exposure_time,
                extract_sources=True,
            )

            files += step_files
            framenos.append(frameno)

            if sources is None or len(sources) == 0:
                command.warning(f"No sources detected at {focus_position} DT.")
                continue

            asources = pandas.concat(sources)
            if len(asources) == 0:
                command.warning(
                    f"No sources found for focus position {focus_position} DT. "
                    "Skipping this point."
                )
                continue

            asources["dt"] = focus_position

            fwhm = float(numpy.percentile(asources.loc[asources.valid == 1].fwhm, 25))
            command.info(
                focus_point=dict(
                    focus=round(focus_position, 2),
                    n_sources=len(asources),
                    fwhm=round(fwhm, 2),
                )
            )

            source_list.append(asources)

        if len(source_list) < 3:
            raise ValueError("Insufficient number of focus points.")

        sources = pandas.concat(source_list)

        fit_data = self.fit_focus(
            sources,
            fit_method=fit_method,
        )

        command.info(
            best_focus=dict(
                focus=numpy.round(fit_data["xmin"], 2),
                fwhm=numpy.round(fit_data["ymin"], 2),
                r2=numpy.round(fit_data["R2"], 3),
            )
        )

        if plot:
            frame0 = min(framenos)
            frame1 = max(framenos)
            basepath = files[-1].absolute().parent

            plot_name = f"focus_{self.telescope}_{frame0}_{frame1}.pdf"
            plot_path = basepath / plot_dir / plot_name

            asyncio.create_task(
                run_in_executor(
                    self.plot,
                    sources,
                    fit_method,
                    plot_path,
                    fit_params=fit_data,
                )
            )

        await self.goto_focus_position(command, numpy.round(fit_data["xmin"], 2))

        return sources, fit_data

    async def goto_focus_position(self, command: GuiderCommand, focus_position: float):
        """Moves the focuser to a position."""

        cmd = await command.send_command(
            self.foc_actor,
            f"moveAbsolute {focus_position} DT",
        )
        if cmd.status.did_fail:
            raise RuntimeError(f"Failed reaching focus {focus_position:.2f} DT.")
        if cmd.replies.get("AtLimit") is True:
            raise RuntimeError("Hit a limit while focusing.")

    def fit_focus(self, sources: pandas.DataFrame, fit_method: str = "parabola"):
        """Fits the data and returns the best focus and measured FWHM."""

        sources_valid = sources.loc[sources.valid == 1]

        if fit_method == "parabola":
            a, b, c, R2 = fit_parabola(
                sources_valid.dt.tolist(),
                sources_valid.fwhm.tolist(),
                sources_valid.xrms.tolist(),
            )

            xmin = -b / 2 / a
            ymin = a * xmin**2 + b * xmin + c

            return {"xmin": xmin, "ymin": ymin, "R2": R2, "coeffs": [a, b, c]}

        elif fit_method == "spline":
            fwhm = sources_valid.groupby("dt").apply(
                lambda gg: pandas.Series(
                    {
                        "fwhm": numpy.percentile(gg.fwhm, 25),
                        "std": gg.fwhm.std(),
                    }
                )
            )

            x = fwhm.index.to_numpy()
            y = fwhm.fwhm.to_numpy()
            w = 1.0 / fwhm["std"].to_numpy()
            spline, R2 = fit_spline(x, y, w=w)

            x_refine = numpy.arange(numpy.min(x), numpy.max(x), 0.01)
            y_refine = spline(x_refine)

            arg_min = numpy.argmin(y_refine)
            xmin = x_refine[arg_min]
            ymin = y_refine[arg_min]

            return {"xmin": xmin, "ymin": ymin, "R2": R2, "spline": spline}

        else:
            raise ValueError("Invalid fit method.")

    def plot(
        self,
        data: pandas.DataFrame,
        fit_method: str,
        plot_path: pathlib.Path,
        fit_params: dict = {},
    ):
        """Produces a plot for the focus sequence."""

        seaborn.set_theme(palette="deep")

        data_valid = data.loc[(data.xfitvalid == 1) & (data.yfitvalid == 1)]

        with plt.ioff():
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 8))

            ax.scatter(
                data_valid.dt,
                data_valid.fwhm,
                s=10,
                color="y",
                ec="None",
                alpha=0.3,
            )

            fwhm_median = data_valid.groupby("dt").fwhm.median()

            ax.scatter(
                fwhm_median.index,
                fwhm_median,
                ec="None",
                color="r",
                alpha=0.8,
                s=20,
            )

            xmin = numpy.min(data_valid.dt.to_numpy()).astype(numpy.float32)
            xmax = numpy.max(data_valid.dt.to_numpy()).astype(numpy.float32)

            xx = numpy.arange(xmin - 0.5, xmax + 0.5, 0.01)

            if fit_method == "parabola":
                a, b, c = fit_params["coeffs"]
                yy = a * xx**2 + b * xx + c

            elif fit_method == "spline":
                spline = fit_params["spline"]
                yy = spline(xx)

            else:
                raise ValueError("Invalid fit_method.")

            ax.plot(xx, yy, "m-", zorder=20)

            ax.axvline(
                x=fit_params["xmin"],
                linestyle="dashed",
                color="k",
                alpha=0.5,
                zorder=10,
            )

            ax.set_ylim(0.5, max(fwhm_median.values.astype(float)) + 0.5)
            ax.set_xlabel("Focuser position [DT]")
            ax.set_ylabel("FWHM [arcsec]")

            ax.set_title(
                f"Best focus: {fit_params['xmin']:.2f} DT; "
                f"FWHM: {fit_params['ymin']:.2f} arcsec"
            )

            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(plot_path))

            plt.close(fig)

        seaborn.reset_defaults()

    def reprocess(
        self,
        files: list[str] | list[pathlib.Path],
        fit_method="parabola",
        plot: bool = True,
        plot_dir: str = "qa/focus",
    ):
        """Reprocesses a list of files.

        Parameters
        ----------
        files
            A list of files to reprocess. They must correspond to a focus
            sweep taken using this telescope.
        fit_method
            The method used to fit the data. Either ``'parabola'`` for a second
            order polynomial fit, or ``'spline'`` to fit a univariate spline
            to the data.
        plot
            Whether to produce focus plots.
        plot_dir
            The path where to save the focus plot. Always relative to the
            path of the AG frames. The plot will be saved as
            ``{plot_dir}/focus_{telescope}_{frame0}_{frame1}.pdf`` where
            ``{frame0}`` and ``{frame1}`` are the first and last frame number
            of the sweep sequence.

        """

        log = get_logger(f"lvmguider.{self.telescope}.reprocess", use_rich_handler=True)

        log.info(f"Reprocessing {len(files)} files.")

        sources = []
        framenos = []

        for file in files:
            framenos.append(int(str(file).split(".")[-2].split("_")[1]))

            hdul = fits.open(str(file))
            focus_position = hdul["RAW"].header["FOCUSDT"]

            sources.append(pandas.DataFrame(hdul["SOURCES"].data))
            sources[-1]["dt"] = focus_position

        data = pandas.concat(sources)

        fit_data = self.fit_focus(
            data,
            fit_method=fit_method,
        )

        log.info(
            f"Best focus: {fit_data['xmin']:.2f} DT; "
            f"FWHM: {fit_data['ymin']: .2f} arcsec; "
            f"R2: {fit_data['R2']:.3f}"
        )

        if plot:
            frame0 = min(framenos)
            frame1 = max(framenos)
            basepath = pathlib.Path(files[-1]).absolute().parent

            plot_name = f"focus_{self.telescope}_{frame0}_{frame1}.pdf"
            plot_path = basepath / plot_dir / plot_name

            self.plot(data, fit_method, plot_path, fit_params=fit_data)
            log.info(f"QA plot saved to {plot_path!s}")
