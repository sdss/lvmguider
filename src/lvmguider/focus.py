#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-03-13
# @Filename: focus.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio

from typing import TYPE_CHECKING

import numpy
import pandas
from astropy.io import fits
from scipy.optimize import curve_fit

from lvmguider.extraction import extract_marginal
from lvmguider.tools import run_in_executor


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand


def fit_hyperbola(
    x_arr: list[float],
    y_arr: list[float],
    y_err: list[float],
) -> tuple[float, float]:
    """Fit a hyperbola.

    Returns
    -------
        Minimum of hyperbola and its uncertainty.

    """

    # initial guess
    ic = numpy.argmin(y_arr)
    ix = numpy.argmax(y_arr)
    b = y_arr[ic]
    c = x_arr[ic]
    x = x_arr[ix]
    slope = numpy.abs((y_arr[ic] - y_arr[ix]) / (c - x))
    a = b / slope

    # initial values
    p0 = [a, b, c]

    # fit
    coeffs, cov = curve_fit(
        lambda xx, aa, bb, cc: bb * numpy.sqrt((xx - cc) ** 2 / aa**2 + 1.0),
        x_arr,
        y_arr,
        sigma=y_err,
        p0=p0,
    )

    # return result
    return coeffs[2], cov[2][2]


class Focuser:
    """Performs focus on a telescope."""

    def __init__(
        self,
        command: GuiderCommand,
        guess: float | None = None,
        step_size: float = 0.5,
        steps: int = 7,
        exp_time: float = 5.0,
    ):
        self.command = command
        self.initial_guess = guess
        self.step_size = step_size
        self.steps = int(steps)
        self.exp_time = exp_time

        if self.steps % 2 == 0:
            self.steps += 1

        self.telescope = self.command.actor.telescope

        self.agcam_actor = f"lvm.{self.telescope}.agcam"
        self.foc_actor = f"lvm.{self.telescope}.foc"

    async def focus(self, max_cov: float = 5.0):
        """Performs the focus routine."""

        if self.initial_guess is None:
            cmd = await self.command.send_command(self.foc_actor, "getPosition")
            if cmd.status.did_fail:
                raise RuntimeError("Failed retrieving position from focuser.")
            self.initial_guess = cmd.replies.get("Position")
            self.command.debug(f"Using focuser position: {self.initial_guess} DT")

        focus_grid = numpy.arange(
            self.initial_guess - (self.steps // 2) * self.step_size,
            self.initial_guess + (self.steps // 2 + 1) * self.step_size,
            self.step_size,
        )

        if numpy.any(focus_grid <= 0):
            raise ValueError("Focus values out of range.")

        mean_fwhm = []
        source_list = []
        for focus_position in focus_grid:
            await self.goto_focus_position(focus_position)
            filenames = await self.expose_cameras()

            sources = pandas.concat(
                await asyncio.gather(*[self.extract_sources(fn) for fn in filenames])
            )
            if len(sources) == 0:
                self.command.warning(f"No sources detected at {focus_position} DT.")
                mean_fwhm.append(1e6)
                continue

            sources.loc[:, "dt"] = focus_position

            fwhm = sources.loc[:, "xstd"].median()
            self.command.info(
                focus_point=dict(
                    focus=focus_position,
                    n_sources=len(sources),
                    fwhm=round(fwhm, 1),
                )
            )
            mean_fwhm.append(fwhm)

            source_list.append(sources)

        sources = pandas.concat(source_list)

        best, cov = fit_hyperbola(
            sources.dt.tolist(),
            sources.xstd.tolist(),
            sources.xrms.tolist(),
        )

        if cov > max_cov:
            best_focus_idx = numpy.argmin(mean_fwhm)
            self.initial_guess = focus_grid[best_focus_idx]
            self.command.warning(
                "Covariance not reached. Trying again "
                f"with initial guess {self.initial_guess:.1f} DT."
            )
            await self.focus()
        else:
            self.command.info(
                best_focus=dict(
                    focus=numpy.round(best, 1),
                    n_sources=len(sources),
                    cov=numpy.round(cov, 2),
                )
            )
            await self.goto_focus_position(numpy.round(best, 2))

    async def goto_focus_position(self, focus_position: float):
        """Moves the focuser to a position."""

        cmd = await self.command.send_command(
            self.foc_actor,
            f"moveAbsolute {focus_position} DT",
        )
        if cmd.status.did_fail:
            raise RuntimeError(f"Failed reaching focus {focus_position:.1f} DT.")
        if cmd.replies.get("AtLimit") is True:
            raise RuntimeError("Hit a limit while focusing.")

    async def expose_cameras(self):
        """Exposes the cameras and returns the filenames."""

        cmd = await self.command.send_command(
            self.agcam_actor,
            f"expose {self.exp_time}",
        )
        if cmd.status.did_fail:
            raise RuntimeError("Failed while exposing cameras.")

        filenames: set[str] = set()
        for reply in cmd.replies:
            for cam_name in ["east", "west"]:
                if cam_name in reply.message:
                    if reply.message[cam_name].get("state", None) == "written":
                        filenames.add(reply.message[cam_name]["filename"])

        if len(filenames) == 0:
            raise ValueError("Exposure did not produce any images.")

        return list(filenames)

    async def extract_sources(self, filename: str):
        """Extracts sources from a file."""

        data = fits.getdata(filename)

        sources = await run_in_executor(extract_marginal, data, box_size=31)

        return sources
