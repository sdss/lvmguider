#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-03-13
# @Filename: focus.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from lvmguider.actor import lvmguider_parser
from lvmguider.focus import Focuser
from lvmguider.tools import wait_until_cameras_are_idle


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand


__all__ = ["focus"]


@lvmguider_parser.command()
@click.option(
    "-m",
    "--fit-method",
    type=click.Choice(["parabola", "spline"], case_sensitive=False),
    default="spline",
    help="The method used to fitting the data, either parabola or spline.",
)
@click.option(
    "-g",
    "--guess",
    type=float,
    help="Initial focus guess, in DT. If not provied, uses current focuser.",
)
@click.option(
    "-s",
    "--step-size",
    type=float,
    default=0.5,
    help="Step size, in DT.",
)
@click.option(
    "-n",
    "--steps",
    type=float,
    default=7,
    help="Number of steps (will be rounded to the closest odd number).",
)
@click.option(
    "-t",
    "--exposure-time",
    type=float,
    default=10,
    help="Exposure time in seconds.",
)
async def focus(
    command: GuiderCommand,
    fit_method: str = "spline",
    guess: float | None = None,
    step_size: float = 0.5,
    steps: int = 7,
    exposure_time: float = 10,
):
    """Performs a focus sweep."""

    await wait_until_cameras_are_idle(command)

    # Force the cameras to check the last image.
    command.actor.cameras.reset_seqno()

    focuser = Focuser(command.actor.telescope)

    try:
        await focuser.focus(
            command,
            initial_guess=guess,
            step_size=step_size,
            steps=steps,
            exposure_time=exposure_time,
            fit_method=fit_method,
        )
    except Exception as err:
        return command.fail(err)

    return command.finish()
