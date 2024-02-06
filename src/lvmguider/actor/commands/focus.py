#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-03-13
# @Filename: focus.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from time import time

from typing import TYPE_CHECKING

import click

from lvmguider.actor import lvmguider_parser
from lvmguider.actor.actor import ReferenceFocus
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
    help="Initial focus guess, in DT. If not provied, "
    "uses a temperature-based estimate.",
)
@click.option(
    "-s",
    "--step-size",
    type=float,
    default=0.2,
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

    if guess is None:
        guess = await focuser.get_from_temperature(command)

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


@lvmguider_parser.command("adjust-focus")
@click.argument("FOCUS_VALUE", type=float)
@click.option(
    "--relative",
    is_flag=True,
    help="Adjusts the focus relative to the current value.",
)
@click.option(
    "--reference",
    is_flag=True,
    help="Set as reference focus position.",
)
async def adjust_focus(
    command: GuiderCommand,
    focus_value: float,
    relative: bool = False,
    reference: bool = False,
):
    """Adjusts the focus to a specific value.

    If FOCUS_VALUE is not provided, the focus will be adjusted to the
    temperature-estimated best focus.

    """

    focuser = Focuser(command.actor.telescope)

    c_temp = await focuser.get_bench_temperature(command)
    c_focus = await focuser.get_focus_position(command)

    if focus_value is None:
        if command.actor._reference_focus is None:
            focus_value = await focuser.get_from_temperature(command, c_temp)
            reference = True
            if relative:
                command.warning("No reference focus found. Using bench temperature.")
                relative = False
        else:
            delta_t = c_temp - command.actor._reference_focus.temperature
            focus_value = delta_t * command.actor.config["focus.model.a"]
            relative = True  # Always relative to the reference focus.

    if relative:
        focus_value = c_focus + focus_value

    await focuser.goto_focus_position(command, focus_value)
    if reference:
        command.actor._reference_focus = ReferenceFocus(
            focus_value,
            -999.0,
            c_temp,
            time(),
        )

    return command.finish(f"Focus adjusted to {focus:.2f}.")
