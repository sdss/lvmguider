#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-03-13
# @Filename: focus.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from datetime import datetime, timezone
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
@click.argument("FOCUS_VALUE", type=float, required=False)
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
    focus_value: float | None = None,
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

    ref_focus = command.actor._reference_focus

    if focus_value is None:
        if ref_focus is None:
            focus_value = await focuser.get_from_temperature(command, c_temp)
            reference = True
            if relative:
                command.warning("No reference focus found. Using bench temperature.")
                relative = False
        else:
            delta_t = c_temp - ref_focus.temperature
            focus_model_a: float = command.actor.config["focus.model.a"]
            focus_value = ref_focus.focus + delta_t * focus_model_a
            relative = False  # We always calculate an absolute focus here.

            command.debug(
                f"Reference temperature: {c_temp:.2f} C. "
                f"Delta temperature: {delta_t:.2f} C."
            )

            delta_focus = round(focus_value - c_focus, 2)
            if abs(delta_focus) > 0.01:
                command.debug(f"Focus will be adjusted by {delta_focus:.2f} DT.")
            else:
                return command.finish(
                    f"Delta focus {delta_focus:.2f} DT is too small. "
                    "Focus was not adjusted."
                )

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

    return command.finish(f"New focuser position: {focus_value:.2f} DT.")


@lvmguider_parser.command(name="focus-info")
async def focus_info(command: GuiderCommand):
    """Returns the current and reference focus position."""

    focuser = Focuser(command.actor.telescope)

    current_temperature = await focuser.get_bench_temperature(command)
    current_focus = await focuser.get_focus_position(command)

    ref = command.actor._reference_focus
    timestamp = datetime.fromtimestamp(ref.timestamp).isoformat() if ref else None

    command.info(
        reference_focus={
            "focus": ref.focus if ref else None,
            "fwhm": ref.fwhm if ref else None,
            "temperature": ref.temperature if ref else None,
            "timestamp": timestamp,
        }
    )

    command.info(
        current_focus={
            "focus": current_focus,
            "temperature": current_temperature,
            "delta_temperature": current_temperature - ref.temperature if ref else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    return command.finish()
