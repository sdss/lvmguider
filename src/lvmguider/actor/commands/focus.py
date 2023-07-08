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


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand


__all__ = ["focus"]


@lvmguider_parser.command()
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
    guess: float | None = None,
    step_size: float = 0.5,
    steps: int = 7,
    exposure_time: float = 10,
):
    """Performs a focus sweep."""

    # Force the cameras to check the last image.
    command.actor.cameras.reset_seqno()

    focuser = Focuser(
        command,
        guess=guess,
        step_size=step_size,
        steps=steps,
        exposure_time=exposure_time,
    )

    try:
        await focuser.focus()
    except Exception as err:
        return command.fail(err)

    return command.finish()
