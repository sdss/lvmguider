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


@lvmguider_parser.group()
def focus():
    """Commands to Focus the telescope."""

    pass


@focus.command()
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
    "--exp-time",
    type=float,
    default=10,
    help="Exposure time in seconds.",
)
async def fine(
    command: GuiderCommand,
    guess: float | None = None,
    step_size: float = 0.5,
    steps: int = 7,
    exp_time: float = 10,
):
    """Performs a focus sweep."""

    focuser = Focuser(
        command,
        guess=guess,
        step_size=step_size,
        steps=steps,
        exp_time=exp_time,
    )

    try:
        await focuser.focus()
    except Exception as err:
        return command.fail(f"Failed to focus with error: {err}")

    return command.finish()
