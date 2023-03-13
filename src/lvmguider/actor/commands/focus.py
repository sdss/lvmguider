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
async def fine(
    command: GuiderCommand,
    guess: float | None = None,
    step_size: float = 0.5,
    steps: int = 7,
):
    """Performs a focus sweep."""

    focuser = Focuser(command, guess=guess, step_size=step_size, steps=steps)
    await focuser.focus()

    return command.finish()
