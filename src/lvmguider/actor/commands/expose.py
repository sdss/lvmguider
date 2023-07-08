#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-07-08
# @Filename: expose.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from lvmguider.actor import lvmguider_parser
from lvmguider.maskbits import GuiderStatus


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand


__all__ = ["expose"]


@lvmguider_parser.command()
@click.option(
    "-t",
    "--exposure-time",
    type=float,
    default=5,
    help="Exposure time.",
)
@click.option(
    "-t",
    "--flavour",
    type=str,
    default="object",
    help="The type of exposure to take.",
)
@click.option(
    "--loop",
    is_flag=True,
    help="Loop exposures until guide stop is called.",
)
async def expose(
    command: GuiderCommand,
    exposure_time: float = 5.0,
    flavour="object",
    loop=False,
):
    """Exposes the cameras without guiding."""

    # Force the cameras to check the last image.
    command.actor.cameras.reset_seqno()

    if flavour not in ["object", "dark", "bias"]:
        return command.fail("Invalid flavour.")

    while True:
        await command.actor.cameras.expose(
            command,
            exposure_time=exposure_time,
            flavour=flavour,
            extract_sources=True,
        )
        if loop is False or (command.actor.status & GuiderStatus.STOPPING):
            break

    command.actor.status = GuiderStatus.IDLE

    return command.finish()
