#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-18
# @Filename: guide.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from lvmguider.actor import lvmguider_parser
from lvmguider.guider import (
    calculate_telescope_offset,
    determine_pointing,
    offset_telescope,
)
from lvmguider.maskbits import GuiderStatus


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand


__all__ = ["guide"]


def is_stopping(command: GuiderCommand):
    return command.actor.status & GuiderStatus.STOPPING


@lvmguider_parser.group()
def guide():
    """Manages the guide loop."""


@guide.command()
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


@guide.command()
@click.argument("FIELDRA", type=float)
@click.argument("FIELDDEC", type=float)
@click.option(
    "--exposure-time",
    "-t",
    type=float,
    default=5,
    help="Exposure time.",
)
@click.option(
    "--reference-pixel",
    type=click.Tuple((float, float)),
    help="The pixel of the master frame to use as pointing reference.",
)
async def start(
    command: GuiderCommand,
    fieldra: float,
    fielddec: float,
    exposure_time: float = 5.0,
    reference_pixel: tuple[float, float] | None = None,
):
    """Starts the guide loop."""

    cameras = command.actor.cameras

    if command.actor.status & GuiderStatus.NON_IDLE:
        return command.finish("Guider is not idle.")

    while True:
        try:
            filenames, sources = await cameras.expose(
                command,
                extract_sources=True,
                exposure_time=exposure_time,
            )
            if sources is None:
                raise ValueError("No sources found.")
        except Exception as err:
            command.warning(f"Failed taking exposure: {err}")
            if is_stopping(command):
                break
            continue

        try:
            ra_p, dec_p = await determine_pointing(
                command.actor.telescope,
                filenames,
                pixel=reference_pixel,
            )
        except Exception as err:
            command.warning(f"Failed determining telescope pointing: {err}")
            if is_stopping(command):
                break
            continue

        offset, sep = calculate_telescope_offset((ra_p, dec_p), (fieldra, fielddec))
        command.info(
            pointing_correction={
                "separation": sep,
                "offset_measured": offset,
                "offset_applied": offset,
            }
        )

        try:
            await offset_telescope(command, *offset)
        except RuntimeError as err:
            command.warning(f"Failed applying pointing correction: {err}")
            if is_stopping(command):
                break
            continue

    command.actor.status = GuiderStatus.IDLE

    return command.finish("The guide loop has finished.")


@guide.command()
async def stop(command: GuiderCommand):
    """Stops the guide loop."""

    command.actor.status |= GuiderStatus.STOPPING
    return command.finish("Stopping the guide loop.")
