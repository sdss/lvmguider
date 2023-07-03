#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-18
# @Filename: guide.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
from contextlib import suppress

from typing import TYPE_CHECKING

import click

from lvmguider.actor import lvmguider_parser
from lvmguider.guider import Guider
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
@click.option(
    "--apply-corrections/--no-apply-corrections",
    is_flag=True,
    default=True,
    help="Whether apply the measured corrections.",
)
@click.option(
    "--use-individual-images",
    is_flag=True,
    default=True,
    help="Whether to use individual images to generate the WCS.",
)
async def start(
    command: GuiderCommand,
    fieldra: float,
    fielddec: float,
    exposure_time: float = 5.0,
    reference_pixel: tuple[float, float] | None = None,
    apply_corrections: bool = True,
    use_individual_images: bool = False,
):
    """Starts the guide loop."""

    actor = command.actor

    if actor.status & GuiderStatus.NON_IDLE:
        return command.finish("Guider is not idle. Stop the guide loop.")

    guider = Guider(command, (fieldra, fielddec), pixel=reference_pixel)

    while True:
        actor.status = GuiderStatus.GUIDING

        try:
            actor.guide_task = asyncio.create_task(
                guider.guide_one(
                    exposure_time,
                    apply_correction=apply_corrections,
                    use_individual_images=use_individual_images,
                )
            )
            await actor.guide_task
        except Exception as err:
            command.warning(f"Failed guiding with error: {err}")
        finally:
            if is_stopping(command):
                break

    actor.status = GuiderStatus.IDLE

    return command.finish("The guide loop has finished.")


@guide.command()
@click.option("--now", is_flag=True, help="Aggressively stops the loop.")
async def stop(command: GuiderCommand, now=False):
    """Stops the guide loop."""

    status = command.actor.status
    if status & GuiderStatus.IDLE:
        return command.finish("Guider is not active.")

    if now:
        if command.actor.guide_task and not command.actor.guide_task.done():
            command.actor.guide_task.cancel()
            with suppress(asyncio.CancelledError):
                await command.actor.guide_task
        return command.finish("Guider was forcibly stopped.")

    if command.actor.status & GuiderStatus.STOPPING:
        return command.fail("Guider loop is already stopping.")

    command.actor.status |= GuiderStatus.STOPPING
    return command.finish("Stopping the guide loop.")
