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
from lvmguider.guider import CriticalGuiderError, Guider
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
    "--mode",
    type=str,
    default="auto",
    help="The guiding mode: auto, acquire, guide.",
)
@click.option(
    "--reference-frame",
    type=int,
    help="The sequence number of the AG frames to use for guiding. "
    "Implies --mode guide.",
)
@click.option(
    "--guide-tolerance",
    type=float,
    help="The acquisition tolerance, in arcsec, after which guiding will be started.",
)
@click.option(
    "--apply-corrections/--no-apply-corrections",
    is_flag=True,
    default=True,
    show_default=True,
    help="Whether apply the measured corrections.",
)
@click.option(
    "--use-motor-offsets/--no-use-motor-offsets",
    is_flag=True,
    default=True,
    show_default=True,
    help="Whether to apply corrections as motor offsets.",
)
@click.option(
    "--use-individual-images/--no-use-individual-images",
    is_flag=True,
    default=True,
    show_default=True,
    help="Whether to use individual images to generate the WCS during acquisition.",
)
@click.option(
    "--one",
    is_flag=True,
    help="Do one single iteration and exit.",
)
async def start(
    command: GuiderCommand,
    fieldra: float,
    fielddec: float,
    exposure_time: float = 5.0,
    reference_pixel: tuple[float, float] | None = None,
    mode: str = "auto",
    reference_frame: int | None = None,
    guide_tolerance: float | None = None,
    apply_corrections: bool = True,
    use_motor_offsets: bool = True,
    use_individual_images: bool = False,
    one: bool = False,
):
    """Starts the guide loop."""

    actor = command.actor

    guider = Guider(command, (fieldra, fielddec), pixel=reference_pixel)

    if reference_frame is not None:
        mode = "guide"
        guider.set_reference_frames(reference_frame)

    if mode not in ["auto", "guide", "acquire"]:
        return command.fail("Invalid mode. Use mode auto, guide, or acquire.")

    if mode == "guide" and reference_frame is None:
        return command.fail("--mode guide requires using --reference-frame.")

    if actor.status & GuiderStatus.NON_IDLE:
        return command.finish("Guider is not idle. Stop the guide loop.")

    while True:
        actor.status = GuiderStatus.GUIDING

        try:
            actor.guide_task = asyncio.create_task(
                guider.guide_one(
                    exposure_time,
                    mode=mode,
                    guide_tolerance=guide_tolerance,
                    apply_correction=apply_corrections,
                    use_individual_images=use_individual_images,
                    use_motor_offsets=use_motor_offsets,
                )
            )
            await actor.guide_task
        except CriticalGuiderError as err:
            return command.fail(f"Stopping the guide loop due to critical error: {err}")
        except Exception as err:
            command.warning(f"Failed guiding with error: {err}")
        finally:
            if is_stopping(command):
                break

        if one:
            break

    actor.status = GuiderStatus.IDLE
    command.actor.guider = None

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
        command.actor.guider = None
        return command.finish("Guider was forcibly stopped.")

    if command.actor.status & GuiderStatus.STOPPING:
        return command.fail("Guider loop is already stopping.")

    command.actor.status |= GuiderStatus.STOPPING
    return command.finish("Stopping the guide loop.")


@guide.command()
@click.argument("PIXEL-X", type=float)
@click.argument("PIXEL-Z", type=float)
async def set_pixel(command: GuiderCommand, pixel_x: float, pixel_z: float):
    """Sets the master frame pixel coordinates on which to guide.

    This command can be issued during active guiding to change the pointing
    of the telescope.

    """

    if not command.actor.guider:
        return command.fail("Guider is not active.")

    command.actor.guider.set_pixel(pixel_x, pixel_z)

    return command.finish(f"Guide pixel is now ({pixel_x:.2f}, {pixel_z:.2f}).")
