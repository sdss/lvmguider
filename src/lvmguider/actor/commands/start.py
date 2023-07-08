#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-18
# @Filename: start.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio

from typing import TYPE_CHECKING

import click
import numpy

from lvmguider.actor import lvmguider_parser
from lvmguider.guider import CriticalGuiderError, Guider
from lvmguider.maskbits import GuiderStatus


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand


__all__ = ["start"]


def is_stopping(command: GuiderCommand):
    return command.actor.status & GuiderStatus.STOPPING


@lvmguider_parser.command()
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
    initial_exposure_time = exposure_time

    guider = Guider(command, (fieldra, fielddec), pixel=reference_pixel)
    command.actor.guider = guider

    if reference_frame is not None:
        mode = "guide"
        guider.set_reference_frames(reference_frame)

    if mode not in ["auto", "guide", "acquire"]:
        return command.fail("Invalid mode. Use mode auto, guide, or acquire.")

    if mode == "guide" and reference_frame is None:
        return command.fail("--mode guide requires using --reference-frame.")

    if actor.status & GuiderStatus.NON_IDLE:
        return command.finish("Guider is not idle. Stop the guide loop.")

    if mode == "auto" or mode == "acquire":
        actor.status = GuiderStatus.ACQUIRING
    else:
        actor.status = GuiderStatus.GUIDING

    while True:
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
            if "No solutions found" in str(err):
                if exposure_time < initial_exposure_time * 4:
                    exposure_time = numpy.clip(exposure_time * 1.5, a_min=1, a_max=18)
                    exposure_time = numpy.round(exposure_time, 1)
                    command.warning(f"Exposure time increased to {exposure_time:.1f} s")
        finally:
            if is_stopping(command):
                break

        if one:
            break

    actor.status = GuiderStatus.IDLE
    command.actor.guider = None

    return command.finish("The guide loop has finished.")
