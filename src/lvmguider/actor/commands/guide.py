#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-18
# @Filename: guide.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio

from typing import TYPE_CHECKING

import click
import numpy

from lvmguider.actor import lvmguider_parser
from lvmguider.guider import CriticalGuiderError, Guider
from lvmguider.maskbits import GuiderStatus
from lvmguider.tools import wait_until_cameras_are_idle


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand


__all__ = ["guide"]


def is_stopping(command: GuiderCommand):
    return command.actor.status & GuiderStatus.STOPPING


@lvmguider_parser.command()
@click.argument("RA", type=float)
@click.argument("DEC", type=float)
@click.argument("PA", type=float, default=0.0, required=False)
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
    help="The pixel of the full frame to use as pointing reference.",
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
    "--one",
    is_flag=True,
    help="Do one single iteration and exit.",
)
async def guide(
    command: GuiderCommand,
    ra: float,
    dec: float,
    pa: float = 0.0,
    exposure_time: float = 5.0,
    reference_pixel: tuple[float, float] | None = None,
    guide_tolerance: float | None = None,
    apply_corrections: bool = True,
    one: bool = False,
):
    """Starts the guide loop."""

    actor = command.actor
    MAX_EXPTIME: float = 18

    guider = Guider(
        command,
        (ra, dec, pa),
        pixel=reference_pixel,
        apply_corrections=apply_corrections,
    )
    command.actor.guider = guider

    if actor.status & GuiderStatus.NON_IDLE:
        return command.finish("Guider is not idle. Stop the guide loop.")

    await wait_until_cameras_are_idle(command)

    # Force the cameras to check the last image.
    command.actor.cameras.reset_seqno()

    while True:
        try:
            actor.guide_task = asyncio.create_task(
                guider.guide_one(
                    exposure_time,
                    guide_tolerance=guide_tolerance,
                )
            )
            await actor.guide_task
        except CriticalGuiderError as err:
            command.actor.status |= GuiderStatus.FAILED
            return command.fail(f"Stopping the guide loop due to critical error: {err}")
        except asyncio.CancelledError:
            # This means that the stop command was issued. All good.
            break
        except Exception as err:
            command.actor.status |= GuiderStatus.FAILED
            command.warning(f"Failed guiding with error: {err}")
            if "No solutions found" in str(err) and exposure_time < MAX_EXPTIME:
                exposure_time = numpy.clip(exposure_time * 1.5, 1, MAX_EXPTIME)
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
