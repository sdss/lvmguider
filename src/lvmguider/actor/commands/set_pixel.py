#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-07-08
# @Filename: set_pixel.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from lvmguider.actor import lvmguider_parser


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand


__all__ = ["set_pixel", "reset_pixel"]


@lvmguider_parser.command()
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


@lvmguider_parser.command()
async def reset_pixel(command: GuiderCommand):
    """Resets the master frame pixel on which to guide to the central pixel..

    This command can be issued during active guiding to change the pointing
    of the telescope.

    """

    if not command.actor.guider:
        return command.fail("Guider is not active.")

    pixel_x, pixel_z = command.actor.guider.set_pixel()

    return command.finish(f"Guide pixel is now ({pixel_x:.2f}, {pixel_z:.2f}).")
