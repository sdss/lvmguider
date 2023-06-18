#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-18
# @Filename: guide.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import TYPE_CHECKING

from lvmguider.actor import lvmguider_parser
from lvmguider.maskbits import GuiderStatus


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand


__all__ = ["guide"]


@lvmguider_parser.group()
def guide():
    """Manages the guide loop."""


@guide.command()
async def start(command: GuiderCommand):
    if command.actor.status & GuiderStatus.NON_IDLE:
        return command.finish("Guider is not idle.")

    while True:
        filenames, sources = await command.actor.cameras.expose(
            command,
            extract_sources=True,
        )
        if command.actor.status & GuiderStatus.STOPPING:
            break

    command.actor.status = GuiderStatus.IDLE

    return command.finish("The guide loop has finished.")


@guide.command()
async def stop(command: GuiderCommand):
    """Stops the guide loop."""

    command.actor.status |= GuiderStatus.STOPPING
    return command.finish("Stopping the guide loop.")
