#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-07-08
# @Filename: stop.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
from contextlib import suppress

from typing import TYPE_CHECKING

from lvmguider.actor import lvmguider_parser
from lvmguider.maskbits import GuiderStatus


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand


__all__ = ["stop"]


@lvmguider_parser.command()
async def stop(command: GuiderCommand):
    """Stops the guide loop."""

    if command.actor.guide_task is None or command.actor.guide_task.done():
        command.actor.status = GuiderStatus.IDLE
        return command.finish("Guider is not active.")

    command.actor.guide_task.cancel()
    with suppress(asyncio.CancelledError):
        await command.actor.guide_task

    command.actor.guider = None
    command.actor.guide_task = None

    command.actor.status = GuiderStatus.IDLE

    return command.finish("Guider has been stopped.")
