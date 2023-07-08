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

import click

from lvmguider.actor import lvmguider_parser
from lvmguider.maskbits import GuiderStatus


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand


__all__ = ["stop"]


@lvmguider_parser.command()
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
