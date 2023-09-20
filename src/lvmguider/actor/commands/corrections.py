#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-09-20
# @Filename: corrections.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from lvmguider.actor import lvmguider_parser


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand


__all__ = ["corrections"]


@lvmguider_parser.command()
@click.argument("MODE", type=click.Choice(["enable", "disable"]))
async def corrections(command: GuiderCommand, mode: str):
    """Enables/disables corrections during guiding."""

    if not command.actor.guider:
        return command.fail("Guider is not running.")

    if mode.lower() == "enable":
        command.actor.guider.apply_corrections = True
    elif mode.lower() == "disable":
        command.actor.guider.apply_corrections = False
    else:
        return command.fail(f"Invalid mode {mode!r}.")

    return command.finish()
