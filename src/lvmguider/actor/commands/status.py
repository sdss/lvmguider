#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-07-14
# @Filename: status.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import TYPE_CHECKING

from lvmguider.actor import lvmguider_parser


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand


__all__ = ["status"]


@lvmguider_parser.command()
async def status(command: GuiderCommand):
    """Outputs the status of the guider."""

    status = command.actor.status

    return command.finish(
        message={
            "status": f"0x{status.value:x}",
            "status_labels": ",".join(status.get_names()),
        }
    )
