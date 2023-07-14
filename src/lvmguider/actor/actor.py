#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-03-10
# @Filename: actor.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio

from typing import TYPE_CHECKING

from clu.actor import AMQPActor

from lvmguider.cameras import Cameras
from lvmguider.maskbits import GuiderStatus


if TYPE_CHECKING:
    from lvmguider.guider import Guider

__all__ = ["LVMGuiderActor"]


class LVMGuiderActor(AMQPActor):
    """The ``lvmguider`` actor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.model and self.model.schema:
            self.model.schema["additionalProperties"] = True

        self.telescope: str = self.config.get("telescope", self.name.split(".")[1])
        self.cameras = Cameras(self.telescope)

        self._status = GuiderStatus.IDLE

        self.guider: Guider | None = None
        self.guide_task: asyncio.Task | None = None

    @property
    def status(self):
        """Returns the guider status."""

        return self._status

    @status.setter
    def status(self, new_value: GuiderStatus):
        """Sets a new status and reports it to the users."""

        if new_value.value != self._status.value:
            self._status = new_value
            self.write(
                "d",
                {
                    "status": f"0x{self._status.value:x}",
                    "status_labels": ",".join(self._status.get_names()),
                },
                internal=True,
            )
