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
from sdsstools import get_logger

from lvmguider.cameras import Cameras
from lvmguider.maskbits import GuiderStatus


if TYPE_CHECKING:
    from lvmguider.guider import Guider

__all__ = ["LVMGuiderActor"]


class LVMGuiderActor(AMQPActor):
    """The ``lvmguider`` actor."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config", None)
        if config is None:
            raise RuntimeError("Actor must be initialised from a configuration file.")

        name = config["actor"]["name"]
        self.telescope: str = kwargs["config"].get("telescope", name.split(".")[1])

        # Use rich handler instead of the currently default CLU logger. Just nicer.
        log = get_logger(f"clu.lvmguider.{self.telescope}", use_rich_handler=True)

        super().__init__(*args, log=log, **kwargs)

        if self.model and self.model.schema:
            self.model.schema["additionalProperties"] = True

        self.cameras = Cameras(self.telescope)

        self._status = GuiderStatus.IDLE

        self.guider: Guider | None = None
        self.guide_task: asyncio.Task | None = None

        # Track model of focuser associated to this telescope.
        self.models.actors.append(f"lvm.{self.telescope}.foc")

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
