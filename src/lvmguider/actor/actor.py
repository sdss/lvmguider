#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-03-10
# @Filename: actor.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from clu.actor import AMQPActor

from lvmguider.cameras import Cameras
from lvmguider.maskbits import GuiderStatus


__all__ = ["LVMGuiderActor"]


class LVMGuiderActor(AMQPActor):
    """The ``lvmguider`` actor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.model and self.model.schema:
            self.model.schema["additionalProperties"] = True

        self.telescope: str = self.config.get("telescope", self.name.split(".")[1])
        self.cameras = Cameras(self.telescope)

        self.status = GuiderStatus.IDLE
