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

from lvmguider import __version__, config, log
from lvmguider.cameras import Cameras
from lvmguider.maskbits import GuiderStatus


if TYPE_CHECKING:
    from clu.model import Property

    from lvmguider.guider import Guider


__all__ = ["LVMGuiderActor"]


class LVMGuiderActor(AMQPActor):
    """The ``lvmguider`` actor."""

    def __init__(self, *args, **kwargs):
        # The package imports a generic config file without actor or telescope info.
        # We expect the actor is initialised with it.
        aconfig = kwargs.get("config", None)
        if aconfig is None:
            raise RuntimeError("Actor must be initialised from a configuration file.")

        # Check that there's an actor section.
        if "actor" not in aconfig:
            raise RuntimeError("The configuration file does not have an actor section.")

        # Update package config.
        config._BASE = dict(config)
        config.update(aconfig)

        name = aconfig["actor"]["name"]
        self.telescope: str = aconfig.get("telescope", name.split(".")[1])

        # Update the config that will be set in the actor instance.
        kwargs["config"] = dict(config)

        super().__init__(*args, log=log, version=__version__, **kwargs)

        if self.model and self.model.schema:
            self.model.schema["additionalProperties"] = True

        self.cameras = Cameras(self.telescope)

        self._status = GuiderStatus.IDLE

        self.guider: Guider | None = None
        self.guide_task: asyncio.Task | None = None

        # Track model of the cameras and focuser associated to this telescope.
        self.models.actors.append(f"lvm.{self.telescope}.foc")

        agcam = f"lvm.{self.telescope}.agcam"
        self._agcam_state: dict[str, str] = {"east": "unknown", "west": "unknown"}
        self.models.actors.append(agcam)
        self.models[agcam]["exposure_state"].register_callback(self.track_agcam_state)
        self.models[agcam]["status"].register_callback(self.track_agcam_state)

    async def start(self, **kwargs):
        """Starts the actor."""

        await super().start(**kwargs)

        # Poll the cameras to update the exposure state.
        await self.send_command(f"lvm.{self.telescope}.agcam", "status")

        return self

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

    def agcam_is_idle(self):
        """Returns `True` if the cameras are idle."""

        if self.telescope == "spec":
            return self._agcam_state["east"] == "idle"

        return all([val == "idle" for val in self._agcam_state.values()])

    async def track_agcam_state(self, prop: Property):
        """Updates the expose state of the cameras."""

        if prop.name not in ["status", "exposure_state"]:
            return

        camera = prop.value["camera"]

        if prop.name == "status":
            self._agcam_state[camera] = prop.value["camera_state"]
        else:
            self._agcam_state[camera] = prop.value["state"]

        print(self._agcam_state)
