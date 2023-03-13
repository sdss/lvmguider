#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-03-13
# @Filename: focus.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand


class Focuser:
    """Performs focus on a telescope."""

    def __init__(
        self,
        command: GuiderCommand,
        guess: float | None = None,
        step_size: float = 0.5,
        steps: int = 7,
        exp_time: float = 5.0,
    ):
        self.command = command
        self.initial_guess = guess
        self.step_size = step_size
        self.steps = int(steps)
        self.exp_time = exp_time

        if self.steps % 2 == 1:
            self.steps += 1

        self.telescope = self.command.actor.telescope

        self.agcam_actor = f"lvm.{self.telescope}.agcam"
        self.foc_actor = f"lvm.{self.telescope}.foc"

    async def focus(self):
        """Performs the focus routine."""

        if self.initial_guess is None:
            cmd = await self.command.send_command(self.foc_actor, "getPosition")
            if cmd.status.did_fail:
                raise RuntimeError("Failed retrieving position from focuser.")
            self.initial_guess = cmd.replies.get("Position")
            self.command.debug(f"Focuser position: {self.initial_guess} DT")

        focus_grid = numpy.arange(
            self.initial_guess - self.steps // 2 * self.step_size,
            self.initial_guess + (self.steps // 2 + 1) * self.step_size,
            self.step_size,
        )

        if numpy.any(focus_grid <= 0):
            raise ValueError("Focus values out of range.")

        for focus_position in focus_grid:
            await self.goto_focus_position(focus_position)
            filenames = await self.expose_cameras()
            print(filenames)

    async def goto_focus_position(self, focus_position: float):
        """Moves the focuser to a position."""

        cmd = await self.command.send_command(
            self.foc_actor,
            f"moveAbsolute {focus_position} DT",
        )
        if cmd.status.did_fail:
            raise RuntimeError(f"Failed reaching focus {focus_position:.1f} DT.")
        if cmd.replies.get("AtLimit") is True:
            raise RuntimeError("Hit a limit while focusing.")

    async def expose_cameras(self):
        """Exposes the cameras and returns the filenames."""

        cmd = await self.command.send_command(
            self.agcam_actor,
            f"expose {self.exp_time}",
        )
        if cmd.status.did_fail:
            raise RuntimeError("Failed while exposing cameras.")

        filenames: set[str] = set()
        for reply in cmd.replies:
            for cam_name in ["east", "west"]:
                if cam_name in reply.message:
                    if reply.message[cam_name].get("state", None) == "written":
                        filenames.add(reply.message[cam_name]["filename"])

        if len(filenames) == 0:
            raise ValueError("Exposure did not produce any images.")

        return filenames
