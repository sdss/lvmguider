#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-03-13
# @Filename: camera.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import pathlib
import re

from typing import TYPE_CHECKING

import pandas
from astropy.io import fits
from astropy.table import Table

from sdsstools.time import get_sjd

from lvmguider.extraction import extract_marginal
from lvmguider.maskbits import GuiderStatus
from lvmguider.tools import run_in_executor


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand


__all__ = ["Cameras"]


class Cameras:
    """Exposures and solves the telescope cameras."""

    def __init__(self, telescope: str):
        self.telescope = telescope
        self.agcam = f"lvm.{telescope}.agcam"

        self.sjd = get_sjd("LCO")
        self.last_seqno = -1

    async def expose(
        self,
        command: GuiderCommand,
        exposure_time: float = 5.0,
        extract_sources: bool = False,
    ) -> tuple[list[str], list[pandas.DataFrame] | None]:
        """Exposes the cameras and returns the filenames."""

        next_seqno = self.get_next_seqno()

        command.actor.status |= GuiderStatus.EXPOSING
        command.actor.status &= ~GuiderStatus.IDLE

        command.debug(f"Taking agcam exposure {self.telescope}-{next_seqno}.")
        cmd = await command.send_command(
            self.agcam,
            f"expose -n {next_seqno} {exposure_time}",
        )

        command.actor.status &= ~GuiderStatus.EXPOSING
        command.actor.status |= GuiderStatus.IDLE

        if cmd.status.did_fail:
            command.actor.status |= GuiderStatus.FAILED
            raise RuntimeError("Failed while exposing cameras.")
        else:
            self.last_seqno = next_seqno

        filenames: set[str] = set()
        for reply in cmd.replies:
            for cam_name in ["east", "west"]:
                if cam_name in reply.message:
                    if reply.message[cam_name].get("state", None) == "written":
                        filenames.add(reply.message[cam_name]["filename"])

        if len(filenames) == 0:
            raise ValueError("Exposure did not produce any images.")

        sources = []
        if extract_sources:
            try:
                sources = await asyncio.gather(
                    *[self.extract_sources(fn) for fn in filenames]
                )
            except Exception as err:
                command.warning(f"Failed extracting sources: {err}")
                extract_sources = False
            else:
                for ifn, fn in enumerate(filenames):
                    with fits.open(fn, mode="update") as hdul:
                        hdul.append(
                            fits.BinTableHDU(
                                data=Table.from_pandas(sources[ifn]),
                                name="SOURCES",
                            )
                        )

        return (list(filenames), list(sources) if extract_sources else None)

    async def extract_sources(self, filename: str):
        """Extracts sources from a file."""

        data = fits.getdata(filename)
        return await run_in_executor(extract_marginal, data, box_size=31)

    def get_next_seqno(self):
        """Determines the next exposure sequence number."""

        if get_sjd("LCO") == self.sjd and self.last_seqno >= 0:
            return self.last_seqno + 1

        self.sjd = get_sjd("LCO")

        path = pathlib.Path(f"/data/agcam/{self.sjd}")
        if not path.exists():
            self.last_seqno = 0
            return self.last_seqno + 1

        files = path.glob(f"*{self.telescope}*")
        seqnos = []
        for fn in files:
            se = re.search(r"\_([0-9]+)", str(fn))
            if not se:
                continue
            seqnos.append(int(se.group(1)))

        if len(seqnos) == 0:
            self.last_seqno = 0
        else:
            self.last_seqno = max(seqnos)

        return self.last_seqno + 1
