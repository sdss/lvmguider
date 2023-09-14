#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-03-13
# @Filename: camera.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import os
import pathlib
import re

from typing import TYPE_CHECKING

import numpy
import pandas
from astropy.io import fits

from sdsstools.time import get_sjd

from lvmguider import __version__
from lvmguider.extraction import extract_sources as extract_sources_func
from lvmguider.maskbits import GuiderStatus
from lvmguider.tools import (
    elapsed_time,
    header_from_model,
    run_in_executor,
    update_fits,
)


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand


__all__ = ["Cameras"]


class Cameras:
    """Exposures and solves the telescope cameras."""

    def __init__(self, telescope: str):
        self.telescope = telescope

        self.agcam = f"lvm.{telescope}.agcam"

        self.pwi = f"lvm.{telescope}.pwi"
        self.km = f"lvm.{telescope}.km"
        self.foc = f"lvm.{telescope}.foc"

        self.sjd = get_sjd("LCO")
        self.last_seqno: int = -1

        self.dark_file: dict[str, str] = {}

    async def expose(
        self,
        command: GuiderCommand,
        exposure_time: float = 5.0,
        flavour: str = "object",
        extract_sources: bool = False,
        nretries: int = 3,
    ) -> tuple[list[pathlib.Path], int, list[pandas.DataFrame] | None]:
        """Exposes the cameras and returns the filenames."""

        command.actor._status &= ~GuiderStatus.IDLE
        command.actor.status |= GuiderStatus.EXPOSING

        focus_position = await self._get_focus_position(command)

        next_seqno = self.get_next_seqno()

        command.debug(f"Taking agcam exposure {self.telescope}-{next_seqno}.")
        cmd = await command.send_command(
            self.agcam,
            f"expose -n {next_seqno} --{flavour} {exposure_time}",
        )

        if cmd.status.did_fail:
            command.actor._status &= ~GuiderStatus.EXPOSING
            command.actor._status |= GuiderStatus.FAILED
            command.actor.status |= GuiderStatus.IDLE

            self.last_seqno = -1

            error = cmd.replies.get("error")

            if nretries > 0:
                command.warning(f"Failed while exposing cameras: {error} Retrying.")
                return await self.expose(
                    command,
                    exposure_time=exposure_time,
                    flavour=flavour,
                    extract_sources=extract_sources,
                    nretries=nretries - 1,
                )

            else:
                raise RuntimeError(f"Failed while exposing cameras: {error}")

        else:
            self.last_seqno = next_seqno

        filenames: set[pathlib.Path] = set()
        for reply in cmd.replies:
            if "filename" in reply.message:
                filename = reply.message["filename"]["filename"]
                cam_name = reply.message["filename"]["camera"]
                filenames.add(pathlib.Path(filename))
                if flavour == "dark":
                    self._write_dark_info(cam_name, filename)

        if len(filenames) == 0:
            raise ValueError("Exposure did not produce any images.")

        if any([self.is_shifted(filename) for filename in filenames]):
            if nretries > 0:
                command.warning("Found shifted images. Retrying.")
                return await self.expose(
                    command,
                    exposure_time=exposure_time,
                    flavour=flavour,
                    extract_sources=extract_sources,
                    nretries=nretries - 1,
                )
            else:
                raise RuntimeError("Run out of retries. Exposing failed.")

        headers: dict[pathlib.Path, fits.Header] = {}

        # Create a new extension with processed data.
        for fn in filenames:
            camname = fits.getval(str(fn), "CAMNAME", "RAW")

            dark_file = self._get_dark_frame(fn, camname)
            if dark_file is None:
                command.warning(f"No dark frame found for camera {camname}.")
                dark_file = ""

            header = header_from_model("PROC")
            header["TELESCOP"] = self.telescope
            header["CAMNAME"] = camname
            header["DARKFILE"] = os.path.basename(dark_file)
            header["DIRNAME"] = str(fn.absolute().parent)
            header["GUIDERV"] = __version__
            header["WCSMODE"] = "none"

            headers[fn] = header

        sources: list[pandas.DataFrame] = []
        if flavour == "object" and extract_sources:
            try:
                sources = await asyncio.gather(
                    *[run_in_executor(extract_sources_func, fn) for fn in filenames]
                )
            except Exception as err:
                command.warning(f"Failed extracting sources: {err}")
                extract_sources = False
            else:
                for ifn, fn in enumerate(filenames):
                    sources_path = fn.with_suffix(".parquet")
                    sources[ifn].to_parquet(sources_path)

                    headers[fn]["SOURCESF"] = sources_path.name

        if len(sources) > 0:
            all_sources = pandas.concat(sources)
            valid = all_sources.loc[all_sources.valid == 1]
            fwhm = numpy.percentile(valid["fwhm"], 25) if len(valid) > 0 else None
        else:
            valid = []
            fwhm = None

        command.info(
            frame={
                "seqno": next_seqno,
                "filenames": [str(fn) for fn in filenames],
                "flavour": flavour,
                "n_sources": len(valid),
                "focus_position": round(focus_position, 1),
                "fwhm": numpy.round(float(fwhm), 3) if fwhm else -999.0,
            }
        )

        command.actor._status &= ~GuiderStatus.FAILED
        command.actor.status &= ~GuiderStatus.EXPOSING

        if not command.actor.status & GuiderStatus.NON_IDLE:
            command.actor.status |= GuiderStatus.IDLE

        with elapsed_time(command, "updating lvm.agcam file"):
            await asyncio.gather(
                *[
                    run_in_executor(update_fits, fn, "PROC", header=headers[fn])
                    for fn in filenames
                ]
            )

        return (list(filenames), next_seqno, list(sources) if extract_sources else None)

    def reset_seqno(self):
        """Resets the seqno.

        This forces a full check of the files next time that
        `.get_next_seqno` is called.

        """

        self.last_seqno = -1

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

    def is_shifted(self, filename: str | pathlib.Path | numpy.ndarray):
        """Determines if an image is "shifted"."""

        data: numpy.ndarray
        if isinstance(filename, (str, pathlib.Path)):
            data = fits.getdata(str(filename))
        else:
            data = filename

        # If more than 10% of the pixels have values >32,000 this is probably
        # a shifted image.
        npix = data.size
        if (data > 32000).sum() / npix > 0.1:
            return True

        # Another case of a bad image.
        if data.sum() == 0:
            return True

        return False

    async def _get_focus_position(self, command: GuiderCommand):
        """Retrieves the focuser position,"""

        try:
            focus_position = command.actor.models[self.foc]["Position"].value
            assert focus_position is not None
        except Exception:
            command.warning("Focus position not found in model. Querying the actor.")
        else:
            return focus_position

        try:
            focus_cmd = await command.send_command(self.foc, "status", internal=True)
            focus_position = focus_cmd.replies.get("Position")
            assert focus_position is not None
        except Exception as err:
            command.warning(f"Failed getting {self.telescope} focuser position: {err}")
            return -999

        return focus_position

    def _get_dark_frame(self, filename: str | pathlib.Path, cam_name: str):
        """Gets the path to the dark frame."""

        filename = pathlib.Path(filename).absolute()

        dark_file = self.dark_file.get(cam_name, None)
        if dark_file is not None:
            return dark_file

        dirname = filename.parent
        path = dirname / f"lvm.{self.telescope}.agcam.{cam_name}.dark.dat"

        if path.exists():
            dark_file = open(path, "r").read().strip()
            self.dark_file[cam_name] = dark_file
            return pathlib.Path(dark_file)

        return None

    def _write_dark_info(self, cam_name: str, filename: str):
        """Writes a file with the current dark frame."""

        self.dark_file[cam_name] = filename

        dirname = os.path.dirname(filename)
        path = os.path.join(dirname, f"lvm.{self.telescope}.agcam.{cam_name}.dark.dat")
        with open(path, "w") as fd:
            print(str(filename), file=fd)
