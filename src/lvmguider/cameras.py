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
import warnings

from typing import TYPE_CHECKING

import numpy
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

        self.pwi = f"lvm.{telescope}.pwi"
        self.km = f"lvm.{telescope}.km"
        self.foc = f"lvm.{telescope}.foc"

        self.sjd = get_sjd("LCO")
        self.last_seqno = -1

        self.dark_file: dict[str, str] = {}

    async def expose(
        self,
        command: GuiderCommand,
        exposure_time: float = 5.0,
        flavour: str = "object",
        extract_sources: bool = False,
        nretries: int = 3,
    ) -> tuple[list[str], int, list[pandas.DataFrame] | None]:
        """Exposes the cameras and returns the filenames."""

        command.actor._status &= ~GuiderStatus.IDLE
        command.actor.status |= GuiderStatus.EXPOSING

        # Update the status of the telescope and get the focuser position.
        await command.send_command(self.pwi, "status", internal=True)

        try:
            focus_cmd = await command.send_command(self.foc, "status", internal=True)
            focus_position = focus_cmd.replies.get("Position")
        except Exception as err:
            warnings.warn(f"Failed getting {self.telescope} focuser position: {err}")
            focus_position = -999

        if self.telescope != "spec":
            await command.send_command(self.pwi, "status", internal=True)

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
            raise RuntimeError("Failed while exposing cameras.")
        else:
            self.last_seqno = next_seqno

        filenames: set[str] = set()
        for reply in cmd.replies:
            for cam_name in ["east", "west"]:
                if cam_name in reply.message:
                    if reply.message[cam_name].get("state", None) == "written":
                        filename = reply.message[cam_name]["filename"]
                        filenames.add(filename)
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

        # Create a new extension with the dark-subtracted image.
        for fn in filenames:
            with fits.open(fn, mode="update") as hdul:
                data = hdul[0].data.astype(numpy.float32)
                exptime = hdul[0].header["EXPTIME"]
                camname = hdul[0].header["CAMNAME"].lower()

                dark_file = self._get_dark_frame(fn, camname)
                if dark_file is None:
                    command.warning(f"No dark frame found for camera {camname}.")
                    continue

                dark_data = fits.getdata(dark_file).astype(numpy.float32)
                dark_exptime = fits.getheader(dark_file)["EXPTIME"]

                data_sub = data - (dark_data / dark_exptime) * exptime

                proc_header = hdul[0].header.copy()
                proc_header["DARKFILE"] = dark_file
                hdul.append(
                    fits.ImageHDU(
                        data=data_sub,
                        header=proc_header,
                        name="PROC",
                    )
                )

        sources = []
        if flavour == "object" and extract_sources:
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
                        camname = hdul[0].header["CAMNAME"].lower()
                        isources = sources[ifn]
                        isources["camera"] = camname
                        hdul.append(
                            fits.BinTableHDU(
                                data=Table.from_pandas(isources),
                                name="SOURCES",
                            )
                        )

        if len(sources) > 0:
            all_sources = pandas.concat(sources)
            fwhm = numpy.median(all_sources["xstd"]) if len(all_sources) > 0 else None
        else:
            all_sources = []
            fwhm = None

        command.info(
            frame={
                "seqno": next_seqno,
                "filenames": list(filenames),
                "flavour": flavour,
                "n_sources": len(all_sources),
                "focus_position": round(focus_position, 1),
                "fwhm": numpy.round(fwhm, 3) if fwhm else fwhm,
            }
        )

        command.actor.status &= ~GuiderStatus.EXPOSING

        if not command.actor.status & GuiderStatus.NON_IDLE:
            command.actor.status |= GuiderStatus.IDLE

        return (list(filenames), next_seqno, list(sources) if extract_sources else None)

    async def extract_sources(self, filename: str):
        """Extracts sources from a file."""

        hdus = fits.open(filename)
        if "PROC" in hdus:
            data = hdus["PROC"].data
        else:
            data = hdus[0].data

        return await run_in_executor(
            extract_marginal,
            data,
            box_size=31,
            threshold=3.0,
            max_detections=50,
            sextractor_quick_options={"minarea": 5},
        )

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

    def _get_dark_frame(self, filename: str, cam_name: str):
        """Gets the path to the dark frame."""

        dark_file = self.dark_file.get(cam_name, None)
        if dark_file is not None:
            return dark_file

        dirname = os.path.dirname(filename)
        path = os.path.join(dirname, f"lvm.{self.telescope}.agcam.{cam_name}.dark.dat")

        if os.path.exists(path):
            dark_file = open(path, "r").read().strip()
            self.dark_file[cam_name] = dark_file
            return dark_file

        return None

    def _write_dark_info(self, cam_name: str, filename: str):
        """Writes a file with the current dark frame."""

        self.dark_file[cam_name] = filename

        dirname = os.path.dirname(filename)
        path = os.path.join(dirname, f"lvm.{self.telescope}.agcam.{cam_name}.dark.dat")
        with open(path, "w") as fd:
            print(str(filename), file=fd)
