#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-19
# @Filename: guider.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os.path

from typing import TYPE_CHECKING, Any

import numpy
import pandas
from astropy.io import fits
from astropy.table import Table
from simple_pid import PID

from lvmguider.maskbits import GuiderStatus

from .tools import run_in_executor
from .transformations import XZ_FULL_FRAME, solve_from_files


if TYPE_CHECKING:
    from astropy.wcs import WCS

    from lvmguider.actor import GuiderCommand


class Guider:
    """Acquisition and guiding sequence.

    Parameters
    ----------
    command
        The actor command used to communicate with the actor system.
    field_centre
        The field centre on which we want to guider.
    pixel
        The ``(x,y)`` pixel of the master frame to use to determine the pointing.
        Default to the central pixel.

    """

    def __init__(
        self,
        command: GuiderCommand,
        field_centre: tuple[float, float],
        pixel: tuple[float, float] | None = None,
    ):
        self.command = command
        self.telescope = command.actor.telescope
        self.cameras = command.actor.cameras

        self.field_centre = field_centre
        self.pixel = pixel

        self.config = command.actor.config

        # Use negative values because we apply the correction in the
        # same direction as the measured offset between pointing and
        # reference position.
        self.pid_ra = PID(
            Kp=-self.config["pid"]["ra"]["Kp"],
            Ki=-self.config["pid"]["ra"]["Ki"],
            Kd=-self.config["pid"]["ra"]["Kd"],
        )

        self.pid_dec = PID(
            Kp=-self.config["pid"]["dec"]["Kp"],
            Ki=-self.config["pid"]["dec"]["Ki"],
            Kd=-self.config["pid"]["dec"]["Kd"],
        )

    async def guide_one(self, exposure_time: float = 5.0):
        """Performs one guide iteration.

        In order, it does the following:

        - Expose the cameras. Subtract dark frame.
        - Extract sources.
        - Create a master frame based on the camera(s)-to-IFU metrology.
        - Solve the frame using astrometry.net.
        - Determine the offsets for a given pixel on the master frame. By default
          the central pixel is used, which corresponds to guiding at the central
          fibre in the IFU, but a different location can be used to guide, for example,
          on one of the spec mask holes.
        - Send PID-corrected offsets to the telescopes.

        """

        filenames, sources = await self.cameras.expose(
            self.command,
            extract_sources=True,
            exposure_time=exposure_time,
        )
        if sources is None:
            raise RuntimeError("No sources found.")

        self.command.actor.status |= GuiderStatus.PROCESSING
        self.command.actor.status &= ~GuiderStatus.IDLE

        try:
            ra_p, dec_p, wcs = await self.determine_pointing(
                filenames,
                pixel=self.pixel,
            )
        except Exception as err:
            raise RuntimeError(f"Failed determining telescope pointing: {err}")

        self.command.info(
            measured_pointing={
                "ra": ra_p,
                "dec": dec_p,
            }
        )

        # Offset is field centre - current pointing.
        offset, sep = self.calculate_telescope_offset((ra_p, dec_p))

        # Calculate the correction.
        offset = numpy.array(offset)

        corr = offset.copy()
        corr[0] = self.pid_ra(corr[0])
        corr[1] = self.pid_dec(corr[1])
        corr = numpy.round(corr, 3)

        self.command.actor.status &= ~GuiderStatus.PROCESSING
        self.command.actor.status |= GuiderStatus.CORRECTING

        try:
            if numpy.any(numpy.abs(corr) > 500):
                raise ValueError(
                    "Correction is too big. Maybe an issue with the "
                    "astrometric solution?"
                )

            await self.offset_telescope(*corr)
        except Exception:
            corr = numpy.array([0.0, 0.0])
            raise
        finally:
            self.command.actor.status &= ~GuiderStatus.CORRECTING

            self.command.info(
                pointing_correction={
                    "separation": sep,
                    "offset_measured": list(offset),
                    "offset_applied": list(corr),
                }
            )

            # Create new proc- image with astrometric solution and
            # measured and applied corrections.
            proc_path = get_proc_path(filenames[0])

            astro_hdu = fits.ImageHDU(name="ASTROMETRY")
            astro_hdu.header["RAFIELD"] = (self.field_centre[0], "[deg] Field RA")
            astro_hdu.header["DECFIELD"] = (self.field_centre[1], "[deg] Field Dec")
            astro_hdu.header["RAMEAS"] = (ra_p, "[deg] Measured RA position")
            astro_hdu.header["DECMEAS"] = (dec_p, "[deg] Measured Dec position")
            astro_hdu.header["OFFRAMEA"] = (offset[0], "[arcsec] RA measured offset")
            astro_hdu.header["OFFDEMEA"] = (offset[1], "[arcsec] Dec measured offset")
            astro_hdu.header["OFFRACOR"] = (corr[0], "[arcsec] RA applied correction")
            astro_hdu.header["OFFDECOR"] = (corr[1], "[arcsec] Dec applied correction")
            astro_hdu.header["RAKP"] = (self.pid_ra.Kp, "RA PID K term")
            astro_hdu.header["RAKI"] = (self.pid_ra.Ki, "RA PID I term")
            astro_hdu.header["RAKD"] = (self.pid_ra.Kd, "RA PID D term")
            astro_hdu.header["DECKP"] = (self.pid_dec.Kp, "Dec PID K term")
            astro_hdu.header["DECKI"] = (self.pid_dec.Ki, "Dec PID I term")
            astro_hdu.header["DECKD"] = (self.pid_dec.Kd, "Dec PID D term")
            astro_hdu.header += wcs.to_header()

            proc_hdu = fits.HDUList([fits.PrimaryHDU(), astro_hdu])
            proc_hdu.append(
                fits.BinTableHDU(
                    data=Table.from_pandas(pandas.concat(sources)),
                    name="SOURCES",
                )
            )
            proc_hdu.writeto(str(proc_path))

    async def determine_pointing(
        self,
        filenames: list[str],
        pixel: tuple[float, float] | None = None,
    ) -> tuple[float, float, WCS]:
        """Returns the pointing of a telescope based on AG frames.

        Parameters
        ----------
        filenames
            A list of two AG frames (one for ``spec``) to be used to determine the
            pointing by calling `.solve_from_files`.
        pixel
            The ``(x,y)`` pixel of the master frame to use to determine the pointing.
            Default to the central pixel.

        Returns
        -------
        pointing
            An tuple with the RA,Dec pointing of the telescope, in degrees.

        """

        telescope = self.telescope

        pixel = pixel or XZ_FULL_FRAME
        wcs: WCS | None = await run_in_executor(solve_from_files, filenames, telescope)

        if wcs is None:
            raise ValueError(f"Cannot determine pointing for telescope {telescope}.")

        pointing: Any = wcs.pixel_to_world(*pixel)

        return (numpy.round(pointing.ra.deg, 6), numpy.round(pointing.dec.deg, 6), wcs)

    def calculate_telescope_offset(
        self,
        pointing: tuple[float, float],
    ) -> tuple[tuple[float, float], float]:
        """Determines the offset to send to the telescope to acquire the field centre.

        Parameters
        ----------
        pointing
            The current pointing of the telescope, as determined
            by `.determine_pointing`.

        Returns
        -------
        offset
            A tuple of ra/dec offsets to acquire the desired field centre, in arcsec.
        angle
            The angle between pointing and field centre, in arcsec.

        """

        pra, pdec = pointing
        fra, fdec = self.field_centre

        # TODO: do this with proper spherical trigonometry! But won't matter much
        # if angle is small.

        mid_dec = (pdec + fdec) / 2

        ra_off: float = (fra - pra) * numpy.cos(numpy.radians(mid_dec))
        dec_off: float = fdec - pdec

        fdec_c = numpy.radians(90 - fdec)
        pdec_c = numpy.radians(90 - pdec)
        ra_diff_rad = numpy.radians(fra - pra)

        cos_dec = numpy.cos(fdec_c) * numpy.cos(pdec_c)
        sin_dec_cos_ra = numpy.sin(fdec_c) * numpy.sin(pdec_c) * numpy.cos(ra_diff_rad)
        cos_sep = cos_dec + sin_dec_cos_ra
        sep = numpy.degrees(numpy.arccos(cos_sep))

        ra_arcsec = numpy.round(ra_off * 3600, 3)
        dec_arcsec = numpy.round(dec_off * 3600, 3)
        sep_arcsec = numpy.round(sep * 3600, 3)

        return ((ra_arcsec, dec_arcsec), sep_arcsec)

    async def offset_telescope(self, ra_off: float, dec_off: float):
        """Sends a correction offset to the telescope.

        Parameters
        ----------
        ra_off
            The RA offset, in arcsec.
        dec_off
            The Dec offset, in arcsec.

        """

        telescope = self.command.actor.telescope
        pwi = f"lvm.{telescope}.pwi"

        cmd = await self.command.send_command(
            pwi,
            f"offset --ra_add_arcsec {ra_off} --dec_add_arcsec {dec_off}",
        )

        if cmd.status.did_fail:
            raise RuntimeError(f"Failed offsetting telescope {telescope}.")

        return True
