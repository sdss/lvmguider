#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-19
# @Filename: guider.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio

from typing import TYPE_CHECKING, Any

import numpy
import pandas
from astropy.io import fits
from astropy.table import Table
from simple_pid import PID

from lvmguider.maskbits import GuiderStatus

from .tools import get_proc_path, run_in_executor
from .transformations import (
    XZ_FULL_FRAME,
    delta_radec2mot_axis,
    solve_from_files,
    wcs_from_single_cameras,
)


if TYPE_CHECKING:
    from astropy.wcs import WCS

    from lvmguider.actor import GuiderCommand
    from lvmguider.astrometrynet import AstrometrySolution


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

    async def guide_one(
        self,
        exposure_time: float = 5.0,
        apply_correction: bool = True,
        use_motor_offsets: bool = True,
        use_individual_images: bool = False,
    ):
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
                use_individual_images=use_individual_images,
            )
        except Exception as err:
            raise RuntimeError(f"Failed determining telescope pointing: {err}")

        # Offset is field centre - current pointing.
        offset_radec, offset_motax, sep = self.calculate_telescope_offset((ra_p, dec_p))

        self.command.info(
            measured_pointing={
                "ra": ra_p,
                "dec": dec_p,
                "offset": list(offset_radec),
                "separation": sep,
            }
        )

        # Calculate the correction.
        corr_radec = numpy.array(offset_radec)  # In RA/Dec
        corr_radec[0] = self.pid_ra(corr_radec[0])
        corr_radec[1] = self.pid_dec(corr_radec[1])

        corr_motax = numpy.array(offset_motax)  # In motor axes
        corr_motax[0] = self.pid_ra(corr_motax[0])
        corr_motax[1] = self.pid_dec(corr_motax[1])

        applied_radec = applied_motax = numpy.array([0.0, 0.0])
        try:
            if apply_correction:
                self.command.actor.status &= ~GuiderStatus.PROCESSING
                self.command.actor.status |= GuiderStatus.CORRECTING

                applied_radec, applied_motax = await self.offset_telescope(
                    *offset_motax,
                    use_motor_axes=use_motor_offsets,
                    max_correction=self.config["max_correction"],
                )

        finally:
            self.command.actor.status &= ~GuiderStatus.CORRECTING

            self.command.info(
                correction_applied={
                    "radec_applied": list(numpy.round(applied_radec, 3)),
                    "motax_applied": list(numpy.round(applied_motax, 3)),
                }
            )

            asyncio.create_task(
                self.write_proc_file(
                    filenames=filenames,
                    wcs=wcs,
                    ra_p=ra_p,
                    dec_p=dec_p,
                    offset_radec=tuple(offset_radec),
                    corr_radec=tuple(corr_radec),
                    corr_motax=tuple(corr_motax),
                    sources=list(sources),
                    is_acquisition=True,
                )
            )

    async def determine_pointing(
        self,
        filenames: list[str],
        pixel: tuple[float, float] | None = None,
        use_individual_images: bool = False,
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
        use_individual_images
            Determine the coordinates of the central pixel of the master frame
            using a WCS generated from the individual frames, instead of trying
            to solve the master frame with astrometry.net.

        Returns
        -------
        pointing
            An tuple with the RA,Dec pointing of the telescope, in degrees.

        """

        telescope = self.telescope

        pixel = pixel or XZ_FULL_FRAME

        if not use_individual_images:
            solution: AstrometrySolution
            solution, _ = await run_in_executor(solve_from_files, filenames, telescope)
            wcs = solution.wcs
        else:
            wcs, solutions = await run_in_executor(
                wcs_from_single_cameras,
                filenames,
                telescope=telescope,
            )

            for camname in solutions:
                if solutions[camname].solved is False:
                    self.command.warning(f"Camera {camname} did not solve.")

        if wcs is None:
            raise ValueError(f"Cannot determine pointing for telescope {telescope}.")

        pointing: Any = wcs.pixel_to_world(*pixel)

        return (numpy.round(pointing.ra.deg, 6), numpy.round(pointing.dec.deg, 6), wcs)

    def calculate_telescope_offset(
        self,
        pointing: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float], float]:
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

        saz_diff_d, sel_diff_d = delta_radec2mot_axis(fra, fdec, pra, pdec)

        return ((ra_arcsec, dec_arcsec), (saz_diff_d, sel_diff_d), sep_arcsec)

    async def offset_telescope(
        self,
        off0: float,
        off1: float,
        use_motor_axes: bool = False,
        max_correction: float | None = None,
    ):
        """Sends a correction offset to the telescope.

        Parameters
        ----------
        off0
            Offset in the first axis, in arcsec.
        off1
            Offset in the second axis, in arcsec.
        use_motor_axes
            Whether to apply the corrections as motor axes offsets.
        max_correction
            Maximum allowed correction. If any of the axes corrections is larger
            than this value an exception will be raised.

        Returns
        -------
        applied_radec
            The applied correction in RA and Dec in arcsec. If the correction
            is actually applied as motor offsets, this value is estimated from
            them.
        applied_motax
            The applied correction in motor axes in arcsec. Zero if the correction
            is applied as RA/Dec.

        """

        if numpy.any(numpy.abs([off0, off1]) > max_correction):
            raise ValueError("Requested correction is too big. Not applying it.")

        telescope = self.command.actor.telescope
        pwi = f"lvm.{telescope}.pwi"

        applied_radec = numpy.array([0.0, 0.0])
        applied_motax = numpy.array([0.0, 0.0])

        if use_motor_axes is False:
            cmd_str = f"offset --ra_add_arcsec {off0} --dec_add_arcsec {off1}"
            applied_radec = numpy.array([off0, off1])
        else:
            cmd_str = f"offset --axis0_add_arcsec {off0} --axis1_add_arcsec {off1}"
            applied_motax = numpy.array([off0, off1])
            # TODO: calculate applied_radec here. Maybe Tom has the
            # delta_mot_axis2radec equivalent, if not it should not be hard
            # to derive.

        cmd = await self.command.send_command(pwi, cmd_str)

        if cmd.status.did_fail:
            raise RuntimeError(f"Failed offsetting telescope {telescope}.")

        return applied_radec, applied_motax

    async def write_proc_file(
        self,
        filenames: list[str],
        wcs: WCS,
        ra_p: float,
        dec_p: float,
        offset_radec: tuple[float, float],
        corr_radec: tuple[float, float],
        corr_motax: tuple[float, float],
        sources: list[pandas.DataFrame],
        is_acquisition=True,
    ):
        """Writes a ``proc-`` image with the astrometric info and applied corrections.

        Parameters
        ----------
        filenames
            A list of the paths of the individual AG frames used for the solution.
        wcs
            The master frame WCS. If ``is_acquisition=False``, this is the WCS of
            the master frame in the reference image.
        ra_p
            RA of the pointing
        dec_p
            Dec of the pointing.
        offset_radec
            Measured offset, in arcsec, in the RA/Dec direction.
        offset_motax
            Measured offset, in arcsec, in the motor axes direction.
        corr_radec
            Correction applied, in arcsec, in the RA/Dec direction.
        corr_motax
            Correction applied, in arcsec, in the motor axes direction.
        sources
            A list of the extracted sources in each AG frame.
        is_acquisition
            `True` if this was an acquisition step. `False` if guiding.

        """

        proc_path = get_proc_path(filenames[0])

        astro_hdu = fits.ImageHDU(name="ASTROMETRY")
        astro_hdr = astro_hdu.header

        astro_hdr["ACQUISIT"] = (is_acquisition, "Acquisition or guiding?")

        for fn, file_ in enumerate(filenames):
            astro_hdr[f"FILE{fn}"] = (str(file_), f"AG frame {fn}")

        astro_hdr["RAFIELD"] = (self.field_centre[0], "[deg] Field RA")
        astro_hdr["DECFIELD"] = (self.field_centre[1], "[deg] Field Dec")
        astro_hdr["RAMEAS"] = (ra_p, "[deg] Measured RA position")
        astro_hdr["DECMEAS"] = (dec_p, "[deg] Measured Dec position")
        astro_hdr["OFFRAMEA"] = (offset_radec[0], "[arcsec] RA measured offset")
        astro_hdr["OFFDEMEA"] = (offset_radec[1], "[arcsec] Dec measured offset")
        astro_hdr["RACORR"] = (corr_radec[0], "[arcsec] RA applied correction")
        astro_hdr["DECORR"] = (corr_radec[1], "[arcsec] Dec applied correction")
        astro_hdr["AX0CORR"] = (corr_motax[0], "[arcsec] Motor axis 0 applied offset")
        astro_hdr["AX1CORR"] = (corr_motax[1], "[arcsec] Motor axis 1 applied offset")
        astro_hdr["RAKP"] = (self.pid_ra.Kp, "RA PID K term")
        astro_hdr["RAKI"] = (self.pid_ra.Ki, "RA PID I term")
        astro_hdr["RAKD"] = (self.pid_ra.Kd, "RA PID D term")
        astro_hdr["DECKP"] = (self.pid_dec.Kp, "Dec PID K term")
        astro_hdr["DECKI"] = (self.pid_dec.Ki, "Dec PID I term")
        astro_hdr["DECKD"] = (self.pid_dec.Kd, "Dec PID D term")
        astro_hdr += wcs.to_header()

        proc_hdu = fits.HDUList([fits.PrimaryHDU(), astro_hdu])
        proc_hdu.append(
            fits.BinTableHDU(
                data=Table.from_pandas(pandas.concat(sources)),
                name="SOURCES",
            )
        )
        proc_hdu.writeto(str(proc_path))

        self.command.info(proc_path=str(proc_path))
        return proc_path
