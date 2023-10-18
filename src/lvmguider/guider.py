#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-19
# @Filename: guider.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import pathlib

from typing import TYPE_CHECKING

import numpy
import pandas
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.time import Time
from simple_pid import PID

from sdsstools.time import get_sjd

from lvmguider import __version__
from lvmguider.dataclasses import CameraSolution, GuiderSolution
from lvmguider.maskbits import GuiderStatus
from lvmguider.tools import (
    elapsed_time,
    estimate_zeropoint,
    get_dark_subtracted_data,
    get_frameno,
    get_guider_path,
    header_from_model,
    nan_or_none,
    run_in_executor,
    update_fits,
)
from lvmguider.transformations import (
    delta_radec2mot_axis,
    match_with_gaia,
    solve_camera_with_astrometrynet,
    wcs_from_gaia,
)


if TYPE_CHECKING:
    from astropy.wcs import WCS

    from lvmguider.actor import GuiderCommand
    from lvmguider.astrometrynet import AstrometrySolution


__all__ = ["Guider", "CriticalGuiderError"]


class CriticalGuiderError(Exception):
    """An exception that should stop the guide loop."""

    pass


class Guider:
    """Acquisition and guiding sequence.

    Parameters
    ----------
    command
        The actor command used to communicate with the actor system.
    field_centre
        The field centre on which we want to guider (RA, Dec, PA).
    pixel
        The ``(x,y)`` pixel of the full frame to use to determine the pointing.
        Default to the central pixel.
    apply_corrections
        Whether to apply the measured corrections. If `False`, outputs the
        measurements but does not command the telescope.

    """

    def __init__(
        self,
        command: GuiderCommand,
        field_centre: tuple[float, float, float],
        pixel: tuple[float, float] | None = None,
        apply_corrections: bool = True,
    ):
        self.command = command
        self.telescope = command.actor.telescope
        self.cameras = command.actor.cameras

        self.config = command.actor.config
        self.guide_tolerance = self.config["guide_tolerance"]
        self.pa_tolerance = self.config["pa_tolerance"]
        self.apply_corrections: bool = apply_corrections

        self.field_centre = field_centre
        self.pixel = pixel or self.config["xz_full_frame"]

        # Use negative values because we apply the correction in the
        # same direction as the measured offset between pointing and
        # reference position.
        self.pid_ax0 = PID(
            Kp=-self.config["pid"]["ax0"]["Kp"],
            Ki=-self.config["pid"]["ax0"]["Ki"],
            Kd=-self.config["pid"]["ax0"]["Kd"],
        )

        self.pid_ax1 = PID(
            Kp=-self.config["pid"]["ax1"]["Kp"],
            Ki=-self.config["pid"]["ax1"]["Ki"],
            Kd=-self.config["pid"]["ax1"]["Kd"],
        )

        self.pid_rot = PID(
            Kp=-self.config["pid"]["rot"]["Kp"],
            Ki=-self.config["pid"]["rot"]["Ki"],
            Kd=-self.config["pid"]["rot"]["Kd"],
        )

        self.site = EarthLocation(**self.config["site"])

        self.solutions: dict[int, GuiderSolution] = {}

    @property
    def last(self):
        """Returns the last solution."""

        if len(self.solutions) == 0:
            return None

        return self.solutions[max(self.solutions)]

    def set_pixel(self, pixel_x: float | None = None, pixel_z: float | None = None):
        """Sets the full frame pixel coordinates ``(x, z)`` on which to guide."""

        if pixel_x is None or pixel_z is None:
            new_pixel = self.config["xz_full_frame"]
        else:
            new_pixel = (pixel_x, pixel_z)

        if self.pixel != new_pixel:
            self.revert_to_acquisition()
            self.pixel = new_pixel

        return self.pixel

    def is_guiding(self):
        """Are we guiding?"""

        return bool(self.command.actor.status & GuiderStatus.GUIDING)

    def revert_to_acquisition(self):
        """Reset the flags for acquisition mode."""

        if self.is_guiding():
            self.command.warning("Reverting to acquisition.")

        self.command.actor._status &= ~GuiderStatus.GUIDING
        self.command.actor._status |= GuiderStatus.ACQUIRING
        self.command.actor.status |= GuiderStatus.DRIFTING

    async def guide_one(
        self,
        exposure_time: float = 5.0,
        force_astrometry_net: bool = False,
        guide_tolerance: float | None = None,
    ):
        """Performs one guide iteration.

        This is the main routine that executes one acquisition or guide iteration.
        Two algorithms are implemented to determine astrometric solution of each
        camera. Initially ``astrometry.net`` is used to produce a WCS based on the
        detected sources. Once the global position of the telescope is within
        ``guide_tolerance``, the last guider solution is used as a reference and
        new frames are matched using a nearest neighbours k-d tree.

        The list of steps include:

        - Expose the cameras. Subtract dark frame.
        - Extract sources.
        - Solve each camera frame using astrometry.net or k-d tree matching.
          Generate a WCS for the camera solution.
        - Query Gaia DR3 to select sources in the FoV of the camera position.
        - Perform aperture photometry on the camera sources and determine the
          LVM magnitude zero point for matched Gaia sources.
        - Determine the offsets in RA, Dec, and PA.
        - Send PID-corrected offsets to the telescopes.

        Parameters
        ----------
        exposure_time
            The exposure time, in seconds.
        force_astrometry_net
            When `True`, skips the k-d tree matching algorithm and always solves
            cameras using ``astrometry.net``.
        guide_tolerance
            The separation between field RA/Dec and measured pointing at which
            to consider than acquisition has been completed and guiding begins.
            If `None`, defaults to the configuration file value.

        """

        self.guide_tolerance = guide_tolerance or self.guide_tolerance

        filenames, frameno, sources = await self.cameras.expose(
            self.command,
            extract_sources=True,
            exposure_time=exposure_time,
        )
        if sources is None:
            raise RuntimeError("No sources found.")

        self.command.actor.status |= GuiderStatus.PROCESSING
        self.command.actor.status &= ~GuiderStatus.IDLE

        default_guide_tolerance: float = self.config.get("guide_tolerance", 5)
        guide_tolerance = guide_tolerance or default_guide_tolerance
        revert_to_acquistion_threshold = self.config["revert_to_acquistion_threshold"]

        # Get astrometric solutions for individual cameras.
        camera_solutions: list[CameraSolution] = await asyncio.gather(
            *[
                self.solve_camera(filename, force_astrometry_net=force_astrometry_net)
                for filename in filenames
            ]
        )

        # Initial guider solution.
        guider_solution = GuiderSolution(
            frameno=frameno,
            solutions=camera_solutions,
            telescope=self.telescope,
            guide_pixel=numpy.array(self.pixel),
            ra_field=self.field_centre[0],
            dec_field=self.field_centre[1],
            pa_field=self.field_centre[2],
            guide_mode="guide" if self.is_guiding() else "acquisition",
        )

        if guider_solution.n_cameras_solved == 0:
            self.command.error("No astrometric solutions found.")

        else:
            try:
                assert guider_solution.sources is not None
                ff_wcs = wcs_from_gaia(guider_solution.sources, ["x_ff", "z_ff"])
                guider_solution.wcs = ff_wcs

            except RuntimeError as err:
                self.command.error(f"Failed generating full frame WCS: {err}")

        offset_radec, offset_motax, offset_pa = self.calculate_offset(guider_solution)

        guider_solution.ra_off = offset_radec[0]
        guider_solution.dec_off = offset_radec[1]
        guider_solution.pa_off = offset_pa
        guider_solution.axis0_off = offset_motax[0]
        guider_solution.axis1_off = offset_motax[1]

        for camera_solution in guider_solution.solutions:
            self.command.debug(
                camera_solution={
                    "frameno": frameno,
                    "camera": camera_solution.camera,
                    "solved": camera_solution.solved,
                    "wcs_mode": camera_solution.wcs_mode,
                    "pa": numpy.round(camera_solution.pa, 4),
                    "zero_point": numpy.round(camera_solution.zero_point, 3),
                }
            )

        self.command.info(
            measured_pointing={
                "frameno": frameno,
                "ra": numpy.round(guider_solution.pointing[0], 6),
                "dec": numpy.round(guider_solution.pointing[1], 6),
                "radec_offset": list(numpy.round(offset_radec, 3)),
                "motax_offset": list(numpy.round(offset_motax, 3)),
                "separation": numpy.round(guider_solution.separation, 3),
                "pa": numpy.round(guider_solution.pa, 4),
                "pa_offset": numpy.round(offset_pa, 4),
                "zero_point": numpy.round(guider_solution.zero_point, 3),
                "mode": "guide" if self.is_guiding() else "acquisition",
            }
        )

        self.solutions[frameno] = guider_solution

        if not guider_solution.solved:
            await self.update_fits(guider_solution)
            self.revert_to_acquisition()
            return

        # Calculate the correction and apply PID loop.
        corr_motax = numpy.array(offset_motax)  # In motor axes
        corr_motax[0] = self.pid_ax0(corr_motax[0])
        corr_motax[1] = self.pid_ax1(corr_motax[1])

        corr_rot = self.pid_rot(offset_pa) or 0.0

        apply_guider_corrections = self.config["apply_guider_corrections"]
        if self.is_guiding() and not apply_guider_corrections:
            apply_correction_this = False
        else:
            apply_correction_this = self.apply_corrections

        applied_motax = numpy.array([0.0, 0.0])
        applied_rot: float = 0.0
        try:
            if apply_correction_this:
                self.command.actor.status &= ~GuiderStatus.PROCESSING
                self.command.actor.status |= GuiderStatus.CORRECTING

                mode = "acquisition" if not self.is_guiding() else "guide"
                timeout = self.config["offset"]["timeout"][mode]

                _, applied_motax, applied_rot = await self.offset_telescope(
                    corr_motax[0],
                    corr_motax[1],
                    corr_rot,
                    use_motor_axes=True,
                    timeout=timeout,
                )

                guider_solution.correction_applied = True

        finally:
            self.command.actor.status &= ~GuiderStatus.CORRECTING

            guider_solution.correction = [*applied_motax.tolist(), applied_rot]

            self.command.info(
                correction_applied={
                    "frameno": frameno,
                    "motax_applied": list(numpy.round(applied_motax, 3)),
                    "rot_applied": numpy.round(applied_rot, 4),
                }
            )

            # Are we close enough to the field centre to guide using Gaia?
            sep_reached = guider_solution.separation < self.guide_tolerance

            # Have we converged in PA? Note that we can only move in positive
            # offsets in the k-mirror, so if the offset_pa is < 0 we've gone
            # too far and we cannot easily fix it. That's why here we don't
            # take abs(offset_pa) since self.pa_tolerance is positive.
            pa_reached: bool
            if self.config["has_kmirror"] and self.config["guide_in_rot"]:
                pa_reached = numpy.isnan(offset_pa) or offset_pa < self.pa_tolerance
            else:
                pa_reached = True

            # If the separation is > certain threshold (which usually is larger
            # than the threshold for considering we are guiding in the first place)
            # then we revert to acquisition.
            revert = guider_solution.separation > revert_to_acquistion_threshold

            if sep_reached and pa_reached and not self.is_guiding():
                self.command.info("Guide tolerance reached. Starting to guide.")
                self.command.actor._status &= ~GuiderStatus.ACQUIRING
                self.command.actor._status &= ~GuiderStatus.DRIFTING
                self.command.actor.status |= GuiderStatus.GUIDING

            elif revert and self.is_guiding():
                self.command.warning("Measured offset exceeds guide tolerance.")
                self.revert_to_acquisition()

            await self.update_fits(guider_solution)

    async def solve_camera(
        self,
        file: str | pathlib.Path,
        force_astrometry_net: bool = False,
    ):
        """Astrometrically solves a single camera using astrometry.net or k-d tree."""

        file = pathlib.Path(file)
        frameno = get_frameno(file)
        hdul = fits.open(str(file))

        data, dark_sub = get_dark_subtracted_data(file)
        if not dark_sub:
            self.command.debug(f"No dark frame found for {file!s}. Fitting background.")

        camname = hdul["RAW"].header["CAMNAME"]

        sources_file = file.parent / file.with_suffix(".parquet")
        if not sources_file.exists():
            self.command.warning(f"Cannot find sources file for camera {camname!r}.")
            return CameraSolution(
                frameno=frameno,
                camera=camname,
                path=file,
                telescope=self.telescope,
            )

        sources = pandas.read_parquet(sources_file)
        matched_sources = sources.copy()

        ra: float = hdul["RAW"].header["RA"]
        dec: float = hdul["RAW"].header["DEC"]

        wcs_mode = "astrometrynet"

        # First determine the algorithm we are going to use. Check the last
        # solution for this camera and see if it was good enough to use as reference.
        if (
            self.is_guiding()
            and not force_astrometry_net
            and self.last
            and self.last.solved
            and self.last.separation < self.guide_tolerance
            and camname in self.last.cameras
            and self.last[camname].solved
        ):
            wcs_mode = "gaia"

        wcs: WCS | None = None
        matched: bool = False
        ref_frame: pathlib.Path | None = None

        if wcs_mode == "astrometrynet":
            # Basename path for the astrometry.net outputs.
            basename = file.name.replace(".fits.gz", "").replace(".fits", "")
            astrometrynet_output_root = str(file.parent / "astrometry" / basename)

            solution: AstrometrySolution = await run_in_executor(
                solve_camera_with_astrometrynet,
                sources,
                ra=ra,
                dec=dec,
                solve_locs_kwargs={"output_root": astrometrynet_output_root},
            )

            # Now match with Gaia.
            if solution.solved:
                matched_sources, _ = match_with_gaia(solution.wcs, sources, concat=True)
                sources = matched_sources
                matched = True
                wcs = solution.wcs
            else:
                self.command.warning(
                    f"Failed solving camera {camname} with astrometry.net."
                )

        else:
            # This is now wcs_mode="gaia".

            # Find the reference file we want to compare with.
            assert self.last is not None

            last_solution = self.last[camname]
            ref_frame = last_solution.path
            ref_solution = self.solutions[get_frameno(ref_frame)]
            ref_wcs = ref_solution[camname].wcs

            # Here we match with Gaia first, then use those matches to
            # define the WCS.
            matched_sources, nmatches = match_with_gaia(
                ref_wcs,
                sources,
                concat=True,
                max_separation=5,
            )

            sources = matched_sources
            matched = True

            if nmatches < 5:
                self.command.warning(
                    f"Failed solving camera {camname}: insufficient number of "
                    "Gaia matches. Cannot generate astrometric solution. "
                    "Trying astrometry.net"
                )
                return await self.solve_camera(file, force_astrometry_net=True)
            else:
                wcs = wcs_from_gaia(matched_sources)
                wcs = wcs

        # Get zero-point. This is safe even if it did not solve.
        zp = estimate_zeropoint(data, sources)
        sources.update(zp)

        camera_solution = CameraSolution(
            frameno=frameno,
            camera=camname,
            path=file,
            sources=sources,
            wcs_mode=wcs_mode,
            wcs=wcs,
            matched=matched,
            ref_frame=ref_frame if wcs is not None else None,
            telescope=self.telescope,
        )

        if camera_solution.solved is False:
            self.command.warning(f"Camera {camname!r} failed to solve.")

        return camera_solution

    def calculate_offset(
        self,
        solution: GuiderSolution,
    ) -> tuple[tuple[float, float], tuple[float, float], float]:
        """Determines the offset to send to the telescope to acquire the field centre.

        Parameters
        ----------
        solution
            The guider solution.

        Returns
        -------
        offset
            A tuples of ra/dec and motor axis offsets to acquire the
            desired field centre, and the PA offset, in arcsec.

        """

        pra, pdec = solution.pixel_pointing
        fra, fdec = self.field_centre[0:2]

        mid_dec = (pdec + fdec) / 2

        ra_off: float = (fra - pra) * numpy.cos(numpy.radians(mid_dec))
        dec_off: float = fdec - pdec
        ra_arcsec = numpy.round(ra_off * 3600, 3)
        dec_arcsec = numpy.round(dec_off * 3600, 3)

        saz_diff_d, sel_diff_d = delta_radec2mot_axis(
            fra,
            fdec,
            pra,
            pdec,
            site=self.site,
        )

        field_pa = self.field_centre[2]
        if numpy.isnan(solution.pa):
            offset_pa = numpy.nan
        else:
            pointing_pa = solution.pa - 180
            offset_pa = field_pa - pointing_pa

            offset_pa %= 360
            if offset_pa > 180:
                offset_pa -= 360

        return ((ra_arcsec, dec_arcsec), (saz_diff_d, sel_diff_d), offset_pa)

    async def offset_telescope(
        self,
        off0: float,
        off1: float,
        off_rot: float,
        timeout: float = 10,
        use_motor_axes: bool = False,
    ):
        """Sends a correction offset to the telescope.

        Parameters
        ----------
        off0
            Offset in the first axis, in arcsec.
        off1
            Offset in the second axis, in arcsec.
        off_rot
            Offset in rotation, in arcsec.
        timeout
            Timeout for the offset.
        use_motor_axes
            Whether to apply the corrections as motor axes offsets.

        Returns
        -------
        applied_radec
            The applied correction in RA and Dec in arcsec. Zero if the correction
            is applied as RA/Dec.
        applied_motax
            The applied correction in motor axes in arcsec. Zero if the correction
            is applied as RA/Dec.
        applied_rot
            Correction applied to the k-mirror, in degrees.

        """

        applied_radec = numpy.array([0.0, 0.0])
        applied_motax = numpy.array([0.0, 0.0])
        applied_rot = 0.0

        max_ax_correction = self.config.get("max_ax_correction", 3600)

        min_rot_correction = self.config.get("min_rot_correction", 0.01)
        max_rot_correction = self.config.get("max_rot_correction", 3)

        if numpy.any(numpy.abs([off0, off1]) > max_ax_correction):
            self.command.error("Requested correction is too big. Not applying it.")
            return applied_radec, applied_motax, applied_rot

        telescope = self.command.actor.telescope
        pwi = f"lvm.{telescope}.pwi"

        if use_motor_axes is False:
            cmd_str = f"offset --ra_add_arcsec {off0} --dec_add_arcsec {off1}"
            applied_radec = numpy.array([off0, off1])
        else:
            cmd_str = f"offset --axis0_add_arcsec {off0} --axis1_add_arcsec {off1}"
            applied_motax = numpy.array([off0, off1])
            # TODO: calculate applied_radec here. Maybe Tom has the
            # delta_mot_axis2radec equivalent, if not it should not be hard
            # to derive.

        # By default lvmpwi requires an axis error < 0.4 to consider an offset done.
        # That's a bit overzealous in mid- to high-wind conditions. We apply a more
        # relaxed axis error and a timeout.
        axis_error = self.config["offset"]["axis_error"]
        cmd_str += f" --axis_error {axis_error} --timeout {timeout}"

        cmd = await self.command.send_command(pwi, cmd_str)

        if cmd.status.did_fail:
            raise RuntimeError(f"Failed offsetting telescope {telescope}.")

        guide_in_rot = self.config["has_kmirror"] and self.config["guide_in_rot"]

        if guide_in_rot and not self.is_guiding():
            if off_rot > max_rot_correction:
                self.command.warning("Requested rotator correction is too big.")

            # Note that we do not take abs(off_rot) here because we cannot apply
            # correction in negative direction. This prevents that.
            elif off_rot > min_rot_correction:
                km = f"lvm.{telescope}.km"
                cmd_km_str = f"slewAdjust --offset_angle {off_rot:.6f}"
                cmd_km = await self.command.send_command(km, cmd_km_str)

                if cmd_km.status.did_fail:
                    self.command.error(f"Failed offsetting k-mirror {telescope}.")
                else:
                    applied_rot = off_rot

        return applied_radec, applied_motax, applied_rot

    async def update_fits(self, guider_solution: GuiderSolution):
        """Updates the ``lvm.agcam`` files and creates the ``lvm.guider`` file."""

        coros = []

        assert guider_solution.sources is not None

        # Update lvm.agcam PROC extension.
        for solution in guider_solution.solutions:
            file = solution.path.absolute()
            pa = numpy.round(solution.pa, 4)
            zeropt = numpy.round(solution.zero_point, 3)
            reffile = solution.ref_frame.name if solution.ref_frame else None

            wcs_cards = {}
            if solution.wcs is not None:
                cards = solution.wcs.to_header().cards
                for card in cards:
                    wcs_cards[card.keyword] = (card.value, card.comment)

            proc_update = {
                "PA": pa if not numpy.isnan(pa) else None,
                "ZEROPT": zeropt if not numpy.isnan(zeropt) else None,
                "REFFILE": reffile,
                "SOLVED": solution.solved,
                "WCSMODE": solution.wcs_mode,
                **wcs_cards,
            }

            coros.append(run_in_executor(update_fits, file, "PROC", header=proc_update))

            # Update the sources file for each camera.
            sources_cam_path = file.with_suffix(".parquet")
            if solution.sources is not None:
                solution.sources.to_parquet(sources_cam_path)

        guider_path = get_guider_path(guider_solution.solutions[0].path)
        sources_path = guider_path.with_suffix(".parquet")

        gheader = header_from_model("GUIDERDATA_HEADER")

        gheader["GUIDERV"] = __version__
        gheader["TELESCOP"] = self.telescope
        gheader["GUIDMODE"] = "guide" if self.is_guiding() else "acquisition"
        gheader["MJD"] = get_sjd("LCO")
        gheader["DATE"] = Time.now().isot
        gheader["FRAMENO"] = guider_solution.frameno

        if "east" in guider_solution.cameras:
            gheader["FILEEAST"] = guider_solution["east"].path.name

        if "west" in guider_solution.cameras:
            gheader["FILEWEST"] = guider_solution["west"].path.name

        gheader["DIRNAME"] = str(guider_solution.solutions[0].path.parent)
        gheader["SOURCESF"] = str(sources_path.name)

        gheader["RAFIELD"] = numpy.round(self.field_centre[0], 6)
        gheader["DECFIELD"] = numpy.round(self.field_centre[1], 6)
        gheader["PAFIELD"] = numpy.round(self.field_centre[2], 4)
        gheader["XFFPIX"] = guider_solution.guide_pixel[0]
        gheader["ZFFPIX"] = guider_solution.guide_pixel[1]
        gheader["SOLVED"] = guider_solution.solved
        gheader["NCAMSOL"] = guider_solution.n_cameras_solved
        gheader["RAMEAS"] = nan_or_none(guider_solution.pointing[0], 6)
        gheader["DECMEAS"] = nan_or_none(guider_solution.pointing[1], 6)
        gheader["PAMEAS"] = nan_or_none(guider_solution.pa, 4)
        gheader["OFFRAMEA"] = nan_or_none(guider_solution.ra_off, 3)
        gheader["OFFDEMEA"] = nan_or_none(guider_solution.dec_off, 3)
        gheader["OFFA0MEA"] = nan_or_none(guider_solution.axis0_off, 3)
        gheader["OFFA1MEA"] = nan_or_none(guider_solution.axis1_off, 3)
        gheader["OFFPAMEA"] = nan_or_none(guider_solution.pa_off, 3)
        gheader["CORRAPPL"] = guider_solution.correction_applied
        gheader["RACORR"] = 0.0
        gheader["DECORR"] = 0.0
        gheader["PACORR"] = nan_or_none(guider_solution.correction[2], 4)
        gheader["AX0CORR"] = nan_or_none(guider_solution.correction[0], 3)
        gheader["AX1CORR"] = nan_or_none(guider_solution.correction[1], 3)
        gheader["AX0KP"] = self.pid_ax0.Kp
        gheader["AX0KI"] = self.pid_ax0.Ki
        gheader["AX0KD"] = self.pid_ax0.Kd
        gheader["AX1KP"] = self.pid_ax1.Kp
        gheader["AX1KI"] = self.pid_ax1.Ki
        gheader["AX1KD"] = self.pid_ax1.Kd
        gheader["ZEROPT"] = nan_or_none(guider_solution.zero_point, 3)

        if guider_solution.wcs:
            gheader.extend(guider_solution.wcs.to_header())

        guider_hdul = fits.HDUList([fits.PrimaryHDU()])
        guider_hdul.append(fits.ImageHDU(data=None, header=gheader, name="GUIDERDATA"))

        coros.append(run_in_executor(guider_hdul.writeto, str(guider_path)))

        coros.append(run_in_executor(guider_solution.sources.to_parquet, sources_path))

        with elapsed_time(self.command, "update PROC HDU and create lvm.guider file"):
            await asyncio.gather(*coros)
