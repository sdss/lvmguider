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

from typing import TYPE_CHECKING, Any, cast

import numpy
import pandas
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from simple_pid import PID

from sdsstools.time import get_sjd

from lvmguider import __version__
from lvmguider.maskbits import GuiderStatus
from lvmguider.tools import elapsed_time, get_frameno, get_proc_path, run_in_executor

from .transformations import (
    XZ_FULL_FRAME,
    calculate_guide_offset,
    delta_radec2mot_axis,
    solve_camera,
    wcs_from_single_cameras,
)


if TYPE_CHECKING:
    from lvmguider.actor import GuiderCommand
    from lvmguider.astrometrynet import AstrometrySolution


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
        self.pixel = pixel or XZ_FULL_FRAME

        self.config = command.actor.config

        self._mode: str = "auto"

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

        self.use_reference_frames: bool = False
        self.reference_frames: dict[str, pathlib.Path] = {}
        self.reference_sources: pandas.DataFrame | None = None
        self.reference_wcs: dict[str, WCS] = {}

        self.site = EarthLocation(**self.config["site"])

    def set_reference_frames(
        self,
        frameno: int | None = None,
        mf_wcs: WCS | None = None,
    ):
        """Sets the reference images to be used for guiding."""

        # Reset to not using reference frames.
        if frameno is None:
            self.use_reference_frames = False
            self.reference_frames = {}
            self.reference_sources = None
            self.reference_wcs = {}
            return

        # Gets images that match frameno.
        sjd = get_sjd("LCO")
        agcam_dir = pathlib.Path(f"/data/agcam/{sjd}")
        files = list(agcam_dir.glob(f"lvm.{self.telescope}.agcam.*_{frameno:08d}.*"))

        if len(files) == 0:
            raise FileNotFoundError(f"No files found for frame number {frameno}.")

        if self.telescope != "spec" and len(files) == 1:
            self.command.warning(f"Only one file found for frame number {frameno}.")

        sources: list[pandas.DataFrame] = []
        for file_ in files:
            try:
                header_raw = fits.getheader(file_, "RAW")
                header_proc = fits.getheader(file_, "PROC")
                file_sources = Table.read(file_, "SOURCES").to_pandas()
            except Exception as err:
                raise ValueError(f"Failed retrieving sources from file {file_}: {err}")

            camname: str = header_raw["CAMNAME"]
            sources.append(file_sources)

            self.reference_frames[camname] = file_
            self.reference_wcs[camname] = WCS(header_proc)

        if mf_wcs is None:
            proc_path = get_proc_path(files[0])
            if not proc_path.exists():
                raise CriticalGuiderError("Cannot retrieve reference frame master WCS.")
            self.reference_wcs = WCS(fits.getheader(str(proc_path), "ASTROMETRY"))
        else:
            self.reference_wcs["master"] = mf_wcs

        self.reference_sources = pandas.concat(sources)
        self.use_reference_frames = True

    def set_pixel(self, pixel_x: float | None = None, pixel_z: float | None = None):
        """Sets the master frame pixel coordinates ``(x, z)`` on which to guide."""

        if pixel_x is None or pixel_z is None:
            new_pixel = XZ_FULL_FRAME
        else:
            new_pixel = (pixel_x, pixel_z)

        if self.pixel != new_pixel and self._mode == "auto":
            # In auto mode, if we change the current pixel, we start drifting
            # towards a new target position.
            self.command.actor.status |= GuiderStatus.DRIFTING
            self.pixel = new_pixel
            self.set_reference_frames()

        return self.pixel

    def reset_acquisition(self):
        """Reset the flags for acquisition mode."""

        self.set_reference_frames()
        self.command.actor._status &= ~GuiderStatus.GUIDING
        self.command.actor._status |= GuiderStatus.ACQUIRING
        self.command.actor.status |= GuiderStatus.DRIFTING

    async def guide_one(
        self,
        exposure_time: float = 5.0,
        mode: str = "auto",
        guide_tolerance: float | None = None,
        apply_correction: bool = True,
        use_motor_offsets: bool = True,
    ):
        """Performs one guide iteration.

        This is the main routine that executes one acquisition or guide iteration.
        When ``mode='auto'``, the code will do an acquisition step unless
        ``use_reference_frames=True``. At the end of the acquisition step, if
        the measured offset is smaller than ``guide_tolerance``, it will set
        ``use_reference_frames`` and the following iterations will be performed
        using the guide mechanism.

        During acquisition, in order, it does the following:

        - Expose the cameras. Subtract dark frame.
        - Extract sources.
        - Create a master frame based on the camera(s)-to-IFU metrology.
        - Solve the frame using astrometry.net.
        - Determine the offsets for a given pixel on the master frame. By default
          the central pixel is used, which corresponds to guiding at the central
          fibre in the IFU, but a different location can be used to guide, for example,
          on one of the spec mask holes.
        - Send PID-corrected offsets to the telescopes.

        Guiding is similar but instead of solving a new master frame WCS, the
        extracted sources are compared with the list of sources from the reference
        frame, and associated using nearest-neighbours. From there, an offset is
        derived and sent to the telescope.

        Parameters
        ----------
        exposure_time
            The exposure time, in seconds.
        mode
            Either ``'auto'``, ``'guide'``, or ``'acquire'``. In auto mode acquisition
            will be performed until the measured offset is smaller than
            ``guide_tolerance``. The other two modes force continuous acquisition
            or guiding.
        guide_tolerance
            The separation between field RA/Dec and measured pointing at which
            to consider than acquisition has been completed and guiding begins.
        apply_corrections
            Whether to apply the measured corrections. If `False`, only measures.
        use_motor_offsets
            If `True`, applies the corrections as motor axis offsets.
        use_individual_images
            If `True`, the WCS for the master frame in acquisition is derived from
            the individual WCS of the AG frames.

        """

        if self.use_reference_frames and self.reference_sources is None:
            raise CriticalGuiderError("No reference sources defined.")

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

        if mode == "auto":
            is_acquisition = bool(
                not self.use_reference_frames
                or self.command.actor.status & GuiderStatus.DRIFTING
            )
        elif mode == "guide":
            is_acquisition = False
        elif mode == "acquire":
            is_acquisition = True
        else:
            raise CriticalGuiderError(f"Invalid mode {mode}.")

        self._mode = mode

        if is_acquisition:
            try:
                ra_p, dec_p, wcs = await self.determine_pointing(
                    filenames,
                    pixel=self.pixel,
                )

                # Offset is field centre - current pointing.
                pnt = (ra_p, dec_p)
                offset_radec, offset_motax, sep = self.calculate_telescope_offset(pnt)

            except Exception as err:
                raise RuntimeError(f"Failed determining telescope pointing: {err}")

        else:
            offset_radec, sep = await self.calculate_guide_offset(sources, filenames)

            if numpy.any(numpy.isnan(offset_radec)) or numpy.isnan(sep):
                self.reset_acquisition()
                raise ValueError(
                    "Invalid values found in guide measurement. "
                    "Reverting to acquisition."
                )

            wcs = self.reference_wcs["master"]

            # Pointing from the reference frame.
            pixel = self.pixel or XZ_FULL_FRAME
            ref_pointing = wcs.pixel_to_world(*pixel)
            ra_ref = ref_pointing.ra.deg
            dec_ref = ref_pointing.dec.deg

            # Calculate current pointing. Note to future José: that although the
            # offset is measured as going from measured to reference sources here
            # we need to add it because that offset is the result of moving the
            # telescope and not the sky (just draw it).
            cos_dec = numpy.cos(numpy.radians(dec_ref))
            ra_p = numpy.round(ra_ref + offset_radec[0] / 3600 / cos_dec, 6)
            dec_p = numpy.round(dec_ref + offset_radec[1] / 3600.0, 6)

            # Where we really want to point.
            fra, fdec = self.field_centre

            # Calculate offset in motor axes.
            saz_diff_d, sel_diff_d = delta_radec2mot_axis(
                fra,
                fdec,
                ra_p,
                dec_p,
                site=self.site,
            )
            offset_motax = (saz_diff_d, sel_diff_d)

        self.command.info(
            measured_pointing={
                "frameno": frameno,
                "ra": ra_p,
                "dec": dec_p,
                "radec_offset": list(numpy.round(offset_radec, 3)),
                "motax_offset": list(numpy.round(offset_motax, 3)),
                "separation": sep,
                "mode": "guide" if self.use_reference_frames else "acquisition",
            }
        )

        if not is_acquisition:
            if self.command.actor.status & GuiderStatus.DRIFTING:
                raise ValueError("Guider is drifting. Reverting to acquisition.")

            if (sep > guide_tolerance) and mode == "auto":
                self.reset_acquisition()
                raise ValueError(
                    "Guide measured offset exceeds guide tolerance. "
                    "Skipping correction and reverting to acquisition."
                )

        # Calculate the correction.
        corr_radec = numpy.array(offset_radec)  # In RA/Dec
        corr_motax = numpy.array(offset_motax)  # In motor axes

        # Apply PID
        if use_motor_offsets:
            corr_motax[0] = self.pid_ra(corr_motax[0])
            corr_motax[1] = self.pid_dec(corr_motax[1])
        else:
            corr_radec[0] = self.pid_ra(corr_radec[0])
            corr_radec[1] = self.pid_dec(corr_radec[1])

        applied_radec = applied_motax = numpy.array([0.0, 0.0])
        try:
            if apply_correction:
                self.command.actor.status &= ~GuiderStatus.PROCESSING
                self.command.actor.status |= GuiderStatus.CORRECTING

                applied_radec, applied_motax = await self.offset_telescope(
                    *corr_motax,
                    use_motor_axes=use_motor_offsets,
                    max_correction=self.config.get("max_correction", 3600),
                )

        finally:
            self.command.actor.status &= ~GuiderStatus.CORRECTING

            self.command.info(
                correction_applied={
                    "frameno": frameno,
                    "radec_applied": list(numpy.round(applied_radec, 3)),
                    "motax_applied": list(numpy.round(applied_motax, 3)),
                }
            )

            # Should we start guiding?
            if (
                mode == "auto"
                and not self.use_reference_frames
                and sep < guide_tolerance
            ):
                self.command.info("Guide tolerance reached. Starting to guide.")
                self.set_reference_frames(frameno, mf_wcs=wcs)
                self.command.actor._status &= ~GuiderStatus.ACQUIRING
                self.command.actor._status &= ~GuiderStatus.DRIFTING
                self.command.actor.status |= GuiderStatus.GUIDING

            asyncio.create_task(
                self.write_proc_file(
                    filenames=filenames,
                    wcs=wcs,
                    ra_p=ra_p,
                    dec_p=dec_p,
                    offset_radec=tuple(offset_radec),
                    corr_radec=tuple(numpy.round(applied_radec, 3)),
                    corr_motax=tuple(numpy.round(applied_motax, 3)),
                    sources=list(sources),
                    is_acquisition=is_acquisition,
                )
            )

            # Guide step was successful, remove any FAILED flags.
            self.command.actor.status &= ~GuiderStatus.FAILED

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

        solutions: list[dict[str, AstrometrySolution]] = await asyncio.gather(
            *[run_in_executor(solve_camera, file, telescope) for file in filenames]
        )

        # Convert to dictionary of camera name to astrometric solution.
        camera_solutions = {k: v for d in solutions for k, v in d.items()}

        wcs = await run_in_executor(
            wcs_from_single_cameras,
            camera_solutions,
            telescope,
        )

        for fn, camname in enumerate(camera_solutions):
            wcs_camera = camera_solutions[camname].wcs
            if camera_solutions[camname].solved is False:
                self.command.warning(f"Camera {camname} did not solve.")
            elif wcs_camera is not None:
                file = filenames[fn]

                # Update proc header with astrometry.net WCS.
                with elapsed_time(self.command, "update astrometry in PROC raw frames"):
                    with fits.open(str(file), mode="update") as hdul:
                        if "PROC" in hdul:
                            proc = hdul["PROC"]
                        else:
                            proc = fits.ImageHDU(name="PROC")
                            hdul.append(proc)

                        proc.header.update(wcs_camera.to_header())
                        proc.header["WCSMODE"] = "astrometrynet"

        if wcs is None:
            raise ValueError(f"Cannot determine pointing for telescope {telescope}.")

        pointing: Any = wcs.pixel_to_world(*pixel)

        wcs = cast(WCS, wcs)

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

        saz_diff_d, sel_diff_d = delta_radec2mot_axis(
            fra,
            fdec,
            pra,
            pdec,
            site=self.site,
        )

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

        # By default lvmpwi requires an axis error < 0.4 to consider an offset done.
        # That's a bit overzealous in mid- to high-wind conditions. We apply a more
        # relaxed axis error and a timeout.
        mode = "guide" if self.use_reference_frames else "acquisition"
        axis_error = self.config["offset"]["axis_error"]
        timeout = self.config["offset"]["timeout"][mode]
        cmd_str += f" --axis_error {axis_error} --timeout {timeout}"

        cmd = await self.command.send_command(pwi, cmd_str)

        if cmd.status.did_fail:
            raise RuntimeError(f"Failed offsetting telescope {telescope}.")

        return applied_radec, applied_motax

    async def calculate_guide_offset(
        self,
        sources: list[pandas.DataFrame],
        filenames: list[str] = [],
    ):
        """Determines the guide offset by matching sources to reference images.

        Parameters
        ----------
        sources
            A Pandas DataFrame with the sources to match to the reference frame.
            Must contain a column ``camera`` with the camera associated with each
            source.
        filenames
            The filenames from which the sources have been extracted. They will
            be used to update the WCS of their ``PROC`` headers.

        Returns
        -------
        offset
            A tuple of RA and Dec offsets, in arcsec.
        separation
            The absolute separation between the reference frame and the
            new set of sources, in arcsec.

        """

        if self.reference_sources is None or "master" not in self.reference_wcs:
            raise CriticalGuiderError("Missing reference frame data. Cannot guide.")

        offset_xy, offset_radec, sep_arcsec, _ = await run_in_executor(
            calculate_guide_offset,
            pandas.concat(sources),
            self.telescope,
            self.reference_sources,
            self.reference_wcs["master"],
        )

        # Update proc- header with new astrometry.
        with elapsed_time(self.command, "update astrometry in PROC raw frames"):
            for file_ in filenames:
                with fits.open(str(file_), mode="update") as hdul:
                    camname = hdul["RAW"].header["CAMNAME"]
                    wcs = self.reference_wcs.get(camname, None)
                    if wcs is None:
                        self.command.warning(f"No WCS for reference frame {file_!s}.")
                        continue

                    wcs = self.reference_wcs[camname].deepcopy()
                    wcs.wcs.crpix -= numpy.array(offset_xy)
                    if "PROC" in hdul:
                        proc = hdul["PROC"]
                    else:
                        proc = fits.ImageHDU()
                    proc.header.update(wcs.to_header())
                    proc.header["WCSMODE"] = "astrometrynet+guide"

        # Rounding.
        offset_radec = tuple(numpy.round(offset_radec, 3))
        sep_arcsec = float(numpy.round(sep_arcsec, 3))

        # Typing.
        offset_radec = cast(tuple[float, float], offset_radec)
        sep_arcsec = cast(float, sep_arcsec)

        return offset_radec, sep_arcsec

    async def write_proc_file(
        self,
        filenames: list[str],
        wcs: WCS | None,
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

        astro_hdr["GUIDERV"] = (__version__, "Version of lvmguider")
        astro_hdr["TELESCOP"] = (self.telescope, "Telescope that took the images")
        astro_hdr["ACQUISIT"] = (is_acquisition, "Acquisition or guiding?")
        astro_hdr["DATE"] = (Time.now().isot, "Date of file creation")
        astro_hdr["FRAMENO"] = (get_frameno(filenames[0]), "Frame number")

        for fn, file_ in enumerate(filenames):
            astro_hdr[f"FILE{fn}"] = (str(file_), f"AG frame {fn}")

        ref_frames = list(self.reference_frames.values())
        for fn in range(2):
            if len(ref_frames) >= fn + 1:
                file_ = str(ref_frames[fn])
            else:
                file_ = ""
            astro_hdr[f"REFFILE{fn}"] = (file_, f"AG reference frame {fn}")

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

        if wcs is not None:
            astro_hdr += wcs.copy().to_header()

        if not is_acquisition:
            # Update WCS from the reference one using measured offsets.
            try:
                cos_dec = numpy.cos(numpy.radians(astro_hdr["CRVAL2"]))
                crval1 = astro_hdr["CRVAL1"] - offset_radec[0] / 3600 * cos_dec
                crval2 = astro_hdr["CRVAL2"] - offset_radec[1] / 3600

                astro_hdr["CRVAL1"] = crval1
                astro_hdr["CRVAL2"] = crval2
            except Exception as err:
                self.command.warning(f"Failed updating master frame WCS: {err}")

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
