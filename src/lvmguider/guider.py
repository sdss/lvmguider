#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-19
# @Filename: guider.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy

from .tools import run_in_executor
from .transformations import XZ_FULL_FRAME, solve_from_files


if TYPE_CHECKING:
    from astropy.wcs import WCS

    from lvmguider.actor import GuiderCommand


async def determine_pointing(
    telescope: str,
    filenames: list[str],
    pixel: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Returns the pointing of a telescope based on AG frames.

    Parameters
    ----------
    telescope
        The name of the telescope for which the pointing is being determined.
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

    pixel = pixel or XZ_FULL_FRAME
    wcs: WCS | None = await run_in_executor(solve_from_files, filenames, telescope)

    if wcs is None:
        raise ValueError(f"Cannot determine pointing for telescope {telescope}.")

    pointing = wcs.pixel_to_world(*pixel)

    return (pointing.ra.deg, pointing.dec.deg)  # type:ignore


def calculate_telescope_offset(
    pointing: tuple[float, float],
    field_centre: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    """Determines the offset to send to the telescope to acquire the field centre.

    Parameters
    ----------
    pointing
        The current pointing of the telescope, as determined by `.determine_pointing`.
    field_centre
        The field centre to which to offset.

    Returns
    -------
    offset
        A tuple of ra/dec offsets to acquire the desired field centre, in arcsec.
    angle
        The angle between pointing and field centre, in arcsec.

    """

    pra, pdec = pointing
    fra, fdec = field_centre

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

    return ((ra_off * 3600.0, dec_off * 3600.0), sep * 3600.0)


async def offset_telescope(command: GuiderCommand, ra_off: float, dec_off: float):
    """Sends a correction offset to the telescope.

    Parameters
    ----------
    command
        The actor command used to communicate with the actor system.
    ra_off
        The RA offset, in arcsec.
    dec_off
        The Dec offset, in arcsec.

    """

    telescope = command.actor.telescope
    pwi = f"lvm.{telescope}.pwi"

    cmd = await command.send_command(
        pwi,
        f"offset --ra_add_arcsec {ra_off} --dec_add_arcsec {dec_off}",
    )

    if cmd.status.did_fail:
        raise RuntimeError(f"Failed offsetting telescope {telescope}.")

    return True
