#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-18
# @Filename: transformations.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import numpy
import pandas
from astropy.io import fits
from astropy.wcs.utils import pixel_to_skycoord

from lvmguider.astrometrynet import astrometrynet_quick


# Middle of an AG frame or full frame
XZ_FULL_FRAME = (2500.0, 1000.0)
XZ_FRAME = (800.0, 550.0)


def rot_shift_locs(
    camera: str,
    in_locs: numpy.ndarray,
    rot_shift: tuple | None = None,
    verbose=False,
):
    """Rotate and shift the star locations from an AG frame to the correct locations in
    the large "master frame" corresponding to the full focal plane.

    Adapted from Tom Herbst's code.

    Parameters
    ----------
    camera:
        Which AG camera: ``skyw-e``, ``skyw-w``, ``sci-c``, ``sci-e``, ``sci-w``,
        ``skye-e``, ``skye-w``, or ``spec-e``.
    in_locs
        Numpy 2D array of (x, y) star locations.
    rot_shift
        List of input ``[theta_deg, Sx, Sz]` to use. If `None`, use best fit values

    Returns
    -------
    rs_loc
        Array of corresponding star locations in "master frame".

    """

    # Get best-fit rotation and shift, based on NonLinLSQ_Fit.py

    if rot_shift is None:  # Use Best-Fit values from Suff_AGC.docx
        cam_idx = [
            "skyw-w",
            "skyw-e",
            "sci-w",
            "sci-e",
            "spec-e",
            "skye-w",
            "skye-e",
            "sci-c",
        ].index(camera)

        # Best-fit rotations and shifts
        bF = numpy.array(
            (
                [
                    [-89.85, -76.53, 430.64],
                    [90.16, 3512.90, 433.74],
                    [-89.84, -28.26, 415.33],
                    [89.98, 3512.74, 447.25],
                    [89.60, 3193.11, 504.26],
                    [-90.69, -79.54, 436.38],
                    [89.71, 3476.00, 457.13],
                    [-89.995, 1707.4, 426.62],
                ]
            )
        )  # Last one is Sci on-axis for 21 Feb 23 - see Doc

        # Pull out correct best-fit rotation and shift
        th, Sx, Sz = (numpy.radians(bF[cam_idx, 0]), bF[cam_idx, 1], bF[cam_idx, 2])

    else:
        # Use supplied values
        th, Sx, Sz = (numpy.radians(rot_shift[0]), rot_shift[1], rot_shift[2])

    if verbose:
        print("")
        print("rot_shift_locs: rot_shift for ", camera, numpy.degrees(th), Sx, Sz)
        print("")

    # Rotation matrix
    M = numpy.array(([numpy.cos(th), -numpy.sin(th)], [numpy.sin(th), numpy.cos(th)]))

    # 2xnPts rotation offset
    rOff = numpy.tile(numpy.array([[800.0, 550.0]]).transpose(), (1, in_locs.shape[0]))

    # Make 2xnPts Sx,Sz offset matrix
    off = numpy.tile(numpy.array([[Sx, Sz]]).transpose(), (1, in_locs.shape[0]))

    # Shift to center, Rotate, shift back, and offset
    rsLoc = numpy.dot(M, in_locs.T - rOff) + rOff + off

    # Return calculated positions (need to transpose for standard layout)
    return rsLoc.T


def solve_locs(
    locs: pandas.DataFrame,
    ra: float,
    dec: float,
    full_frame=True,
    verbose=False,
):
    """Use astrometry.net to solve field, given a series of star locations.

    Adapted from Tom Herbst's code.

    Parameters
    ----------
    locs
        A Pandas DataFrame of ``x``, ``y``, and ``flux`` star locations.
    ra
        The estimated RA of the field.
    dec
        The estimated Dec of the field.
    full_frame
        Whether this is a full "master frame" field, requiring a different set of
        index files.
    verbose
        Output additional information.

    Returns
    -------
    myWCS
        A WCS entity (or `None` if it fails) with appropriate location information.
    solveInfo
        A list containing information about the solve:
        solveInfo[0] - [raSlv,deSlv]  RA,Dec of central pixel
                 [1] - angSlv         Rotation of field
                 [2] - pxScl          Arcsec / pixel of solve

    """

    lower_bound = 0.8  # Pixel scale hints (arcsec/pixel)
    upper_bound = 1.2
    radius = 1  # Search radius in degrees

    if full_frame:
        midX, midZ = XZ_FULL_FRAME  # Middle of Master Frame
        series = 5200
        scales = [4, 5, 6]
    else:
        midX, midZ = XZ_FRAME  # Middle of AG camera Frame
        series = 5200
        scales = [5, 6]

    wcs = astrometrynet_quick(
        f"/data/astrometrynet/{series}",
        locs,
        ra=ra,
        dec=dec,
        radius=radius,
        pixel_scale=(upper_bound + lower_bound) / 2,
        pixel_scale_factor_hi=upper_bound,
        pixel_scale_factor_lo=lower_bound,
        scales=scales,
        series=series,
    )

    if verbose:
        print("")
        print("Solution has match ? ", wcs is not None)

    if wcs is None:
        return (None, None)

    # Calculate center and rotation of field

    tM = wcs.wcs.cd  # Transformation matrix
    angSlv = numpy.degrees(numpy.arctan2(tM[0, 0], tM[1, 0]))  # Angle from scaling

    raSlv = pixel_to_skycoord(midX, midZ, wcs).ra
    deSlv = pixel_to_skycoord(midX, midZ, wcs).dec

    pxScl = numpy.abs(tM[0, 0] * 3600)

    solveInfo = [[raSlv, deSlv], angSlv, pxScl]  # Solve information

    if verbose:
        print("Central pixel         :", midX, midZ)
        print("RA central pixel      :", raSlv)
        print("Dec central pixel     :", deSlv)
        print("Field rotation (deg)  :", angSlv)
        print("Pixel scale           :", pxScl)

    return wcs, solveInfo


def solve_from_files(files: list[str], telescope: str):
    """Determines the telescope pointing from a set of AG frames.

    Parameters
    ----------
    files
        The AG FITS files to use determine the telescope pointing. Normally
        these are a pair of east/west camera frames, except for ``spec``.
    telescope
        The telescope to which these exposures are associated.

    Returns
    -------
    wcs
        The wcs of the astrometric solution or `None` if no solution was found.

    """

    if len(files) == 0:
        raise ValueError("No files provided.")

    mf_sources: list[pandas.DataFrame] = []
    ra = 0.0
    dec = 0.0
    for file in files:
        header = fits.getheader(file)
        camname = header["CAMNAME"][0].lower()
        ra = header["RA"]
        dec = header["DEC"]

        sources = pandas.DataFrame(fits.getdata(file, "SOURCES"))
        xy = sources[["x", "y"]].values

        camera = f"{telescope}-{camname}"
        file_locs = rot_shift_locs(camera, xy)
        sources.loc[:, ["x", "y"]] = file_locs
        mf_sources.append(sources)

    locs = pandas.concat(mf_sources)
    wcs, _ = solve_locs(locs[["x", "y", "flux"]], ra=ra, dec=dec, full_frame=True)

    return wcs
