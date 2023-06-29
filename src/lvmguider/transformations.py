#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-18
# @Filename: transformations.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import warnings
from datetime import datetime

import numpy
import pandas
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.time import Time
from astropy.wcs.utils import pixel_to_skycoord

from lvmguider.astrometrynet import astrometrynet_quick
from lvmguider.extraction import extract_marginal
from lvmguider.tools import get_proc_path


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
    return (rsLoc.T, (th, Sx, Sz))


def solve_locs(
    locs: pandas.DataFrame,
    ra: float,
    dec: float,
    full_frame=True,
    output_root: str | None = None,
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
    output_root
        The path where to write the astrometry.net files. If `None`, uses temporary
        files.
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
    radius = 5  # Search radius in degrees

    if full_frame:
        midX, midZ = XZ_FULL_FRAME  # Middle of Master Frame
    else:
        midX, midZ = XZ_FRAME  # Middle of AG camera Frame

    wcs = astrometrynet_quick(
        ["/data/astrometrynet/5200", "/data/astrometrynet/4100"],
        locs,
        ra=ra,
        dec=dec,
        radius=radius,
        pixel_scale=(upper_bound + lower_bound) / 2,
        pixel_scale_factor_hi=upper_bound,
        pixel_scale_factor_lo=lower_bound,
        width=midX * 2,
        height=midZ * 2,
        verbose=True,
        plot=True,
        output_root=output_root,
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

        try:
            sources = pandas.DataFrame(fits.getdata(file, "SOURCES"))
        except KeyError:
            warnings.warn("SOURCES ext not found. Extracting sources", UserWarning)
            sources = extract_marginal(fits.getdata(file))

        xy = sources[["x", "y"]].values

        camera = f"{telescope}-{camname}"
        file_locs, _ = rot_shift_locs(camera, xy)
        sources.loc[:, ["x", "y"]] = file_locs
        mf_sources.append(sources)

    locs = pandas.concat(mf_sources)

    # Generate root path for astrometry files.
    proc_path = get_proc_path(files[0])
    dirname = proc_path.parent
    proc_base = proc_path.name.replace(".fits", "")
    output_root = str(dirname / "astrometry" / proc_base)

    wcs, _ = solve_locs(
        locs[["x", "y", "flux"]],
        ra=ra,
        dec=dec,
        full_frame=True,
        output_root=output_root,
    )

    return wcs


def radec2azel(raD, decD, lstD):
    """Returns the Azimuth and Elevation for the supplied RA, Dec, and sidereal time.

    From Tom Herbst and Florian Briegel.

    Parameters
    ----------
    ra,dec
        Right Ascension and Declination in decimal degrees.
    lst
        Local Sideral Time in decimal degrees.

    Returns
    -------
    az,el
        Azimuth and Elevation (Altitude) in decimal degrees

    """

    site = EarthLocation.of_site("Las Campanas Observatory")

    lat_r = numpy.radians(site.lat.deg)

    ra, dec, lst = (
        numpy.radians(raD),
        numpy.radians(decD),
        numpy.radians(lstD),
    )  # Convert to radians

    ha = lst - ra

    el = numpy.arcsin(
        numpy.sin(dec) * numpy.sin(lat_r)
        + numpy.cos(dec) * numpy.cos(lat_r) * numpy.cos(ha)
    )

    rat = (numpy.sin(dec) - numpy.sin(el) * numpy.sin(lat_r)) / (
        numpy.cos(el) * numpy.cos(lat_r)
    )  # Ratio - need to pin [-1,1]

    if rat < -1.0:  # Goes wonky if roundoff puts it outside [1,1]
        rat = -1.0

    if rat > 1.0:
        rat = 1.0

    if numpy.sin(ha) < 0.0:
        az = numpy.arccos(rat)
    else:
        az = 2.0 * numpy.pi - numpy.arccos(rat)

    return numpy.degrees(az), numpy.degrees(el)


def azel2sazsel(azD, elD):
    """Returns the siderostat coordinates (saz, Sel) for the supplied Az-El.

    From Tom Herbst and Florian Briegel.

    Parameters
    ----------
    az,el
        Azimuth and Elevation (Altitude) in decimal degrees

    Returns
    -------
    sazD,selD
        Siderostat angles in degrees

    """

    r90 = numpy.radians(90.0)  # 90 deg in radians
    az, el = numpy.radians(azD), numpy.radians(elD)  # Convert to radians
    SEl = numpy.arccos(numpy.cos(el) * numpy.cos(az)) - r90  # SEl in radians
    rat = numpy.sin(el) / numpy.cos(SEl)  # Ratio
    if azD < 180.0:
        SAz = r90 - numpy.arcsin(rat)  # saz in radians
    else:
        SAz = numpy.arcsin(rat) - r90

    return numpy.degrees(SAz), numpy.degrees(SEl)  # Return values in degrees


def delta_radec2mot_axis(ra_ref, dec_ref, ra_new, dec_new):
    """RA/Dec offset to motor axes.

    From Tom Herbst and Florian Briegel.

    """

    observing_location = EarthLocation.of_site("Las Campanas Observatory")
    observing_time = Time(datetime.utcnow(), scale="utc", location=observing_location)
    lst = observing_time.sidereal_time("mean")

    ref_az_d, ref_el_d = radec2azel(ra_ref, dec_ref, lst.deg)
    new_az_d, new_el_d = radec2azel(ra_new, dec_new, lst.deg)

    ref_saz_d, ref_sel_d = azel2sazsel(ref_az_d, ref_el_d)
    new_saz_d, new_sel_d = azel2sazsel(new_az_d, new_el_d)

    saz_diff_d = ref_saz_d - new_saz_d
    sel_diff_d = ref_sel_d - new_sel_d

    # To arcsec
    saz_diff_arcsec = saz_diff_d * -3600.0
    sel_diff_arcsec = sel_diff_d * -3600.0

    return saz_diff_arcsec, sel_diff_arcsec
