#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-18
# @Filename: transformations.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
from datetime import datetime

from typing import Any, cast

import numpy
import pandas
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils.iers import conf
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
from scipy.spatial import KDTree

from lvmguider import config
from lvmguider.astrometrynet import AstrometrySolution, astrometrynet_quick
from lvmguider.extraction import extract_sources
from lvmguider.types import ARRAY_1D_F32, ARRAY_2D_F32


# Prevent astropy from downloading data.
conf.auto_max_age = None
conf.iers_degraded_accuracy = "ignore"


def ag_to_full_frame(
    camera: str,
    in_locs: numpy.ndarray,
    rot_shift: tuple | None = None,
    verbose=False,
):
    """Rotate and shift the star locations from an AG frame to the correct locations in
    the large "full frame" (master frame) corresponding to the full focal plane.

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
        Array of corresponding star locations in "full frame".

    """

    camera = camera.lower()

    # Get best-fit rotation and shift, based on NonLinLSQ_Fit.py
    if rot_shift is None:  # Use Best-Fit values from Suff_AGC.docx
        # Best-fit rotations and shifts
        # Last one is Sci on-axis for 21 Feb 23 - see Doc
        met_data = {
            "skyw-w": [-89.85, -76.53, 430.64],
            "skyw-e": [90.16, 3512.90, 433.74],
            "sci-w": [-89.84, -28.26, 415.33],
            "sci-e": [89.98, 3512.74, 447.25],
            "spec-e": [89.60, 3193.11, 504.26],
            "skye-w": [-90.69, -79.54, 436.38],
            "skye-e": [89.71, 3476.00, 457.13],
            "sci-c": [-89.995, 1707.4, 426.62],
        }

        # Pull out correct best-fit rotation and shift
        bF = met_data[camera]
        th, Sx, Sz = (numpy.radians(bF[0]), bF[1], bF[2])

    else:
        # Use supplied values
        th, Sx, Sz = (numpy.radians(rot_shift[0]), rot_shift[1], rot_shift[2])

    if verbose:
        print("")
        print("ag_to_full_frame: rot_shift for ", camera, numpy.degrees(th), Sx, Sz)
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


def full_frame_to_ag(camera: str, locs: numpy.ndarray):
    """Transforms the star locations from the full frame
    to the internal pixel locations in the AG frame.

    Adapted from Tom Herbst.

    Parameters
    ----------
    camera:
        Which AG camera: ``skyw-e``, ``skyw-w``, ``sci-c``, ``sci-e``, ``sci-w``,
        ``skye-e``, ``skye-w``, or ``spec-e``.
    locs
        Numpy 2D array of (x, y) star locations.
    Returns
    -------
    ag_loc
        Array of corresponding star locations in the AG frame.

    """

    camera = camera.lower()

    met_data = {
        "skyw-w": [-89.85, -76.53, 430.64],
        "skyw-e": [90.16, 3512.90, 433.74],
        "sci-w": [-89.84, -28.26, 415.33],
        "sci-e": [89.98, 3512.74, 447.25],
        "spec-e": [89.60, 3193.11, 504.26],
        "skye-w": [-90.69, -79.54, 436.38],
        "skye-e": [89.71, 3476.00, 457.13],
        "sci-c": [-89.995, 1707.4, 426.62],
    }

    # Pull out correct best-fit rotation and shift
    bF = met_data[camera]
    th, Sx, Sz = (numpy.radians(bF[0]), bF[1], bF[2])

    # Rotation matrix
    M = numpy.array(([numpy.cos(th), -numpy.sin(th)], [numpy.sin(th), numpy.cos(th)]))

    # 2xnPts rotation offset
    rOff = numpy.tile(numpy.array([[800.0, 550.0]]).transpose(), (1, locs.shape[0]))

    # Make 2xnPts Sx,Sz offset matrix
    off = numpy.tile(numpy.array([[Sx, Sz]]).transpose(), (1, locs.shape[0]))

    # Transform FF --> AG. See Single_Sensor_Solve doc
    rsLoc = numpy.dot(M, (locs.T - rOff - off)) + rOff

    # Return calculated positions (need to transpose for standard layout)
    return rsLoc.T


def solve_locs(
    locs: pandas.DataFrame,
    ra: float,
    dec: float,
    full_frame=True,
    index_paths: dict[int, str] | None = None,
    scales: dict[int, list[int]] | None = None,
    output_root: str | None = None,
    verbose: bool = False,
    raise_on_unsolved: bool = True,
    **astrometrynet_kwargs,
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
        Whether this is a full "full frame" field, requiring a different set of
        index files.
    index_paths
        The paths to the index files to use. A dictionary of series number
        to path on disk, e.g.,
        ``{5200: "/data/astrometrynet/5200", 4100: "/data/astrometrynet/5100"}``.
    scales
        Index scales to use. Otherwise uses all index files. The format
        is a dictionary of series number to a list of scales, e.g.,
        ``{5200: [4, 5], 4100: [10, 11]}``.
    output_root
        The path where to write the astrometry.net files. If `None`, uses temporary
        files.
    verbose
        Output additional information.
    raise_on_unsolved
        Whether to raise an error if the field was not solved.
    astrometrynet_kwargs
        Arguments to pass directly to `.astrometrynet_quick`.

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
        midX, midZ = config["xz_full_frame"]  # Middle of full Frame
        index_paths_default = {
            5200: "/data/astrometrynet/5200",
            4100: "/data/astrometrynet/4100",
        }
    else:
        midX, midZ = config["xz_ag_frame"]  # Middle of AG camera Frame
        index_paths_default = {5200: "/data/astrometrynet/5200"}

    index_paths = index_paths or index_paths_default
    scales = scales or {5200: [4, 5, 6]}

    locs = locs.copy()
    if full_frame:
        locs = locs.rename(columns={"x_full": "x", "y_full": "y"})

    solution = astrometrynet_quick(
        index_paths,
        locs,
        ra=ra,
        dec=dec,
        radius=radius,
        pixel_scale=(upper_bound + lower_bound) / 2,
        pixel_scale_factor_hi=upper_bound,
        pixel_scale_factor_lo=lower_bound,
        width=midX * 2,
        height=midZ * 2,
        scales=scales,
        raise_on_unsolved=raise_on_unsolved,
        output_root=output_root,
        **astrometrynet_kwargs,
    )

    if verbose:
        print("")
        print("Solution has match ? ", solution.solved)

    return solution


def radec2azel(raD, decD, lstD, site: EarthLocation | None = None):
    """Returns the Azimuth and Elevation for the supplied RA, Dec, and sidereal time.

    From Tom Herbst and Florian Briegel.

    Parameters
    ----------
    ra,dec
        Right Ascension and Declination in decimal degrees.
    lst
        Local Sidereal Time in decimal degrees.

    Returns
    -------
    az,el
        Azimuth and Elevation (Altitude) in decimal degrees

    """

    if site is None:
        site = EarthLocation.from_geodetic(
            lon=-70.70166667,
            lat=-29.00333333,
            height=2282.0,
        )

    assert isinstance(site, EarthLocation)

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


def delta_radec2mot_axis(
    ra_ref,
    dec_ref,
    ra_new,
    dec_new,
    date: datetime | None = None,
    site: EarthLocation | None = None,
):
    """RA/Dec offset to motor axes.

    From Tom Herbst and Florian Briegel.

    """

    if site is None:
        site = EarthLocation.from_geodetic(
            lon=-70.70166667,
            lat=-29.00333333,
            height=2282.0,
        )

    assert isinstance(site, EarthLocation)

    date = date or datetime.utcnow()
    observing_time = Time(date, scale="utc", location=site)
    lst = observing_time.sidereal_time("mean")

    ref_az_d, ref_el_d = radec2azel(ra_ref, dec_ref, lst.deg, site=site)
    new_az_d, new_el_d = radec2azel(ra_new, dec_new, lst.deg, site=site)

    ref_saz_d, ref_sel_d = azel2sazsel(ref_az_d, ref_el_d)
    new_saz_d, new_sel_d = azel2sazsel(new_az_d, new_el_d)

    saz_diff_d = ref_saz_d - new_saz_d
    sel_diff_d = ref_sel_d - new_sel_d

    # To arcsec
    saz_diff_arcsec = saz_diff_d * -3600.0
    sel_diff_arcsec = sel_diff_d * -3600.0

    return saz_diff_arcsec, sel_diff_arcsec


def solve_camera_with_astrometrynet(
    input: pandas.DataFrame | pathlib.Path | str,
    ra: float,
    dec: float,
    solve_locs_kwargs: dict | None = None,
) -> AstrometrySolution:
    """Astrometrically solves a single camera using ``astrometry.net``.

    Parameters
    ----------
    input
        The input data. This is normally a Pandas data frame, the result of
        running `.extract_sources` on an image. It can also be a path to
        a file, in which case sources will be extracted here.
    ra,dec
        An initial guess of the RA and Dec of the centre of the camera.
    solve_locs_kwargs
        A dictionary of kwargs to pass to `.solve_locs`.

    Returns
    -------
    solution
       An object containing the astrometric solution.

    """

    solve_locs_kwargs = solve_locs_kwargs if solve_locs_kwargs is not None else {}

    if isinstance(input, (str, pathlib.Path)):
        input = extract_sources(input)

    sources = input.copy()

    locs_i = sources.loc[:, ["x", "y", "flux"]].copy()
    locs_i.loc[:, ["x", "y"]] -= 0.5  # We want centre of pixel to be (0.5, 0.5)

    camera_solution = solve_locs(
        locs_i,
        ra,
        dec,
        full_frame=False,
        raise_on_unsolved=False,
        **solve_locs_kwargs.copy(),
    )

    return camera_solution


def get_crota2(wcs: WCS):
    """Determines the ``CROTA2`` angle, in degrees, from a CD matrix."""

    cd: ARRAY_2D_F32

    if wcs.wcs.has_crota():
        return wcs.wcs.crota[1]
    elif wcs.wcs.has_cd():
        cd = wcs.wcs.cd
    elif wcs.wcs.has_pc():
        cd = wcs.wcs.pc * wcs.wcs.cdelt
    else:
        raise ValueError("WCS does not have information to determine CROTA2.")

    crota2 = numpy.degrees(numpy.arctan2(-cd[0, 1], cd[1, 1]))
    return float(crota2 if crota2 > 0 else crota2 + 360)


def match_with_gaia(
    wcs: WCS,
    sources: pandas.DataFrame,
    gaia_sources: pandas.DataFrame | None = None,
    max_separation: float = 2,
    concat: bool = False,
    db_connection_params: dict[str, Any] = {},
) -> tuple[pandas.DataFrame, int]:
    """Match detections to Gaia sources using nearest neighbours.

    Parameters
    ----------
    wcs
        The WCS associated with the detections. Used to determine pixels on the
        image frame.
    sources
        A data frame with extracted sources.
    gaia_sources
        A data frame with Gaia sources to be matched. If `None`, Gaia sources
        will be queried from the database.
    max_separation
        Maximum separation between detections and matched Gaia sources, in arcsec.
    concat
        If `True`, the returned data frame is the input ``sources`` concatenated
        with the match information.
    db_connection_params
        Database connection parameters to pass to `.get_gaia_sources` if
        ``gaia_sources`` are not passed.

    Returns
    -------
    matches
        A tuple with the matched data frame and the number of matches.

    """

    from lvmguider.tools import get_gaia_sources

    PIXSCALE = 1.009  # arcsec/pix

    # Epoch difference with Gaia DR3.
    GAIA_EPOCH = 2016.0
    epoch = Time.now().jyear
    epoch_diff = epoch - GAIA_EPOCH

    if gaia_sources is None:
        gaia_sources = get_gaia_sources(wcs, db_connection_params=db_connection_params)

    sources = sources.copy()
    gaia_sources = gaia_sources.copy()

    # Match detections with Gaia sources. Take into account proper motions.
    ra_gaia: ARRAY_1D_F32 = gaia_sources.ra.to_numpy(numpy.float32)
    dec_gaia: ARRAY_1D_F32 = gaia_sources.dec.to_numpy(numpy.float32)
    pmra: ARRAY_1D_F32 = gaia_sources.pmra.to_numpy(numpy.float32)
    pmdec: ARRAY_1D_F32 = gaia_sources.pmdec.to_numpy(numpy.float32)

    pmra_gaia = numpy.nan_to_num(pmra) / 1000 / 3600  # deg/yr
    pmdec_gaia = numpy.nan_to_num(pmdec) / 1000 / 3600

    ra_epoch = ra_gaia + pmra_gaia / numpy.cos(numpy.radians(dec_gaia)) * epoch_diff
    dec_epoch = dec_gaia + pmdec_gaia * epoch_diff

    gaia_sources["ra_epoch"] = ra_epoch
    gaia_sources["dec_epoch"] = dec_epoch

    # Calculate x/y pixels of the Gaia detections. We use origin 0 but the
    # sep/SExtractor x/y in sources assume that the centre of the lower left
    # pixel is (1,1) so we adjust the returned pixel values.
    xpix, ypix = wcs.wcs_world2pix(ra_epoch, dec_epoch, 0)
    gaia_sources["xpix"] = xpix + 0.5
    gaia_sources["ypix"] = ypix + 0.5

    tree = KDTree(gaia_sources.loc[:, ["xpix", "ypix"]].to_numpy())
    dd, ii = tree.query(sources.loc[:, ["x", "y"]].to_numpy())
    valid = dd < max_separation

    # Get Gaia rows for the valid matches. Change their indices to those
    # of their matching sources (which are 0..len(sources)-1 since we reindexed).
    matches: pandas.DataFrame = gaia_sources.iloc[ii[valid]].copy()
    assert isinstance(matches, pandas.DataFrame)

    matches.index = pandas.Index(numpy.arange(len(ii))[valid])

    dx = (matches.xpix - sources.x).astype(numpy.float32)
    dy = (matches.ypix - sources.y).astype(numpy.float32)
    match_sep = cast(pandas.Series, numpy.hypot(dx, dy) * PIXSCALE)
    matches.loc[:, "match_sep"] = match_sep.loc[matches.index]

    matches = matches.drop(columns=["xpix", "ypix"])

    if concat:
        sources.loc[:, matches.columns] = matches
        return sources, valid.sum()

    return matches, valid.sum()


def wcs_from_gaia(sources: pandas.DataFrame, xy_cols: list[str] = ["x", "y"]) -> WCS:
    """Creates a WCS from Gaia-matched sources."""

    # Get useful columns. Drop NaNs.
    matched_sources = sources.loc[:, ["ra_epoch", "dec_epoch", *xy_cols]]
    matched_sources.dropna(inplace=True)

    if len(matched_sources) < 5:
        raise RuntimeError("Insufficient number of Gaia matches.")

    skycoords = SkyCoord(
        ra=matched_sources.ra_epoch,
        dec=matched_sources.dec_epoch,
        unit="deg",
        frame="icrs",
    )

    return fit_wcs_from_points(
        (matched_sources[xy_cols[0]], matched_sources[xy_cols[1]]),
        skycoords,
    )
