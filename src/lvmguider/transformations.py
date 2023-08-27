#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-18
# @Filename: transformations.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import warnings
from datetime import datetime

import nptyping as npt
import numpy
import pandas
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.utils.iers import conf
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
from scipy.spatial import KDTree

from lvmguider.astrometrynet import AstrometrySolution, astrometrynet_quick
from lvmguider.extraction import extract_marginal


# Prevent astropy from downloading data.
conf.auto_download = False
conf.iers_degraded_accuracy = "ignore"

# Middle of an AG frame or full frame
XZ_FULL_FRAME = (2500.0, 1000.0)
XZ_AG_FRAME = (800.0, 550.0)

ARRAY_2D_F32 = npt.NDArray[npt.Shape["*, *"], npt.Float32]


def ag_to_master_frame(
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
        print("ag_to_master_frame: rot_shift for ", camera, numpy.degrees(th), Sx, Sz)
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
        Whether this is a full "master frame" field, requiring a different set of
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
        midX, midZ = XZ_FULL_FRAME  # Middle of Master Frame
        index_paths_default = {
            5200: "/data/astrometrynet/5200",
            4100: "/data/astrometrynet/4100",
        }
    else:
        midX, midZ = XZ_AG_FRAME  # Middle of AG camera Frame
        index_paths_default = {5200: "/data/astrometrynet/5200"}

    index_paths = index_paths or index_paths_default
    scales = scales or {5200: [4, 5, 6]}

    locs = locs.copy()
    if full_frame:
        locs = locs.rename(columns={"x_master": "x", "y_master": "y"})

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
        Local Sideral Time in decimal degrees.

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

    observing_time = Time(datetime.utcnow(), scale="utc", location=site)
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


def solve_camera(
    file: str | pathlib.Path,
    telescope: str | None = None,
    reextract_sources: bool = False,
    solve_locs_kwargs: dict | None = None,
) -> dict[str, AstrometrySolution]:
    """Astrometrically solves a single camera, potentially extracting sources.

    Parameters
    ----------
    file
        The AG FITS file to solve.
    telescope
        The telescope to which the exposure is associated. If `None`, it
        is determined from the header.
    reextract_sources
        Runs the source extraction algorithm again. If `False`, extraction is only
        done if a ``SOURCES`` extensions is not found in the files.
    solve_locs_kwargs
        A dictionary of kwargs to pass to `.solve_locs`.

    Returns
    -------
    solutions
       A dictionary of camera name to astrometric solution.

    """

    solve_locs_kwargs = solve_locs_kwargs if solve_locs_kwargs is not None else {}

    file = pathlib.Path(file).absolute()
    header = fits.getheader(file, "RAW")
    camname: str = header["CAMNAME"].lower()
    ra: float = header["RA"]
    dec: float = header["DEC"]
    telescope = telescope or header["TELESCOP"].lower()

    if reextract_sources is False:
        try:
            sources = pandas.DataFrame(fits.getdata(file, "SOURCES"))
        except KeyError:
            warnings.warn("SOURCES ext not found. Extracting sources", UserWarning)
            return solve_camera(
                file,
                telescope=telescope,
                reextract_sources=True,
                solve_locs_kwargs=solve_locs_kwargs,
            )
    else:
        hdul = fits.open(file)
        if "PROC" in hdul:
            data = hdul["PROC"].data
        else:
            data = hdul["RAW"].data
        sources = extract_marginal(data)
        sources["camera"] = camname

    solve_locs_kwargs_cam = solve_locs_kwargs.copy()
    if "output_root" not in solve_locs_kwargs_cam:
        # Generate root path for astrometry files.
        basename = file.name.replace(".fits.gz", "").replace(".fits", "")
        output_root = str(file.parent / "astrometry" / basename)
        solve_locs_kwargs_cam["output_root"] = output_root

    camera_solution = solve_locs(
        sources.loc[:, ["x", "y", "flux"]],
        ra,
        dec,
        full_frame=False,
        raise_on_unsolved=False,
        **solve_locs_kwargs_cam,
    )

    if camera_solution.solved and camera_solution.stars is not None:
        camera = f"{telescope}-{camname[0]}"
        xy = camera_solution.stars.loc[:, ["field_x", "field_y"]].to_numpy()
        mf_locs, _ = ag_to_master_frame(camera, xy)
        camera_solution.stars.loc[:, ["x_master", "y_master"]] = mf_locs

    return {camname: camera_solution}


def wcs_from_single_cameras(
    solutions: dict[str, AstrometrySolution],
    telescope: str,
) -> WCS:
    """Determines the telescope pointing using individual AG frames.

    The main difference between this function and `.solve_from_files` is
    that here we solve each AG frame independently with astrometry.net and
    then use those solutions to generate a master frame WCS.

    Parameters
    ----------
    camera_solutions
        A dictionary of the astrometric solutions for each camera.
    telescope
        The telescope to which the exposures are associated.

    Returns
    -------
    wcs
        The wcs of the master frame astrometric solution.

    """

    nsolved = len([sol for sol in solutions.values() if sol.solved])
    if nsolved == 0:
        raise ValueError("No solutions found.")

    # Check for repeat images, in which East and West are almost identical. In
    # this case the coordinates of their central pixels will also be very close,
    # instead of being about a degree apart.
    wcs_east = solutions["east"].wcs if "east" in solutions else None
    wcs_west = solutions["west"].wcs if "west" in solutions else None
    if telescope != "spec" and wcs_east is not None and wcs_west is not None:
        east_cen = wcs_east.pixel_to_world(*XZ_AG_FRAME)
        west_cen = wcs_west.pixel_to_world(*XZ_AG_FRAME)
        ew_separation = east_cen.separation(west_cen).arcsec
        if ew_separation < 3000:
            raise ValueError("Found potential double image.")

    # Build a master frame WCS from the individual WCS solutions. We use astropy's
    # fit_wcs_from_points routine, where we use the coordinates of the stars
    # identified by astrometry.net and their corresponding xy pixels on the master
    # frame (as determined above with ag_to_master_frame).
    all_stars = pandas.concat(
        [ss.stars for ss in solutions.values() if ss.stars is not None]
    )
    if len(all_stars) == 0:
        raise ValueError("No solved fields found.")

    skycoords = SkyCoord(
        ra=all_stars.index_ra,
        dec=all_stars.index_dec,
        unit="deg",
        frame="icrs",
    )
    wcs = fit_wcs_from_points((all_stars.x_master, all_stars.y_master), skycoords)

    return wcs


def master_frame_to_ag(camera: str, locs: numpy.ndarray):
    """Transforms the star locations from the master frame
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

    # Transform MF --> AG. See Single_Sensor_Solve doc
    rsLoc = numpy.dot(M, (locs.T - rOff - off)) + rOff

    # Return calculated positions (need to transpose for standard layout)
    return rsLoc.T


def calculate_guide_offset(
    sources: pandas.DataFrame,
    telescope: str,
    reference_sources: pandas.DataFrame,
    reference_wcs: WCS,
    max_separation: float = 5,
) -> tuple[tuple[float, float], tuple[float, float], float, pandas.DataFrame]:
    """Determines the guide offset by matching sources to reference images.

    Parameters
    ----------
    sources
        A list of sources extracted from the AG frames. Must contain a column
        ``camera`` with the camera associated with each source.
    telescope
        The telescope associated with these images.
    reference_sources
        As ``sources``, a list of extracted detections from the reference frames.
    reference_wcs
        The WCS of the master frame corresponding to the reference images.
    max_separation
        Maximum separation between reference and test sources, in pixels.

    Returns
    -------
    offset_pix
        A tuple of x/y pixel offsets, in arcsec. This is the offset to
        go from the measured sources to reference sources.
    offset_radec
        A tuple of RA and Dec offsets, in arcsec.
    separation
        The absolute separation between the reference frame and the
        new set of sources, in arcsec.
    matches
        A Pandas data frame with the matches between test and reference sources.

    """

    sources["camera"] = sources["camera"].astype(str)
    reference_sources["camera"] = reference_sources["camera"].astype(str)

    matches: pandas.DataFrame | None = None

    cameras = sources["camera"].unique()
    for camera in cameras:
        cam_ref_sources = reference_sources.loc[reference_sources["camera"] == camera]
        cam_sources = sources.loc[sources["camera"] == camera]

        if len(cam_ref_sources) == 0 or len(cam_sources) == 0:
            continue

        # Create a KDTree with the reference pixels.
        tree = KDTree(cam_ref_sources.loc[:, ["x", "y"]].to_numpy())

        # Find nearest neighbours
        dd, ii = tree.query(cam_sources.loc[:, ["x", "y"]].to_numpy())

        # Select valid matches.
        valid = dd < max_separation

        # Get the valid xy in the test and reference source lists.
        cam_matches = cam_sources.loc[valid, ["x", "y"]]
        cam_ref_matches = cam_ref_sources.iloc[ii[valid]]
        cam_ref_matches = cam_ref_matches.loc[:, ["x", "y"]]

        # Calculate MF coordinates for test and references pixels.
        camname = f"{telescope}-{camera[0]}"
        cam_matches_mf, _ = ag_to_master_frame(camname, cam_matches.to_numpy())
        cam_ref_matches_mf, _ = ag_to_master_frame(camname, cam_ref_matches.to_numpy())

        # Build a DF with the MF coordinates. Add the camera name and concatenate
        # to build a main DF with all the matches for both cameras.

        matches_array = numpy.hstack(
            (
                cam_matches,
                cam_ref_matches,
                cam_matches_mf,
                cam_ref_matches_mf,
            )
        )

        matches_df = pandas.DataFrame(
            matches_array,
            columns=[
                "x",
                "y",
                "xref",
                "yref",
                "x_mf",
                "y_mf",
                "xref_mf",
                "yref_mf",
            ],
        )
        matches_df["camera"] = camname

        if matches is None:
            matches = matches_df
        else:
            matches = pandas.concat([matches, matches_df])

    if matches is None:
        raise ValueError("No matches found.")

    # Determine offsets in xy.
    offset_x = matches.xref_mf - matches.x_mf
    offset_y = matches.yref_mf - matches.y_mf

    # Rotate to align with RA/Dec
    PC = reference_wcs.wcs.pc if reference_wcs.wcs.has_pc() else reference_wcs.wcs.cd
    offset_ra_d, offset_dec_d = PC @ numpy.array([offset_x, offset_y])

    # To arcsec.
    offset_ra_arcsec = numpy.mean(offset_ra_d * 3600)
    offset_dec_arcsec = numpy.mean(offset_dec_d * 3600)

    # Calculate separation. This is somewhat approximate but good enough
    # for small angles.
    sep_arcsec = numpy.sqrt(offset_ra_arcsec**2 + offset_dec_arcsec**2)

    return (
        (float(numpy.mean(offset_x)), float(numpy.mean(offset_y))),
        (offset_ra_arcsec, offset_dec_arcsec),
        sep_arcsec,
        matches,
    )


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
    return crota2 if crota2 > 0 else crota2 + 360
