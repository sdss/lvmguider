#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-18
# @Filename: transformations.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import pathlib
import warnings
from datetime import datetime

import numpy
import pandas
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points, pixel_to_skycoord

from lvmguider.astrometrynet import AstrometrySolution, astrometrynet_quick
from lvmguider.extraction import extract_marginal
from lvmguider.tools import get_proc_path


# Middle of an AG frame or full frame
XZ_FULL_FRAME = (2500.0, 1000.0)
XZ_AG_FRAME = (800.0, 550.0)


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
    index_paths: dict[int, str] | None = None,
    scales: dict[int, list[int]] | None = None,
    output_root: str | None = None,
    verbose=False,
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
        raise_on_unsolved=True,
        plot=True,
        output_root=output_root,
        **astrometrynet_kwargs,
    )

    if verbose:
        print("")
        print("Solution has match ? ", solution.solved)

    return solution


def solve_from_files(
    files: list[str],
    telescope: str,
    reextract_sources: bool = False,
    solve_locs_kwargs: dict = {},
):
    """Determines the telescope pointing from a set of AG frames.

    Parameters
    ----------
    files
        The AG FITS files to use determine the telescope pointing. Normally
        these are a pair of east/west camera frames, except for ``spec``.
    telescope
        The telescope to which these exposures are associated.
    reextract_sources
        Runs the source extraction algorithm again. If `False`, extraction is only
        done if a ``SOURCES`` extensions is not found in the files.
    solve_locs_kwargs
        A dictionary of kwargs to pass to `.solve_locs`.

    Returns
    -------
    wcs
        The wcs of the astrometric solution or `None` if no solution was found.
    locs
        The locations used to solve with astrometry.net. This is basically the
        extracted sources with two new ``x_master`` and ``y_master`` columns.

    """

    if len(files) == 0:
        raise ValueError("No files provided.")

    mf_sources: list[pandas.DataFrame] = []
    ra = 0.0
    dec = 0.0
    for file in files:
        header = fits.getheader(file)
        camname = header["CAMNAME"].lower()
        ra = header["RA"]
        dec = header["DEC"]

        if reextract_sources is False:
            try:
                sources = pandas.DataFrame(fits.getdata(file, "SOURCES"))
            except KeyError:
                warnings.warn("SOURCES ext not found. Extracting sources", UserWarning)
                return solve_from_files(
                    files,
                    telescope,
                    reextract_sources=True,
                    solve_locs_kwargs=solve_locs_kwargs,
                )
        else:
            hdul = fits.open(file)
            if "PROC" in hdul:
                data = hdul["PROC"].data
            else:
                data = hdul[0].data
            sources = extract_marginal(data)
            sources["camera"] = camname

        xy = sources[["x", "y"]].values

        camera = f"{telescope}-{camname[0]}"
        file_locs, _ = rot_shift_locs(camera, xy)
        sources.loc[:, ["x_master", "y_master"]] = file_locs
        mf_sources.append(sources)

    locs = pandas.concat(mf_sources)

    if "output_root" not in solve_locs_kwargs:
        # Generate root path for astrometry files.
        proc_path = get_proc_path(files[0])
        dirname = proc_path.parent
        proc_base = proc_path.name.replace(".fits", "")
        output_root = str(dirname / "astrometry" / proc_base)
        solve_locs_kwargs["output_root"] = output_root

    solution = solve_locs(
        locs[["x_master", "y_master", "flux"]],
        ra=ra,
        dec=dec,
        full_frame=True,
        **solve_locs_kwargs,
    )

    return solution, locs


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


def wcs_from_single_cameras(
    files: list[str | pathlib.Path],
    telescope: str | None = None,
    method: str = "astropy",
    reextract_sources: bool = False,
    astrometrynet_params: dict = {},
):
    if len(files) == 0:
        raise ValueError("No files provided.")

    solutions: dict[str, AstrometrySolution] = {}

    for file in files:
        file = pathlib.Path(file).absolute()
        header = fits.getheader(file)
        camname = header["CAMNAME"].lower()
        ra = header["RA"]
        dec = header["DEC"]
        telescope = telescope or header["TELESCOP"].lower()

        if reextract_sources is False:
            try:
                sources = pandas.DataFrame(fits.getdata(file, "SOURCES"))
            except KeyError:
                warnings.warn("SOURCES ext not found. Extracting sources", UserWarning)
                return wcs_from_single_cameras(
                    files,
                    telescope=telescope,
                    reextract_sources=True,
                    astrometrynet_params=astrometrynet_params,
                )
        else:
            hdul = fits.open(file)
            if "PROC" in hdul:
                data = hdul["PROC"].data
            else:
                data = hdul[0].data
            sources = extract_marginal(data)
            sources["camera"] = camname

        if "output_root" not in astrometrynet_params:
            # Generate root path for astrometry files.
            basename = file.name.replace(".fits", "")
            output_root = str(file.parent / "astrometry" / basename)
            astrometrynet_params["output_root"] = output_root

        camera_solution = solve_locs(
            sources[["x", "y", "flux"]],
            ra,
            dec,
            full_frame=False,
            **astrometrynet_params,
        )

        # Calculate master frame coordinates for the pixels of the stars
        # identified by astrometry.net.
        if camera_solution.solved and camera_solution.stars is not None:
            camera = f"{telescope}-{camname[0]}"
            file_locs, _ = rot_shift_locs(
                camera,
                camera_solution.stars.loc[:, ["field_x", "field_y"]].to_numpy(),
            )
            camera_solution.stars.loc[:, ["x_master", "y_master"]] = file_locs

        solutions[camname] = camera_solution

    if len(solutions) == 0:
        raise ValueError("No solutions found.")

    # Check for repeat images, in which East and West are almost identical. In
    # this case the coordinates of their central pixels will also be very close,
    # instead of being about a degree apart.
    wcs_east = solutions["east"].wcs
    wcs_west = solutions["west"].wcs
    if telescope != "spec" and wcs_east is not None and wcs_west is not None:
        east_cen = wcs_east.pixel_to_world(*XZ_AG_FRAME)
        west_cen = wcs_west.pixel_to_world(*XZ_AG_FRAME)
        ew_separation = east_cen.separation(west_cen).arcsec
        if ew_separation < 3000:
            raise ValueError("Found potential double image.")

    # Build a master frame WCS from the individual WCS solutions. We implement two
    # methods. Astropy uses the fit_wcs_from_points routine, where we use the
    # coordinates of the stars identified by astrometry.net and their corresponding
    # xy pixels on the master frame (as determined above with rot_shift_locs).
    # Tom's method manually builds the WCS using the individual WCS solutions.
    if method == "astropy":
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

    elif method == "tom":
        if len(solutions) != 2:
            raise ValueError("Tom's method requires two cameras.")

        west_wcs = solutions["west"].wcs
        east_wcs = solutions["east"].wcs

        assert west_wcs and east_wcs, "Found unsolved fields."

        west_sky = west_wcs.pixel_to_world(*XZ_AG_FRAME)

        # RA, Dec of central pixel of West AG Camera
        CRVAL = numpy.array([west_sky.ra.deg, west_sky.dec.deg])  # type:ignore
        # Pixel location in MF of central px of W camera
        CRPIX = rot_shift_locs(f"{telescope}-w", numpy.array([XZ_AG_FRAME]))[0][0]

        deg_per_pix = 1.0089 / 3600.0  # Pixel scale in degrees / pixel
        CDELT = numpy.array([-1.0 * deg_per_pix, deg_per_pix])  # CDELT entity for WCS

        # Derive position angle of field
        # West point in AG frame
        west_PA = master_frame_to_ag(f"{telescope}-w", numpy.array([[700.0, 1000.0]]))
        # and East point
        east_PA = master_frame_to_ag(f"{telescope}-e", numpy.array([[4300.0, 1000.0]]))

        # Get sky coords of these pixels
        wPA_sky = pixel_to_skycoord(west_PA[0][0], west_PA[0][1], west_wcs)
        ePA_sky = pixel_to_skycoord(east_PA[0][0], east_PA[0][1], east_wcs)

        # This is the angle we need
        west_pa = wPA_sky.position_angle(ePA_sky).degree  # type: ignore
        PA = numpy.radians((west_pa + 90.0) % 360)

        # Rotation matrix
        PC = numpy.array(
            [
                [numpy.cos(PA), -numpy.sin(PA)],
                [numpy.sin(PA), numpy.cos(PA)],
            ]
        )

        # Assemble WCS
        wcs = WCS(naxis=2)  # Create new WCS entity
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # Set CTYPE
        wcs.wcs.crpix = CRPIX  # Reference pixel indices
        wcs.wcs.crval = CRVAL  # Reference pixel RA/Dec
        wcs.wcs.cdelt = CDELT  # Pixel scale in deg/px
        wcs.wcs.pc = PC  # Rotation matrix

    else:
        raise ValueError("Invalid method.")

    return wcs, solutions


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
