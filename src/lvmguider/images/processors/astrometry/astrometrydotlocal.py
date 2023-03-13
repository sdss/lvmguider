#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
# Copied from lvmagp

import warnings

import astrometry
from astropy.wcs import WCS

from lvmguider.images import Image

from .astrometry import Astrometry


__all__ = ["AstrometryDotLocal"]


class AstrometryDotLocal(Astrometry):
    """Perform astrometry using python astrometry.net"""

    solver = None

    def __init__(
        self,
        source_count: int = 42,
        radius: float = 3.0,
        cache_directory: str = "/data/astrometrynet",
        scales={5, 6},
        exceptions: bool = True,
    ):
        """Init new astronomy.net processor.

        Parameters
        ----------
        source_count
            Number of sources to send.
        radius
            Radius to search in.
        exceptions
            Whether to raise Exceptions.

        """

        self.source_count = source_count
        self.radius = radius
        self.exceptions = exceptions

        if not self.solver:
            self.solver = astrometry.Solver(
                astrometry.series_5200.index_files(
                    cache_directory=cache_directory,
                    scales=scales,
                )
            )

    def source_solve_default(self, image: Image):
        """Solve an image using astrometry.net."""

        if not self.solver:
            raise ValueError("Solver not defined")

        solution = self.solver.solve(
            stars=image.catalog["x", "y"],
            size_hint=astrometry.SizeHint(
                lower_arcsec_per_pixel=0.9,
                upper_arcsec_per_pixel=1.1,
            ),
            position_hint=astrometry.PositionHint(
                ra_deg=image.header["RA"],
                dec_deg=image.header["DEC"],
                radius_deg=self.radius,
            ),
            solution_parameters=astrometry.SolutionParameters(
                logodds_callback=lambda logodds_list: astrometry.Action.STOP,
            ),
        )

        if solution.has_match():
            wcs = WCS(solution.best_match().wcs_fields)
            return wcs

    def __call__(self, image: Image, sort_by="peak") -> Image:
        """Find astrometric solution on given image.

        Parameters
        ----------
        image
            Image to analyse.

        """

        # copy image
        img = image.copy()

        # Get catalog
        if img.catalog is None:
            warnings.warn("No catalog found in image.", UserWarning)
            return img

        cat = img.catalog["x", "y", "flux", "peak"]

        # nothing?
        if cat is None or len(cat) < 3:
            img.header["WCSERR"] = 1
            return img

        # sort it, remove saturated stars and take N brightest sources
        cat.sort(sort_by)
        cat.reverse()
        cat = cat[cat["peak"] < 60000]
        cat = cat[: self.source_count]

        img.catalog = cat

        img.astrometric_wcs = self.source_solve_default(img)

        # finished
        return img
