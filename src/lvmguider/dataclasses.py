#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-08-30
# @Filename: dataclasses.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import nptyping as npt
import numpy
import pandas
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from lvmguider import config
from lvmguider.transformations import get_crota2
from lvmguider.types import ARRAY_2D_F32


__all__ = ["CameraSolution", "GuiderSolution", "FrameData"]


@dataclass
class CameraSolution:
    """A camera solution, including the determined WCS."""

    frameno: int
    camera: str
    path: pathlib.Path
    sources: pandas.DataFrame
    wcs: WCS | None = None
    matched: bool = False
    zero_point: float = numpy.nan
    pa: float = numpy.nan
    ref_frame: pathlib.Path | None = None
    solution_mode: str = "none"

    def post_init(self):
        if numpy.isnan(self.pa) and self.wcs is not None:
            self.pa = get_crota2(self.wcs)

    @property
    def solved(self):
        """Was the frame solved?"""

        return self.wcs is not None


@dataclass
class GuiderSolution:
    """A class to hold an astrometric solution determined by the guider.

    This class abstracts an astrometric solution regardless of whether it was
    determined using astrometry.net or Gaia sources.

    """

    frameno: int
    solutions: list[CameraSolution]
    guide_pixel: npt.NDArray[npt.Shape[2], npt.Float32]
    mf_wcs: WCS | None = None
    pa: float = numpy.nan
    ra_off: float = numpy.nan
    dec_off: float = numpy.nan
    axis0_off: float = numpy.nan
    axis1_off: float = numpy.nan
    pa_off: float = numpy.nan
    correction_applied: bool = False
    correction: list[float] = field(default_factory=lambda: [0.0, 0.0])

    def post_init(self):
        self.sources = pandas.concat([cs.sources for cs in self.solutions], axis=0)

        zps = self.sources.loc[:, "zp"].copy().dropna()
        self.zero_point = zps.median()

        if numpy.isnan(self.pa) and self.mf_wcs is not None:
            self.pa = get_crota2(self.mf_wcs)

    def __getitem__(self, key: str):
        """Returns the associated camera solution."""

        for solution in self.solutions:
            if solution.camera == key:
                return solution

        raise KeyError(f"Invalid camera name {key!r}.")

    @property
    def pointing(self) -> npt.NDArray[npt.Shape[2], npt.Float64]:
        """Returns the telescope pointing."""

        if not self.mf_wcs:
            return numpy.array([numpy.nan, numpy.nan])

        skyc = self.mf_wcs.pixel_to_world(*config["xz_full_frame"])
        return numpy.array([skyc.ra.deg, skyc.dec.deg])

    @property
    def solved(self):
        """Was the frame solved?"""

        return self.mf_wcs is not None

    @property
    def cameras(self):
        """Returns a list of cameras."""

        return [cs.camera for cs in self.solutions]

    @property
    def separation(self):
        """Returns the separation between field centre and telescope pointing."""

        return numpy.hypot(self.ra_off, self.dec_off)

    @property
    def n_cameras_solved(self):
        """Returns the number of cameras solved."""

        return len([cs for cs in self.solutions if cs.solved])

    def all_cameras_solved(self):
        """Returns `True` if all the cameras have solved."""

        return all([cs.solved for cs in self.solutions])


@dataclass
class FrameData:
    """Data associated with a frame."""

    file: pathlib.Path
    raw_header: fits.Header
    frameno: int
    date_obs: Time
    sjd: int
    camera: str
    telescope: str
    exptime: float
    image_type: str
    kmirror_drot: float
    focusdt: float
    proc_header: fits.Header | None = None
    proc_file: pathlib.Path | None = None
    guide_error: float | None = None
    fwhm_median: float | None = None
    fwhm_std: float | None = None
    guide_mode: str = "guide"
    stacked: bool = False
    wcs: WCS | None = None
    wcs_mode: str | None = None
    data: ARRAY_2D_F32 | None = None
