#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-08-30
# @Filename: dataclasses.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import warnings
from dataclasses import dataclass, field

from typing import TYPE_CHECKING, Literal

import nptyping as npt
import numpy
import pandas
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
from packaging.version import Version

from sdsstools.time import get_sjd

from lvmguider import config, log
from lvmguider.tools import dataframe_from_model, get_frameno
from lvmguider.transformations import get_crota2


if TYPE_CHECKING:
    from pandas._libs.missing import NAType

    from lvmguider.types import ARRAY_2D_F32


__all__ = ["CameraSolution", "GuiderSolution", "FrameData"]


WCS_MODE_T = Literal["none"] | Literal["astrometrynet"] | Literal["gaia"]


@dataclass(kw_only=True)
class BaseSolution:
    """A base class for an object with an astrometric solution."""

    telescope: str
    wcs: WCS | None = None
    matched: bool = False

    def __post_init__(self):
        if self.sources is None:
            self.sources = dataframe_from_model("SOURCES")

    @property
    def solved(self):
        """Was the frame solved?"""

        return self.wcs is not None

    @property
    def pa(self):
        """Returns the position angle from the WCS."""

        if self.wcs is not None:
            return get_crota2(self.wcs)

        return numpy.nan

    @property
    def zero_point(self):
        """Returns the median zero point."""

        if self.sources is not None:
            if len(data := self.sources.zp.dropna()) > 0:
                return float(data.median())

        return numpy.nan

    @property
    def fwhm(self):
        """Returns the FWHM from the extracted sources."""

        if self.sources is not None:
            fwhms = self.sources.loc[self.sources.valid == 1].fwhm.dropna()
            if len(fwhms) == 0:
                return numpy.nan
            perc_25 = numpy.percentile(fwhms, 25)
            return float(perc_25)

        return numpy.nan

    @property
    def pointing(self) -> npt.NDArray[npt.Shape["2"], npt.Float64]:
        """Returns the camera pointing at its central pixel."""

        if not self.wcs:
            return numpy.array([numpy.nan, numpy.nan])

        skyc = self.wcs.pixel_to_world(*config["xz_ag_frame"])
        return numpy.array([skyc.ra.deg, skyc.dec.deg])


@dataclass(kw_only=True)
class CameraSolution(BaseSolution):
    """A camera solution, including the determined WCS."""

    frameno: int
    camera: str
    path: pathlib.Path
    date_obs: Time = field(default_factory=lambda: Time.now())
    sources: pandas.DataFrame | None = None
    ref_frame: pathlib.Path | None = None
    wcs_mode: WCS_MODE_T = "none"
    guider_version: Version = Version("0.0.0")
    hdul: fits.HDUList | None = None

    @classmethod
    def open(cls, file: str | pathlib.Path):
        """Creates an instance from an ``lvm.agcam`` file."""

        file = pathlib.Path(file)
        hdul = fits.open(str(file))

        if "PROC" not in hdul:
            raise ValueError("HDU list does not have a PROC extension.")

        raw = hdul["RAW"].header
        proc = hdul["PROC"].header

        if "GUIDERV" not in proc:
            raise ValueError("Guider version not found.")

        guiderv = Version(proc["GUIDERV"])
        if guiderv < Version("0.4.0a0") and guiderv != Version("0.0.0"):
            raise ValueError(
                "The file was generated with an unsupported version of lvmguider."
            )

        sources: pandas.DataFrame | None = None
        if proc["SOURCESF"]:
            dirname = file.parent
            sources = pandas.read_parquet(
                dirname / proc["SOURCESF"],
                dtype_backend="pyarrow",
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FITSFixedWarning)
            solved = proc["SOLVED"]
            wcs_mode = proc["WCSMODE"].lower()
            wcs = WCS(proc) if solved and wcs_mode != "none" else None

            # Fix cases when WCSMODE is not "none" but the WCS is not valid.
            if wcs is not None and not wcs.is_celestial:
                log.warning(f"Invalid WCS found for {file!s}.")
                wcs = None

        ref_file = proc["REFFILE"]
        ref_frame = pathlib.Path(ref_file) if ref_file is not None else None

        return CameraSolution(
            frameno=get_frameno(file),
            telescope=raw["TELESCOP"],
            camera=raw["CAMNAME"],
            date_obs=Time(raw["DATE-OBS"], format="isot"),
            path=file.absolute(),
            sources=sources,
            wcs=wcs,
            wcs_mode=wcs_mode,
            matched=len(sources.ra.dropna()) > 0 if sources is not None else False,
            ref_frame=ref_frame,
            guider_version=guiderv,
            hdul=hdul,
        )

    @property
    def is_focus_sweep(self):
        """Returns whether the frame is part of a focus sweep."""

        proc_ext = (
            self.hdul["PROC"]
            if self.hdul
            and "PROC" in self.hdul
            and "ISFSWEEP" in self.hdul["PROC"].header
            else None
        )
        if proc_ext is None:
            log.warning(
                f"Cannot determine focus sweep status for {self.path!s}. "
                "Returning False."
            )
            return False

        return proc_ext.header["ISFSWEEP"] is True

    def to_framedata(self):
        """Returns a `.FrameData` instance."""

        raw_header = {}
        exptime = numpy.nan
        image_type = ""
        kmirror_drot = numpy.nan
        focusdt = numpy.nan
        airmass = numpy.nan
        data = None

        raw_ext = self.hdul["RAW"] if self.hdul and "RAW" in self.hdul else None
        if self.hdul is not None and raw_ext is not None:
            raw_header = dict(raw_ext.header)
            exptime = raw_ext.header["EXPTIME"]
            image_type = raw_ext.header["IMAGETYP"]
            kmirror_drot = raw_ext.header["KMIRDROT"]
            focusdt = raw_ext.header["FOCUSDT"]
            airmass = raw_ext.header["AIRMASS"]
            data = raw_ext.data

        return FrameData(
            frameno=self.frameno,
            path=self.path,
            raw_header=raw_header,
            date_obs=self.date_obs,
            sjd=get_sjd("LCO", self.date_obs.to_datetime()),
            camera=self.camera,
            telescope=self.telescope,
            exptime=exptime,
            airmass=airmass,
            image_type=image_type,
            kmirror_drot=kmirror_drot,
            focusdt=focusdt,
            stacked=False,
            reprocessed=False,
            data=data,
            solution=self,
        )


@dataclass(kw_only=True)
class GuiderSolution(BaseSolution):
    """A class to hold an astrometric solution determined by the guider.

    This class abstracts an astrometric solution regardless of whether it was
    determined using astrometry.net or Gaia sources.

    """

    frameno: int
    telescope: str
    solutions: list[CameraSolution]
    guide_pixel: npt.NDArray[npt.Shape["2"], npt.Float32]
    ra_field: float = numpy.nan
    dec_field: float = numpy.nan
    pa_field: float = numpy.nan
    ra_off: float = numpy.nan
    dec_off: float = numpy.nan
    pa_off: float = numpy.nan
    axis0_off: float = numpy.nan
    axis1_off: float = numpy.nan
    correction_applied: bool = False
    correction: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    guide_mode: str = "guide"
    guider_version: Version = Version("0.0.0")

    @property
    def sources(self):
        """Concatenates sources from the co-added solutions."""

        return pandas.concat(
            [cs.sources for cs in self.solutions if cs.sources is not None],
            axis=0,
        )

    def __getitem__(self, key: str):
        """Returns the associated camera solution."""

        for solution in self.solutions:
            if solution.camera == key:
                return solution

        raise KeyError(f"Invalid camera name {key!r}.")

    @property
    def pointing(self) -> npt.NDArray[npt.Shape["2"], npt.Float64]:
        """Returns the telescope pointing at boresight from the WCS."""

        if not self.wcs:
            return numpy.array([numpy.nan, numpy.nan])

        skyc = self.wcs.pixel_to_world(*config["xz_full_frame"])
        return numpy.array([skyc.ra.deg, skyc.dec.deg])

    @property
    def pixel_pointing(self) -> npt.NDArray[npt.Shape["2"], npt.Float64]:
        """Returns the telescope pointing at the full frame pixel from the WCS."""

        if not self.wcs:
            return numpy.array([numpy.nan, numpy.nan])

        skyc = self.wcs.pixel_to_world(*self.guide_pixel)
        return numpy.array([skyc.ra.deg, skyc.dec.deg])

    @property
    def solved(self):
        """Was the frame solved?"""

        return self.wcs is not None

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

    @classmethod
    def open(cls, file: str | pathlib.Path, dirname: str | pathlib.Path | None = None):
        """Creates an instance from an ``lvm.guider`` file."""

        file = pathlib.Path(file)
        hdul = fits.open(str(file))

        guider_data = hdul["GUIDERDATA"].header

        if "GUIDERV" not in guider_data:
            raise ValueError("Guider version not found.")

        guiderv = Version(guider_data["GUIDERV"])
        if guiderv < Version("0.4.0a0") and guiderv != Version("0.0.0"):
            raise ValueError(
                "The file was generated with an unsupported version of lvmguider."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FITSFixedWarning)
            wcs = WCS(guider_data) if guider_data["SOLVED"] else None

        solutions: list[CameraSolution] = []
        for key in ["FILEEAST", "FILEWEST"]:
            if guider_data[key] is not None:
                dirname = pathlib.Path(dirname or guider_data["DIRNAME"] or file.parent)
                solutions.append(CameraSolution.open(dirname / guider_data[key]))

        if "XFFPIX" in guider_data:
            guide_pixel = numpy.array([guider_data["XFFPIX"], guider_data["ZFFPIX"]])
        elif "XMFPIX" in guider_data:
            # From when we called it master frame.
            guide_pixel = numpy.array([guider_data["XMFPIX"], guider_data["ZMFPIX"]])
        else:
            log.warning("Missing pixel information. Default to central pixel.")
            guide_pixel = config["xz_full_frame"]

        return GuiderSolution(
            frameno=get_frameno(file),
            telescope=guider_data["TELESCOP"],
            solutions=solutions,
            guide_pixel=guide_pixel,
            wcs=wcs,
            ra_off=guider_data["OFFRAMEA"] or numpy.nan,
            dec_off=guider_data["OFFDEMEA"] or numpy.nan,
            pa_off=guider_data["OFFPAMEA"] or numpy.nan,
            axis0_off=guider_data["OFFA0MEA"] or numpy.nan,
            axis1_off=guider_data["OFFA1MEA"] or numpy.nan,
            correction_applied=guider_data["CORRAPPL"],
            correction=[
                guider_data["AX0CORR"],
                guider_data["AX1CORR"],
                guider_data["PACORR"],
            ],
            guider_version=guiderv,
        )


@dataclass(kw_only=True)
class FrameData:
    """Data associated with a frame."""

    frameno: int
    path: pathlib.Path
    raw_header: dict
    date_obs: Time
    sjd: int
    camera: str
    telescope: str
    exptime: float
    airmass: float
    image_type: str
    kmirror_drot: float
    focusdt: float
    solution: CameraSolution
    stacked: bool = False
    reprocessed: bool = False
    data: ARRAY_2D_F32 | None = None

    @property
    def sources(self):
        """Accessor to the camera solution sources."""

        return self.solution.sources

    def to_dataframe(self):
        """Returns a dataframe with the `.FrameData` information."""

        df = dataframe_from_model("FRAMEDATA")
        pointing = self.solution.pointing

        new_row = dict(
            frameno=self.frameno,
            mjd=self.sjd,
            date_obs=self.date_obs.isot,
            camera=self.camera,
            telescope=self.telescope,
            exptime=numpy.float32(self.exptime),
            kmirror_drot=numpy.float32(self.kmirror_drot),
            focusdt=numpy.float32(self.focusdt),
            fwhm=numpy.float32(self.solution.fwhm),
            ra=numpy.float64(pointing[0]),
            dec=numpy.float64(pointing[1]),
            pa=numpy.float32(self.solution.pa),
            airmass=numpy.float32(self.airmass),
            zero_point=numpy.float32(self.solution.zero_point),
            stacked=bool(self.stacked),
            solved=bool(self.solution.solved),
            wcs_mode=self.solution.wcs_mode,
            is_focus_sweep=self.solution.is_focus_sweep,
        )
        df.loc[0, list(new_row)] = list(new_row.values())

        return df


class CoAddWarningsMixIn:
    """Methods to calculate warnings for co-added data classes."""

    zero_point: float
    fwhm: float
    sources: pandas.DataFrame

    def transp_warning(self):
        """Determines whether to raise a transparency warning."""

        # We raise a transparency warning when the median ZP is > some nominal plus
        # a number of magnitudes. The nominal ZP corresponds to a photometric night.
        nominal_zp = config["coadds"]["warnings"]["nominal_zp"]
        zp_overmag_warning = config["coadds"]["warnings"]["zp_overmag_warning"]

        if numpy.isnan(self.zero_point):
            return True
        else:
            return self.zero_point > (nominal_zp + zp_overmag_warning)

    def fwhm_warning(self):
        """Determines whether to raise a transparency warning."""

        # Similarly, for FWHM we raise a warning if the co-added FWHM
        # is > factor * median FWHM. This may indicate that there was a jump
        # during guiding.
        if numpy.isnan(self.zero_point):
            return True
        else:
            fwhm_factor_warn = config["coadds"]["warnings"]["fwhm_factor_warning"]
            sources_fwhm = self.sources.loc[self.sources.valid == 1, "fwhm"].dropna()
            if len(sources_fwhm) == 0:
                return False
            perc_25 = numpy.percentile(sources_fwhm, 25)
            return self.fwhm > perc_25 * fwhm_factor_warn


@dataclass(kw_only=True)
class CoAdd_CameraSolution(CameraSolution, CoAddWarningsMixIn):
    """A camera solution for a co-added frame."""

    path: pathlib.Path = pathlib.Path("")
    coadd_image: ARRAY_2D_F32 | None = None
    frames: list[FrameData] | None = None
    sources: pandas.DataFrame | None = None
    sigmaclip: bool = False
    sigmaclip_sigma: float | None = None

    def airmass(self):
        """Returns the average airmass of the guide sequence."""

        fd = self.frame_data()

        return float(numpy.mean(fd.airmass))

    def frame_data(self):
        """Returns a Pandas data frame from a list of `.FrameData`."""

        if self.frames is None:
            return dataframe_from_model("FRAMEDATA")

        fd_dfs: list[pandas.DataFrame] = []
        for fd in self.frames:
            fd_dfs.append(fd.to_dataframe())

        df = pandas.concat(fd_dfs, axis=0, ignore_index=True)
        df = df.sort_values(["frameno", "camera"])
        df.reset_index(drop=True, inplace=True)

        return df


@dataclass(kw_only=True)
class GlobalSolution(BaseSolution, CoAddWarningsMixIn):
    """A global solution from co-added frames."""

    coadd_solutions: list[CoAdd_CameraSolution]
    guider_solutions: list[GuiderSolution]
    path: pathlib.Path = pathlib.Path("")
    matched: bool = True

    @property
    def n_cameras_solved(self):
        """Returns the number of cameras solved."""

        return len([cs for cs in self.coadd_solutions if cs.solved])

    @property
    def sources(self):
        """Concatenates sources from the co-added solutions."""

        return pandas.concat(
            [cs.sources for cs in self.coadd_solutions if cs.sources is not None],
            axis=0,
        )

    @property
    def pointing(self) -> npt.NDArray[npt.Shape["2"], npt.Float64]:
        """Returns the camera pointing at its central pixel."""

        if not self.wcs:
            return numpy.array([numpy.nan, numpy.nan])

        skyc = self.wcs.pixel_to_world(*config["xz_full_frame"])
        return numpy.array([skyc.ra.deg, skyc.dec.deg])

    def airmass(self):
        """Returns the average airmass of the guide sequence."""

        fd = self.frame_data()

        return float(numpy.mean(fd.airmass))

    def frame_data(self):
        """Concatenates the frame data and returns a Pandas data frame."""

        frame_data = pandas.concat([cs.frame_data() for cs in self.coadd_solutions])
        frame_data.sort_values(["frameno", "camera"], inplace=True)

        return frame_data

    def guider_data(self):
        """Constructs a data frame with guider data."""

        df = dataframe_from_model("GUIDERDATA_FRAME")

        for ii, gs in enumerate(self.guider_solutions):
            pa = get_crota2(gs.wcs) if gs.wcs is not None else numpy.nan
            pointing = gs.pointing

            date_obs: str = ""
            mjd: int | NAType = pandas.NA

            if len(gs.solutions) > 0:
                date_obs = gs.solutions[0].date_obs.isot
                mjd = get_sjd("LCO", gs.solutions[0].date_obs.to_datetime())
                airmass = gs.solutions[0].to_framedata().airmass

            new_row = dict(
                frameno=gs.frameno,
                telescope=gs.telescope,
                mjd=mjd,
                date_obs=date_obs,
                fwhm=numpy.float32(gs.fwhm),
                pa=numpy.float32(pa),
                zero_point=numpy.float32(gs.zero_point),
                solved=bool(gs.solved),
                n_cameras_solved=int(gs.n_cameras_solved),
                guide_mode=gs.guide_mode,
                x_ff_pixel=numpy.float32(gs.guide_pixel[0]),
                z_ff_pixel=numpy.float32(gs.guide_pixel[1]),
                ra=numpy.float64(pointing[0]),
                dec=numpy.float64(pointing[1]),
                airmass=numpy.float32(airmass),
                ra_field=numpy.float64(gs.ra_field),
                dec_field=numpy.float64(gs.dec_field),
                pa_field=numpy.float32(gs.pa_field),
                ra_off=numpy.float32(gs.ra_off),
                dec_off=numpy.float32(gs.dec_off),
                pa_off=numpy.float32(gs.pa_off),
                axis0_off=numpy.float32(gs.axis0_off),
                axis1_off=numpy.float32(gs.axis1_off),
                applied=bool(gs.correction_applied),
                ax0_applied=numpy.float32(gs.correction[0]),
                ax1_applied=numpy.float32(gs.correction[1]),
                rot_applied=numpy.float32(gs.correction[2]),
            )
            df.loc[ii, list(new_row)] = list(new_row.values())

        df = df.sort_values(["frameno"])
        df.reset_index(drop=True, inplace=True)

        return df
