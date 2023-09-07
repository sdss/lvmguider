#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-09-07
# @Filename: process_all_lco.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from lvmguider import log
from lvmguider.coadd import process_all_spec_frames


AGCAM_PATH = pathlib.Path("/data/agcam")
SPECTRO_PATH = pathlib.Path("/data/spectro")


def process_all_lco():
    """Processes all the MJDs. Customised for LVM@LCO paths."""

    ag_mjd_paths = list(sorted(AGCAM_PATH.glob("601*")))

    for ag_mjd_path in ag_mjd_paths[::-1]:
        mjd = ag_mjd_path.absolute().name

        if (ag_mjd_path / "coadds").exists():
            log.warning(f"Skipping {ag_mjd_path!s} which has already been processed.")
            continue

        spectro_path = SPECTRO_PATH / str(mjd)

        log.info(f"Processing files in {spectro_path} ...")

        (ag_mjd_path / "coadds").mkdir(parents=True, exist_ok=True)
        log.start_file_logger(str(ag_mjd_path / "coadds" / f"coadds_{mjd}.log"))

        process_all_spec_frames(spectro_path)


if __name__ == "__main__":
    process_all_lco()
