#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-09-10
# @Filename: load_all_to_db.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import re

from lvmguider import log
from lvmguider.coadd import coadd_to_database


AGCAM_PATH = pathlib.Path("/data/agcam")


def load_all_to_db():
    """Processes all the MJDs and loads co-added data to the database."""

    log.sh.setLevel(5)

    ag_mjd_paths = list(sorted(AGCAM_PATH.glob("601*")))

    for ag_mjd_path in ag_mjd_paths[::-1]:
        if not (ag_mjd_path / "coadds").exists():
            continue

        coadd_files = (ag_mjd_path / "coadds").glob("lvm.*.coadd_s*.fits")
        for file in coadd_files:
            log.info(f"Loading file {file} to database")

            spec_frameno_match = re.search("s([0-9]{8})", file.name)
            if spec_frameno_match is None:
                log.error(f"Cannot determine spec frame number for {file}")
                continue

            coadd_to_database(file, exposure_no=int(spec_frameno_match.group(1)))


if __name__ == "__main__":
    load_all_to_db()
