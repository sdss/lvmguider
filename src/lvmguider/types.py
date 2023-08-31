#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-08-30
# @Filename: types.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import nptyping as npt


# Array types
ARRAY_2D_U16 = npt.NDArray[npt.Shape["*, *"], npt.UInt16]
ARRAY_2D_F32 = npt.NDArray[npt.Shape["*, *"], npt.Float32]
ARRAY_2D_F64 = npt.NDArray[npt.Shape["*, *"], npt.Float64]

ARRAY_1D_F32 = npt.NDArray[npt.Shape["*"], npt.Float32]
