#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-18
# @Filename: maskbits.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from enum import Flag


__all__ = ["GuiderStatus"]


class GuiderStatus(Flag):
    """Maskbits with the guider status."""

    IDLE = 1 << 0
    ACQUIRING = 1 << 1
    GUIDING = 1 << 2
    EXPOSING = 1 << 3
    PROCESSING = 1 << 4
    CORRECTING = 1 << 5
    STOPPING = 1 << 6
    FAILED = 1 << 7
    WAITING = 1 << 8
    DRIFTING = 1 << 9

    NON_IDLE = (
        ACQUIRING
        | GUIDING
        | EXPOSING
        | PROCESSING
        | CORRECTING
        | STOPPING
        | WAITING
        | DRIFTING
    )

    def get_names(self):
        """Returns a list of active bit names."""

        return [bit.name for bit in GuiderStatus if self & bit and bit.name]

    def __repr__(self):
        return str(" | ".join(self.get_names()))
