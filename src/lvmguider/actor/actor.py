#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-03-10
# @Filename: actor.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from clu.actor import AMQPActor


__all__ = ["LVMGuiderActor"]


class LVMGuiderActor(AMQPActor):
    """The ``lvmguider`` actor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.telescope: str
        self.telescope = self.config.get("telescope", None) or self.name.split(".")[1]
