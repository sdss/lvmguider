#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
# Copied from lvmagp

from lvmguider.images import Image

from .background import Background


class DarkImageBackground(Background):
    """Base class for background."""

    def __init__(self, filename: str | None = None):
        """Initializes a wrapper."""

    def __call__(self, image: Image, **kwargs) -> Image:
        """return given image substracted with fits dark image.

        Args:
            image: Image.

        Returns:
            Background in float.
        """
        ...


__all__ = ["DarkImageBackground"]
