#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
# Copied from lvmagp

import sep

from lvmguider.images import Image

from .background import Background


class SepBackground(Background):
    """Base class for background."""

    __module__ = "lvmguider.images.processors.background"

    def __init__(self, **kwargs):
        """Initializes a wrapper.

        https://sep.readthedocs.io/en/v1.1.x/api/sep.Background.html

        Parameters:

        mask : 2-d ndarray, optional

            Mask array, optional

        maskthresh : float, optional

            Mask threshold. This is the inclusive upper limit on the mask value in order
            for the corresponding pixel to be unmasked. For boolean arrays, False and
            True are interpreted as 0 and 1, respectively. Thus, given a threshold of
            zero, True corresponds to masked and False corresponds to unmasked.

        bw, bh : int, optional

            Size of background boxes in pixels. Default is 64.

        fw, fh : int, optional

            Filter width and height in boxes. Default is 3.

        fthresh : float, optional

            Filter threshold. Default is 0.0.

        """

        self.kwargs = kwargs

    def __call__(self, image: Image, **kwargs) -> Image:
        """Return given image substracted with fits dark image.

        Args:
            image: Image.

        Returns:
            Image with subtracted background in float.
        """

        return sep.Background(image.data.astype(float), **{**self.kwargs, **kwargs})


__all__ = ["SepBackground"]
