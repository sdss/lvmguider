#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
# Copied from lvmagp

from abc import ABCMeta, abstractmethod

from lvmguider.images import Image


__all__ = ["ImageProcessor"]


class ImageProcessor(object, metaclass=ABCMeta):
    """Image processor."""

    @abstractmethod
    async def __call__(self, image: Image) -> Image:
        """Processes an image.

        Parameters
        ----------
        image
            Image to process.

        Returns
        -------
        image
            Processed image.

        """

        pass

    async def reset(self) -> None:
        """Resets state of image processor"""

        pass
