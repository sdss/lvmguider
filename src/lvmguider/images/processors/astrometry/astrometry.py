#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
# Copied from lvmagp

from abc import ABCMeta, abstractmethod

from lvmagp.images import Image
from lvmagp.images.processor import ImageProcessor


__all__ = ["Astrometry"]


class Astrometry(ImageProcessor, metaclass=ABCMeta):
    """Base class for source astrometry."""

    @abstractmethod
    def __call__(self, image: Image) -> Image:
        """Find astrometric solution on given image.

        Parameters
        ----------
        image
            Image to analyse.

        Returns
        -------
        proc_image
            Processed image.

        """

        pass
