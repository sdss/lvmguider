# encoding: utf-8

import pathlib

from sdsstools import Configuration, get_package_version


NAME = "lvmguider"


# package name should be pip package name
__version__ = get_package_version(path=__file__, package_name=NAME)


# Temporary config. Default to sci. This will be updated by the actor.
config = Configuration(pathlib.Path(__file__).parent / "etc/lvm.sci.guider.yml")
