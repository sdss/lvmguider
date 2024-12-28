#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-01-17
# @Filename: test_lvmguider.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


from pytest_mock import MockerFixture

from lvmguider import config
from lvmguider.actor.actor import LVMGuiderActor
from lvmguider.guider import Guider


def test_placeholder(mocker: MockerFixture):
    command = mocker.MagicMock()
    command.actor.config = config

    guider = Guider(command, (0.0, 10.0, 0.0))

    assert guider.command is not None


async def test_actor(actor: LVMGuiderActor):
    assert isinstance(actor, LVMGuiderActor)
    assert actor.name == "lvm.sci.guider"
